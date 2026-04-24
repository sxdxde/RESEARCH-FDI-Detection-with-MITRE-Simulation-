"""
EV FDI Real-Time Detection Demo — FastAPI + WebSocket backend
═════════════════════════════════════════════════════════════
Detection pipeline
──────────────────
  Primary:   CUSUM control chart  (Page 1954)
               S_t = max(0, S_{t-1} + eoe_t - k)
               k = μ_clean (mean clean EoE)
               Alarm when S_t > h = H_sigma * σ_clean
             This accumulates evidence over successive steps — critical for
             subtle attacks (θ<10 min) where individual samples never exceed
             the IQR bound.

  Secondary: IQR sliding window  (paper baseline)
               Flag spike if eoe > Q3 + k_iqr*IQR
               Alarm if q=3 consecutive spikes
             Only fires on strong individual anomalies (θ≥20 min).

  Combined: detected = CUSUM alarm  OR  IQR alarm
            The CUSUM is far more sensitive; it is what rescues recall at
            low θ.  IQR adds specificity for burst attacks.

  Sensitivity slider (1–10):  scales both alarm thresholds by
               exp((5 – s) × 0.3)  so range ≈ 15:1

Model selection
───────────────
  NARX:    single hidden-layer MLP, flat lag-vector input
  BiLSTM:  2-layer bidirectional LSTM + additive attention
  Both are pre-loaded at startup; switching replays from t=0 with
  the chosen model's calibrated (μ, σ) thresholds.

Usage
─────
    python -m src.realtime.run   (from narx_ev_fdi/ project root)
"""

import asyncio
import json
import math
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from src.data.dataset            import build_datasets          # noqa: E402
from src.models.narx             import NARXNet                 # noqa: E402
from src.models.attention_bilstm import AttentionBiLSTM         # noqa: E402
from src.eval.evaluate           import compute_iqr_bounds      # noqa: E402
from src.attack.mitre_fdi        import MITREFDIAttacker        # noqa: E402

STATIC = Path(__file__).parent / "static"

# ─────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────
app = FastAPI(title="EV FDI Real-Time Demo")

# ─────────────────────────────────────────────────────────────────
# Model assets (populated at startup)
#
# _models["narx"]   / _models["bilstm"]  each hold:
#   y_true    : (N,)  unscaled true kWh
#   y_pred    : (N,)  clean model predictions
#   iqr_ub    : IQR upper bound  (k=3, more sensitive than paper k=5)
#   cusum_k   : mean clean EoE  — CUSUM slack parameter
#   cusum_h   : 5 × std clean EoE — CUSUM alarm level at default sensitivity
#   baseline  : mean positive y_true (for attack injection scaling)
#   n         : len(y_true)
#   name      : display label
# ─────────────────────────────────────────────────────────────────
_models: dict         = {}
_READY: bool          = False


def _calibrate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute detection thresholds from clean predictions."""
    eoe = np.abs(y_true - y_pred)
    _, iqr_ub   = compute_iqr_bounds(eoe, k=3.0)   # k=3 is more sensitive than paper k=5
    cusum_k     = float(np.mean(eoe))
    cusum_h     = float(5.0 * np.std(eoe))          # alarm at 5σ above clean mean
    pos         = y_true[y_true > 1e-6]
    baseline    = float(np.mean(pos)) if len(pos) else 0.05
    return dict(
        y_true=y_true, y_pred=y_pred,
        iqr_ub=iqr_ub, cusum_k=cusum_k, cusum_h=cusum_h,
        baseline=baseline, n=len(y_true),
    )


# ─────────────────────────────────────────────────────────────────
# Startup: load NARX + BiLSTM, calibrate thresholds
# ─────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def _startup() -> None:
    global _READY, _models

    import pickle
    PROC   = ROOT / "data"  / "processed"
    CKPT   = ROOT / "checkpoints"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[DEMO] Loading data …")
    df_train = pd.read_csv(PROC / "acn_train_clean.csv")
    df_estim = pd.read_csv(PROC / "acn_estim_clean.csv")
    for col in ["connectionTime", "doneChargingTime", "modifiedAt", "requestedDeparture"]:
        for df in (df_train, df_estim):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True)

    # ── 1. NARX ───────────────────────────────────────────────────
    print("[DEMO] Loading NARX model …")
    narx_data = build_datasets(df_train, df_estim, mx=2, my=2,
                               val_ratio=0.15, test_ratio=0.15,
                               batch_size=512, model_type="narx")
    sy_narx  = narx_data["scalers"]["y"]
    X_es_n   = narx_data["raw"]["X_estim_w"]
    y_es_n   = narx_data["raw"]["y_estim_w"]

    narx = NARXNet(input_size=narx_data["shapes"]["input_size"], hidden_size=10).to(device)
    narx.load_state_dict(torch.load(CKPT / "narx_best.pt", map_location=device))
    narx.eval()

    with torch.no_grad():
        p_n = narx(torch.tensor(X_es_n, dtype=torch.float32, device=device)).cpu().numpy().flatten()
    yt_n = sy_narx.inverse_transform(y_es_n.reshape(-1, 1)).flatten()
    yp_n = sy_narx.inverse_transform(p_n.reshape(-1, 1)).flatten()
    _models["narx"] = _calibrate(yt_n, yp_n)
    _models["narx"]["name"] = "NARX"

    # ── 2. Attention-BiLSTM ───────────────────────────────────────
    try:
        print("[DEMO] Loading Attention-BiLSTM model …")
        SEQ_LEN = 4
        bl_data = build_datasets(df_train, df_estim,
                                 val_ratio=0.15, test_ratio=0.15,
                                 batch_size=512, model_type="bilstm",
                                 seq_len=SEQ_LEN)
        n_feat = bl_data["shapes"]["n_features"]

        with open(CKPT / "bilstm_scalers.pkl", "rb") as f:
            bl_scalers = pickle.load(f)
        sy_bl = bl_scalers["y"]

        X_es_b = bl_data["raw"]["X_estim_w"]   # (N, seq_len, n_features)
        y_es_b = bl_data["raw"]["y_estim_w"]

        bilstm = AttentionBiLSTM(n_features=n_feat, seq_len=SEQ_LEN,
                                 hidden_size=128, num_layers=2, dropout=0.3).to(device)
        bilstm.load_state_dict(torch.load(CKPT / "bilstm_best.pt", map_location=device))
        bilstm.eval()

        with torch.no_grad():
            p_b = bilstm(
                torch.tensor(X_es_b, dtype=torch.float32, device=device)
            ).cpu().numpy().flatten()

        yt_b = sy_bl.inverse_transform(y_es_b.reshape(-1, 1)).flatten()
        yp_b = sy_bl.inverse_transform(p_b.reshape(-1, 1)).flatten()
        _models["bilstm"] = _calibrate(yt_b, yp_b)
        _models["bilstm"]["name"] = "Attention-BiLSTM"
    except Exception as e:
        print(f"[DEMO] BiLSTM failed to load ({e}) — NARX only")

    _READY = True

    for key, m in _models.items():
        print(f"[DEMO] {m['name']:20s}  N={m['n']:>8,}  "
              f"IQR_UB={m['iqr_ub']:.4f}  "
              f"CUSUM_k={m['cusum_k']:.4f}  CUSUM_h={m['cusum_h']:.4f}")
    print(f"[DEMO] Dashboard → http://localhost:8000")


# ─────────────────────────────────────────────────────────────────
# Frontend
# ─────────────────────────────────────────────────────────────────
@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(STATIC / "index.html"))


# ─────────────────────────────────────────────────────────────────
# WebSocket simulation
# ─────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()

    # Wait up to 120 s for models to finish loading before proceeding
    for _ in range(240):
        if _READY:
            break
        await asyncio.sleep(0.5)
    else:
        await ws.send_json({"type": "error", "msg": "Model loading timed out"})
        await ws.close()
        return

    # ── Per-connection state ──────────────────────────────────────
    state = {
        "speed":          10.0,    # timesteps / second (1–200)
        "attack_prob":     0.0,    # prob per step of auto-starting attack
        "theta":          20.0,    # attack intensity (extra minutes, 1–60)
        "burst_duration": 15,
        "paused":         False,
        "model":          "narx",  # active model key
        "sensitivity":    5,       # 1=conservative … 10=aggressive
        "switch_model":   False,   # signal to restart from t=0
    }

    def get_model():
        k = state["model"] if state["model"] in _models else list(_models.keys())[0]
        return _models[k]

    attacker = MITREFDIAttacker()
    attacker.set_baseline(get_model()["baseline"])
    attacker._theta = state["theta"]

    # Per-step detection state
    cusum     = 0.0
    spike_win = deque(maxlen=3)     # IQR sliding window, q=3

    # Running confusion matrix
    tp = fp = fn = tn = 0

    # ── Receive loop ──────────────────────────────────────────────
    async def recv_loop() -> None:
        nonlocal cusum, tp, fp, fn, tn
        while True:
            try:
                raw = await ws.receive_text()
                msg = json.loads(raw)
            except (WebSocketDisconnect, Exception):
                break

            action = msg.get("action", "")

            if action == "set_speed":
                state["speed"] = max(1.0, min(200.0, float(msg.get("value", 10))))

            elif action == "set_attack_prob":
                state["attack_prob"] = max(0.0, min(0.5, float(msg.get("value", 0))))

            elif action == "set_theta":
                v = max(1.0, min(60.0, float(msg.get("value", 20))))
                state["theta"] = v
                attacker._theta = v

            elif action == "set_sensitivity":
                state["sensitivity"] = max(1, min(10, int(msg.get("value", 5))))

            elif action == "set_model":
                new_model = msg.get("value", "narx")
                if new_model in _models and new_model != state["model"]:
                    state["model"] = new_model
                    state["switch_model"] = True   # signal send_loop to restart
                    attacker.set_baseline(get_model()["baseline"])
                    cusum = 0.0
                    spike_win.clear()
                    tp = fp = fn = tn = 0

            elif action == "inject_now":
                dur = int(msg.get("duration", state["burst_duration"]))
                attacker.start_attack(state["theta"], dur)

            elif action == "stop_attack":
                attacker.stop_attack()

            elif action == "pause":
                state["paused"] = True

            elif action == "resume":
                state["paused"] = False

            elif action == "reset":
                cusum = 0.0
                spike_win.clear()
                tp = fp = fn = tn = 0
                attacker.stop_attack()

            elif action == "scenario":
                scenarios = {
                    "clean":      {"attack_prob": 0.0,  "theta": 20.0},
                    "subtle":     {"attack_prob": 0.03, "theta": 5.0 },
                    "moderate":   {"attack_prob": 0.08, "theta": 20.0},
                    "aggressive": {"attack_prob": 0.15, "theta": 60.0},
                }
                name = msg.get("name", "")
                if name in scenarios:
                    for k_s, v_s in scenarios[name].items():
                        state[k_s] = v_s
                    attacker._theta = state["theta"]

    recv_task = asyncio.get_event_loop().create_task(recv_loop())

    # ── Send loop — loops the dataset forever ─────────────────────
    loop_count = 0
    try:
        while True:
            loop_count += 1
            m          = get_model()
            N          = m["n"]
            state["switch_model"] = False   # clear restart flag

            for i in range(N):
                # Model switch requested — break inner loop to restart
                if state["switch_model"]:
                    break

                # Pause
                while state["paused"]:
                    await asyncio.sleep(0.05)

                y_true = float(m["y_true"][i])
                y_pred = float(m["y_pred"][i])

                # Apply (or maybe start) attack
                y_obs  = attacker.maybe_inject(y_true, state["attack_prob"])
                atk    = attacker.status()

                # ── Sensitivity scaling ───────────────────────────
                # exp((5 – s) × 0.3): s=1→3.32×, s=5→1.0×, s=10→0.22×
                scale_s = math.exp((5 - state["sensitivity"]) * 0.3)
                eff_ub  = m["iqr_ub"] * scale_s
                eff_h   = m["cusum_h"] * scale_s

                # ── EoE ───────────────────────────────────────────
                eoe = abs(y_obs - y_pred)

                # ── CUSUM (primary detector) ───────────────────────
                cusum = max(0.0, cusum + eoe - m["cusum_k"])
                cusum_detected = cusum > eff_h

                # ── IQR sliding-window (secondary) ────────────────
                is_spike = eoe > eff_ub
                spike_win.append(is_spike)
                iqr_detected = (len(spike_win) == 3 and all(spike_win))

                # ── Combined detection ────────────────────────────
                detected = cusum_detected or iqr_detected
                det_method = ("CUSUM" if cusum_detected else
                              "IQR"   if iqr_detected   else "")

                confidence = min(1.0, cusum / max(eff_h, 1e-9))

                # ── Online confusion-matrix ───────────────────────
                if   atk.active and     detected: tp += 1
                elif atk.active and not detected: fn += 1
                elif not atk.active and detected: fp += 1
                else:                             tn += 1

                total = tp + fp + fn + tn
                prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                acc   = (tp + tn) / total            if total > 0         else 0.0

                payload = {
                    "type":            "tick",
                    "t":               i + 1,
                    "loop":            loop_count,
                    "n_total":         N,
                    "model":           m["name"],
                    # Signal values
                    "y_true":          round(y_true, 6),
                    "y_pred":          round(y_pred, 6),
                    "y_obs":           round(y_obs,  6),
                    "eoe":             round(eoe, 6),
                    "threshold_iqr":   round(eff_ub, 6),
                    "threshold_cusum": round(eff_h,  6),
                    "cusum":           round(cusum,  4),
                    # Detection
                    "is_spike":        bool(is_spike),
                    "iqr_detected":    bool(iqr_detected),
                    "cusum_detected":  bool(cusum_detected),
                    "detected":        bool(detected),
                    "det_method":      det_method,
                    "confidence":      round(confidence, 4),
                    # Attack ground-truth
                    "attack_active":   bool(atk.active),
                    "technique_id":    atk.technique_id,
                    "technique_name":  atk.technique_name,
                    "tactic":          atk.tactic,
                    "theta":           float(atk.theta),
                    "burst_remaining": int(atk.burst_remaining),
                    # Running metrics
                    "metrics": {
                        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                        "precision": round(prec, 4),
                        "recall":    round(rec,  4),
                        "f1":        round(f1,   4),
                        "accuracy":  round(acc,  4),
                        "total":     total,
                    },
                }
                await ws.send_json(payload)
                await asyncio.sleep(1.0 / max(0.5, state["speed"]))

            # After inner loop: either switch happened or data ended
            if not state["switch_model"]:
                loop_count += 1   # natural loop of the dataset

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        print(f"[WS] Error: {exc}")
    finally:
        recv_task.cancel()
