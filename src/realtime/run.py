"""
Entry point for the EV FDI Real-Time Detection Demo.

Usage (from narx_ev_fdi/ project root):
    python -m src.realtime.run
    python -m src.realtime.run --port 8080
    python -m src.realtime.run --host 0.0.0.0 --port 8000
"""

import argparse
import os
import sys

# Fix hash seed BEFORE any project imports so siteID hashing is
# consistent with the training run.
os.environ.setdefault("PYTHONHASHSEED", "0")

import uvicorn  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="EV FDI Demo Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Hot-reload on code changes")
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  MITRE ATT&CK ICS · EV Charging FDI Real-Time Demo          ║
║  Model: NARX + Session-Aware CUSUM                           ║
╠══════════════════════════════════════════════════════════════╣
║  Starting server …                                           ║
║  Dashboard → http://{args.host}:{args.port}                          ║
╚══════════════════════════════════════════════════════════════╝
""")

    uvicorn.run(
        "src.realtime.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="warning",   # suppress uvicorn noise; demo prints its own
    )


if __name__ == "__main__":
    main()
