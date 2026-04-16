# MASTER PROMPT FOR SONNET 4.6 — IEEE Research Paper Generation (LaTeX)

---

## YOUR TASK

You are an expert academic researcher and LaTeX typesetter. Your job is to write a **complete, publication-ready IEEE Transactions-style research paper in LaTeX** that serves as an **extension** of the following base paper:

> **"False Data Injection Attack Detection in EV Charging Network Using NARX Neural Network"**
> Basumatary, Khatua, Nath — IEEE Transactions on Transportation Electrification, Vol. 11, No. 4, August 2025.

You will be given:
1. The **current project status / new contributions** by the user (pasted below the prompt).
2. The **structural template** of the base paper (described in detail below).

Your output must be **100% valid LaTeX code**, using the `IEEEtran` document class, suitable for direct compilation. Do not output anything outside the LaTeX code block.

---

## STRICT STRUCTURAL REQUIREMENTS

Replicate the **exact section hierarchy** of the base paper:

```
\begin{document}
  \title{...}
  \author{...}
  \maketitle
  \begin{abstract}...\end{abstract}
  \begin{IEEEkeywords}...\end{IEEEkeywords}

  \section{INTRODUCTION}
    \subsection{Motivation and Challenges}
    \subsection{Contribution}

  \section{RELATED WORK}

  \section{PRELIMINARIES}
    % Mathematical foundations of your proposed method
    % Include numbered equations using \begin{equation}...\end{equation}

  \section{PROPOSED SCHEME}
    \subsection{System Model}
    \subsection{Attack Model}        % or Threat Model
    \subsection{Attack Implementation}
    \subsection{Impacts of the Attack}
    \subsection{Why [YOUR METHOD] for Estimation/Detection?}
    \subsection{[Your Detection/Mitigation Scheme]}
    \subsection{Complexity Analysis}

  \section{SIMULATION AND RESULTS}
    \subsection{Dataset Description}
    \subsection{Training Under Normal Condition}
    \subsection{Estimation/Inference Results}
    \subsection{Performance Analysis}
    \subsection{Comparative Study}

  \section{REAL-TIME ANALYSIS}
    % Validation using hardware-in-loop or equivalent framework

  \section{CONCLUSION}

  \begin{thebibliography}{99}
    ...
  \end{thebibliography}
\end{document}
```

---

## LATEX FORMATTING RULES

Use the following LaTeX preamble exactly:

```latex
\documentclass[journal]{IEEEtran}
\usepackage{amsmath, amssymb, amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{cite}
\usepackage{url}
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usetikzlibrary{shapes.geometric, arrows, positioning, fit, calc}
```

---

## NOMENCLATURE SECTION

After the abstract and keywords, include a nomenclature table exactly like the base paper, using this format:

```latex
\section*{NOMENCLATURE}
\begin{IEEEdescription}[\IEEEusemathlabelsep\IEEEsetlabelwidth{$\omega_i$}]
\item[$\omega_i$] Output of $i$th neuron in the previous layer.
\item[$m$] Number of input signals.
% ... add all your parameters
\end{IEEEdescription}
```

---

## DIAGRAM AND FIGURE REQUIREMENTS

The base paper contains the following types of figures. You MUST generate **equivalent LaTeX/TikZ figures** for each category. Do NOT use `\includegraphics` for diagrams — draw them in TikZ/PGFPlots so the paper compiles standalone.

### Figure 1 — Neural Network Neuron Structure
Replicate Fig. 1 from the base paper: a single neuron diagram showing:
- Input nodes α₁, α₂, ..., αₘ on the left
- Weights w_hid1, w_hid2, ..., w_hidm
- A summation node (Σ) leading to activation function F(·)
- Output y on the right
- Bias b_hid feeding into the summation node

Generate this as a TikZ figure:
```latex
\begin{figure}[!t]
\centering
\begin{tikzpicture}[...]
  % Draw input nodes, weight labels, summation circle, activation box, output arrow
\end{tikzpicture}
\caption{Structure of a neuron in NN.}
\label{fig:neuron}
\end{figure}
```

### Figure 2 — System Architecture / Attack Scenario Diagram
Replicate the style of Fig. 2 from the base paper: a network topology block diagram showing:
- User layer (U₁, U₂, ..., Uₙ) at the top with user input parameters (Start Time, End Time, Energy Demand, Minutes Available)
- HTTP communication arrow going to CMS block
- An Attacker node intercepting the HTTP channel, with "Attack" and "Impact" arrows
- OCPP protocol arrow from CMS going down to EVSE layer (EVSE₁, ..., EVSEₘ)
- IEC/ISO-15118 / IEC 61851 protocol arrow from EVSEs going down to EV layer (EV₁, ..., EVₙ)
- Annotate the attack injection point with the false data formula

Adapt this diagram to your **new system model**. Keep the same layered topology style. Generate in TikZ.

### Figures 3–9 — Time Series Signal Plots
These are MATLAB-style 2D plots of energy (kWh) vs. sample points. Replicate using `pgfplots`:
```latex
\begin{figure}[!t]
\centering
\begin{tikzpicture}
\begin{axis}[
  width=\columnwidth, height=4.5cm,
  xlabel={Number of Sample Points ($\times 10^4$)},
  ylabel={Energy Delivered (kWh)},
  xmin=0, xmax=3,
  legend style={at={(0.98,0.98)}, anchor=north east, font=\small},
  grid=major, grid style={dashed, gray!40}
]
\addplot[blue, thick] coordinates {...}; \addlegendentry{Actual data ($y$)}
\addplot[red, thick, dashed] coordinates {...}; \addlegendentry{Estimated data ($\bar{y}$)}
\end{axis}
\end{tikzpicture}
\caption{Actual and estimated data by the model during training.}
\label{fig:training}
\end{figure}
```
Generate representative synthetic data coordinates that match the shape and scale of the plots in the base paper (small oscillating values around 0–0.05 kWh, with transition spikes between sessions).

### Figure 10 — IQR Outlier Detection Plot
Replicate the style of Fig. 10: EoE on y-axis vs. sample points on x-axis, with:
- A horizontal dashed line for Q₂ (median) at 0
- Upper dashed line for UB
- Lower dashed line for LB
- Blue dots for normal (benign) data
- Red dots for outlier/attack instances
- Annotations: "Normal case" and "Attacked case" regions

### Tables
Generate all tables using `\begin{table}[!t]` with `\begin{tabular}` and `\hline` borders, matching the IEEE style of the base paper. Required tables:

**Table I** — Summary of Cyberattacks (columns: Reference, Attack Type, Method, Limitation):
```latex
\begin{table}[!t]
\caption{Summary of Cyberattacks in the Smart Grid System}
\label{tab:survey}
\centering
\begin{tabular}{|p{1cm}|p{2cm}|p{2cm}|p{2.5cm}|}
\hline
\textbf{Ref.} & \textbf{Attack Type} & \textbf{Method} & \textbf{Limitation} \\
\hline
...
\hline
\end{tabular}
\end{table}
```

**Table II** — Your Model's Hyperparameters (e.g., activation function, layers, neurons, delay, optimizer, epochs).

**Table III** — Charging Details of different EV types (PEV vs. PHEV comparison with battery capacity, charging rate, time).

**Table IV** — Dataset Field Descriptions (two columns: Data Field | Description).

**Table V** — Confusion Matrix for Proposed Scheme (2×2 matrix: TN, FP, FN, TP).

**Table VI** — Confusion Matrix for Baseline Model.

**Table VII** — Comparison of Models (rows: Accuracy, Precision, Recall, F1-Score; columns: Your Model | Baseline 1 | Baseline 2).

---

## MATHEMATICAL CONTENT REQUIREMENTS

The paper must contain the following categories of equations, properly numbered:

1. **NARX/model mapping equation** — nonlinear function F(·) relating past inputs x(t−1), x(t−2)... and past outputs y(t−1), y(t−2)... to current output y(t). Adapt this to your proposed model.

2. **Single hidden layer expansion** — showing weight matrix W_hid, bias b_hid, activation F_hid, and output layer W_out, b_out, F_out.

3. **Input vector definition** — U(t−1) stacking exogenous and endogenous inputs.

4. **Attack/Threat model equation** — showing how the attacker modifies a parameter (e.g., M̃A = MA − φ), mirroring Eq. (6) from the base paper.

5. **Session update under attack** — CS̃ = f(ST, ET, ED, M̃A), mirroring Eq. (7).

6. **Error of Estimation (EoE)** — EoE(t) = y(t) − ȳ(t).

7. **IQR bounds** — IQR = Q₃ − Q₁, LB = Q₁ − β·IQR, UB = Q₃ + β·IQR.

All equations must use proper LaTeX math environments and be cross-referenced in the text.

---

## WRITING STYLE REQUIREMENTS

- Write in **formal IEEE academic English**. Third person only.
- Every claim must cite a reference in `\cite{key}` format.
- Subsections in the Introduction must discuss: (a) the research gap in existing literature explicitly, (b) the stealthy/unpredictable nature of the attack being addressed, (c) why existing ML classifiers are insufficient.
- The "Why [METHOD]?" subsection must argue: (1) no system model required, (2) handles imbalanced data better than classifiers, (3) memory/recurrence advantage.
- The Related Work section must reference at least 10 prior works with specific critiques of each.
- The Complexity Analysis subsection must derive Big-O notation for training, sorting (for IQR), and inference phases separately.
- The Conclusion must (a) summarize results numerically, (b) acknowledge a specific limitation, (c) state future work direction.

---

## ACCURACY METRICS FORMAT

In the Performance Analysis subsection, report results in this exact style:
> "The performance of the proposed model is evaluated using several metrics, which show **X%** accuracy, **Y%** recall, **Z%** precision, and **W%** F1-score."

In the Comparative Study, use a sentence like:
> "The proposed [METHOD]-based attack detection gives **X%** accuracy, whereas the existing schemes give **Y%** and **Z%** accuracies, respectively."

---

## REAL-TIME VALIDATION SECTION

This section must describe:
1. The simulation environment (e.g., OPAL-RT, Simulink, or equivalent).
2. A TikZ block diagram of the real-time setup (CPU ↔ LAN ↔ RT Simulator), styled like Fig. 11/12 of the base paper.
3. A pgfplots figure of the scope output showing EoE over time, the IQR bounds as dashed lines, and the attack detection region highlighted in red — styled like Fig. 13 of the base paper.
4. Specific timing: state at which sample point / timestamp the attack was injected and at which point it was detected, and calculate the detection delay in time steps.

---

## WHAT THE USER WILL PROVIDE BELOW

After this prompt, the user will describe:
- The **new proposed method** (e.g., a new neural architecture, a new dataset, a new attack model)
- The **dataset** being used
- The **performance results** (accuracy, precision, recall, F1)
- Any **specific changes** to the system model or attack model
- Names of **comparison baselines**

Use all of that information to fill in the paper content. Where the user has not specified exact numbers, **generate plausible synthetic but realistic values** consistent with the performance range of the base paper (accuracy ~98–99.5%), and mark them with a LaTeX comment `% [PLACEHOLDER — replace with real result]` so the user knows where to substitute.

---

## OUTPUT FORMAT

Output **only** the complete `.tex` file content. Start with `\documentclass` and end with `\end{document}`. No explanation, no markdown, no preamble text outside the LaTeX. The file must compile with `pdflatex` or `xelatex` without errors.

---

*End of master prompt. Paste your project status and new contributions below this line.*
