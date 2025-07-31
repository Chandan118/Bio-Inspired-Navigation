# AutoOpticalDiagnostics

AutoOpticalDiagnostics is an end-to-end, physics-informed pipeline for automatic surface-defect detection using synthetic optical data.

The project reproduces the main contributions from the accompanying research paper:

1. **Physics-Based Simulation** – We synthesise Optical Coherence Tomography (OCT) and Laser Speckle Contrast Imaging (LSCI) frames together with perfect ground-truth masks.
2. **Industrial Realism** – The generator injects thermal noise and motion-blur artefacts to emulate hostile factory environments (see paper Table 10).
3. **AI-Driven Inspection** – A lightweight U-Net is trained on the synthetic data to segment defects.
4. **Evaluation & Reporting** – Integrated scripts benchmark the model (Dice, IoU, PR curves) and emit a markdown/text report + sample visualisations.
5. **Modular & Professional Codebase** – Every concern lives in its own module, all configurable via `src/config.py` and orchestrated from `main.py`.

## Project Layout
```text
AutoOpticalDiagnostics/
├── data/                # Synthetic dataset appears here after generation
├── models/              # Trained weights
├── outputs/             # Evaluation reports + plots
├── src/                 # Source code (simulation, training, evaluation)
├── main.py              # One-click pipeline entry-point
├── requirements.txt     # Python dependencies
└── README.md            # You are here
```

## Quick-Start
```bash
# 1. Setup environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Run full pipeline (generate ➔ train ➔ evaluate)
python main.py --run_all

# 3. Explore results
open outputs/evaluation_results/report.txt  # dice/IoU metrics
open outputs/evaluation_results/sample_prediction.png
```

## Customisation
Adjust any hyper-parameters, dataset sizes, or noise characteristics in `src/config.py`.

## Licence
MIT