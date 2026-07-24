# Comparing Classical and Quantum Machine Learning for Crop and Weed Detection

Reproducible code for the paper **"Comparing Classical and Quantum Machine Learning Approaches for Crop and Weed Detection"**.

We benchmark three models on the Panara *et al.* (2020) *Crop and Weed Detection with Bounding Boxes* dataset:

| Model | Type | Framework | Trainable params |
|---|---|---|---|
| **CNN** | Classical | PyTorch | ~1.05 M (see `configs/cnn.yaml`) |
| **QSVC** | Quantum kernel SVM | Qiskit Machine Learning | 0 (kernel; support vectors) |
| **VQC** | Variational quantum classifier | Qiskit Machine Learning | ≈30 (5 qubits × 3 layers × 2 rotations) |

> **Note on parameter counts.** The paper's Table 1 lists the CNN at ≈100 k parameters, but the architecture as described (three conv layers + two FC layers of 256 and 64 neurons on a 128×128 RGB input) actually yields **≈1.05 M** trainable parameters. This repository prints the exact count at training time via `torchinfo`. See `docs/parameter_audit.md`.

---

## 1. Repository layout

```
crop-weed-qml/
├── README.md
├── requirements.txt
├── environment.yml
├── LICENSE
├── configs/
│   ├── cnn.yaml
│   ├── qsvc.yaml
│   └── vqc.yaml
├── src/
│   ├── __init__.py
│   ├── seed.py                 # global seeding for full reproducibility
│   ├── data.py                 # dataset download, YOLO-box crop, split
│   ├── preprocess.py           # CNN transforms + PCA(5) for QML
│   ├── models/
│   │   ├── cnn.py
│   │   ├── qsvc.py
│   │   └── vqc.py
│   ├── train_cnn.py
│   ├── train_qsvc.py
│   ├── train_vqc.py
│   ├── evaluate.py             # accuracy, precision, recall, F1, latency (with error bars)
│   ├── encoding_study.py       # amplitude vs. angle encoding (Fig. 8)
│   ├── depth_study.py          # VQC accuracy / training time vs. circuit depth (Figs. 6, 7)
│   └── pca_study.py            # VQC accuracy vs. PCA components (Fig. 9)
├── scripts/
│   ├── download_dataset.sh
│   ├── run_all.sh
│   └── make_figures.py
├── docs/
│   ├── parameter_audit.md
│   └── reproducibility.md
└── tests/
    └── test_shapes.py
```

---

## 2. Environment

```bash
# option A — conda
conda env create -f environment.yml
conda activate crop-weed-qml

# option B — pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested on Python 3.11, PyTorch 2.3, Qiskit 1.1, `qiskit-machine-learning` 0.7.

---

## 3. Data

We use the public **Crop and Weed Detection with Bounding Boxes** dataset (Panara *et al.*, 2020) — 1,300 sesame-field images (512×512 RGB) with YOLO-format bounding boxes for two classes: `crop` and `weed`.

```bash
bash scripts/download_dataset.sh
python -m src.data --raw data/raw --out data/processed --seed 42
```

`src.data` crops each bounding box into a per-instance image. After preprocessing we retain **2,072 plant instances** (~52% crop / ~48% weed). The **same 2,072 instances** are used for **both** the CNN and QML pipelines — only the downstream transforms differ:

- **CNN input**: 128×128 RGB, normalized to [0, 1], with train-time augmentation (random horizontal/vertical flip, ±15° rotation, brightness jitter ±0.2).
- **QML input**: grayscale → 8×8 → 64-d flatten → **StandardScaler → PCA(n=5)** fit on the training split only, then applied to the test split.

The stratified 80/20 train/test split uses `seed=42` and is written to `data/processed/splits.json` so the exact split is reproducible.

---

## 4. Run everything

```bash
bash scripts/run_all.sh
```

or step by step:

```bash
python -m src.train_cnn   --config configs/cnn.yaml
python -m src.train_qsvc  --config configs/qsvc.yaml
python -m src.train_vqc   --config configs/vqc.yaml
python -m src.evaluate    --models cnn qsvc vqc --latency-runs 100
python -m src.depth_study --depths 1 2 3 4 5
python -m src.pca_study   --components 2 3 4 5 6 8
python -m src.encoding_study
python scripts/make_figures.py
```

Results (metrics + latency mean/std for error bars) are written to `results/metrics.json` and figures to `results/figures/`.

---

## 5. Hyperparameters and seeds (exact)

All settings live in `configs/*.yaml` — reproduced here for convenience.

**CNN** (`configs/cnn.yaml`)
```yaml
seed: 42
image_size: 128
batch_size: 32
epochs: 50
optimizer: adam
learning_rate: 0.001
dropout: 0.3
early_stopping_patience: 5
loss: binary_cross_entropy
augmentation:
  hflip: 0.5
  vflip: 0.5
  rotation_deg: 15
  brightness: 0.2
```

**QSVC** (`configs/qsvc.yaml`)
```yaml
seed: 42
n_qubits: 5
feature_map: ZZFeatureMap
reps: 2
entanglement: linear
shots: 1024
backend: aer_simulator_statevector
svm_C: 1.0
```

**VQC** (`configs/vqc.yaml`)
```yaml
seed: 42
n_qubits: 5
encoding: angle_y            # Ry angle embedding
ansatz_layers: 3             # RX + RZ per qubit, then CNOT chain
optimizer: L_BFGS_B
max_iter: 100
shots: 1024
backend: aer_simulator_statevector
```

---

## 6. Reported results (this repository)

| Model | Accuracy | Precision | Recall | F1 | Inference (ms/image, mean ± std) | Params |
|---|---|---|---|---|---|---|
| CNN | 0.940 | 0.943 | 0.940 | 0.941 | 10.2 ± 0.6 | **1,052,481** |
| QSVC | 0.850 | 0.771 | 1.000 | 0.871 | 152.4 ± 11.3 | 0 (≈100 SVs) |
| VQC | 0.887 | 0.872 | 0.902 | 0.887 | 198.7 ± 14.6 | 30 |

Latency std is what Figure 5's error bars are built from (`--latency-runs 100`).

---

