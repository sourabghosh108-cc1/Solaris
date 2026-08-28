# ☀️ Solar Filament Segmentation Challenge 2026 (MAGFiLO 1.0)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics YOLO11](https://img.shields.io/badge/Ultralytics-YOLO11-00FFFF.svg)](https://docs.ultralytics.com/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

Official solution repository for the **Solar Filament Segmentation Challenge 2026**, hosted by the **Earth-Space AI Research (ESAIR) Lab**, sponsored by the **U.S. National Science Foundation (NSF)** and the **National Solar Observatory (NSO)** in conjunction with **IEEE BigData 2026**.

---

## 📌 Problem Overview & Dataset

Solar filaments are dense, magnetic structures suspended above photospheric neutral lines that serve as prime indicators for Coronal Mass Ejections (CMEs) and geomagnetic storms. Accurately delineating their fine-scale morphology (spines, thin thread-like barbs) from ground-based GONG H-Alpha observations ($2048 \times 2048$ grayscale images) is critical for space weather forecasting.

This solution is trained and evaluated on the **MAGFiLO 1.0** dataset (*Manually Annotated GONG Filaments from H-Alpha Observations*), evaluated via the **Panoptic Quality (PQ)** metric ($PQ = DQ \times SQ$).

---

## 🏗️ Pipeline Architecture

Our solution uses a **Two-Stage Detection and Crop Segmentation Pipeline** with Panoptic Post-Processing:

```
                  ┌──────────────────────────────────────────────┐
                  │ 2048x2048 Full Grayscale Solar Observations  │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │ Stage 1: YOLO11m Detector (@ 1280x1280)   │
                   │ • Multi-Annotator Bounding Box Fusion     │
                   │ • Optimized Candidate Localization        │
                   └─────────────────────┬─────────────────────┘
                                         │ Bounding Box Proposals
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │ Stage 2: UNet++ (EfficientNet-B4 Backbone)│
                   │ • 512x512 Contextual Padded Crops         │
                   │ • Box Jitter (0.12) Invariance Training   │
                   │ • Focal Tversky Loss (α=0.3, β=0.7, γ=1.3)│
                   └─────────────────────┬─────────────────────┘
                                         │ Sigmoid Probabilities
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │ Panoptic Post-Processing & RLE Assembly   │
                   │ • Test-Time Augmentation (H/V Flips)      │
                   │ • Optimal Confidence & Mask Thresholding  │
                   │ • Min-Area Filter & Morphological Closing │
                   │ • Disjoint Non-Overlapping Instance Masks │
                   └─────────────────────┬─────────────────────┘
                                         │
                                         ▼
                            submission.csv (COCO RLE)
```

---

## 🚀 Key Methodological Features

1. **Multi-Annotator Box Consensus**: Merges overlapping annotations across independent human experts via IoU-based NMS ($0.70$ threshold), exposing the detector to the full distribution of valid filaments.
2. **CLAHE Contrast Normalization**: Enhances subtle chromospheric contrast variations and solar limb darkening effects.
3. **Box Jitter Invariance**: Injects $\pm 12\%$ random bounding box translation/scaling during segmenter training to ensure robustness against imperfect detector proposals.
4. **Focal Tversky Loss**: Heavily penalizes false negatives on thin, delicate filament barbs and focuses on hard boundary pixels.
5. **Panoptic Disjoint Painter**: Claims pixels sequentially in descending order of detector confidence, strictly guaranteeing non-overlapping masks with no fragmentation penalties.

---

## 📂 Repository Structure

```
├── sfscs.ipynb           # End-to-end training, validation, & submission pipeline notebook
├── requirements.txt      # Python dependencies with pinned minimum versions
├── README.md             # Solution overview, setup, and reproduction guide
└── (artifacts/)          # Generated submission.csv, models, and evaluation sweeps
```

---

## ⚙️ Environment Setup & Installation

### 1. Prerequisites
- Python 3.10+
- CUDA 11.8+ / 12.x enabled GPU (e.g., NVIDIA Tesla T4, P100, RTX 3090/4090 with $\ge 15\text{ GB}$ VRAM)

### 2. Install Dependencies
```bash
# Clone repository
git clone https://github.com/<your-username>/solar-filament-segmentation-2026.git
cd solar-filament-segmentation-2026

# Create and activate virtual environment
python -m venv venv
# Linux / MacOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

---

## 🔄 Reproduction Guide

### Running on Kaggle (Recommended)
1. Create a new notebook on Kaggle and attach the dataset:
   `competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026`
2. Upload and open [`sfscs.ipynb`](sfscs.ipynb).
3. Set Accelerator to **GPU T4 x2** or **GPU P100**.
4. Click **"Run All"** / **"Save & Run All (Commit)"**.
5. Output files generated in `/kaggle/working/`:
   - `submission.csv` (Ready for submission)
   - `best_seg.pth` (Trained UNet++ weights)
   - `yolo_runs/` (Trained YOLO11 detector checkpoints)
   - `seg_history.csv` & `threshold_sweep.csv` (Validation logs)

### Running Locally / Custom Server
Ensure the MAGFiLO dataset is placed under `./MAGFiLO_1.0_Kaggle_2026/`, update `ROOT` path in Cell 2 of [`sfscs.ipynb`](sfscs.ipynb), and execute all notebook cells sequentially.

---

## 📊 Evaluation & Metrics Summary

The solution is evaluated using **Panoptic Quality (PQ)**:

$$\text{PQ} = \text{DQ} \times \text{SQ} = \frac{\sum_{(p, g) \in \text{TP}} \text{IoU}(p, g)}{|\text{TP}| + \frac{1}{2}|\text{FP}| + \frac{1}{2}|\text{FN}|}$$

- **Detection Quality (DQ)**: Measures precision & recall of identified filament instances.
- **Segmentation Quality (SQ)**: Measures mean IoU for matched true positive segmentations.
- **Human Benchmark**: Average agreement between expert human annotators on MAGFiLO is $\approx 0.36$.

---

## 📜 Citations & References

If using this codebase or dataset, please cite the MAGFiLO dataset paper and competition:

```bibtex
@article{magfilo2024,
  title={MAGFiLO: A Ground-Truth Dataset of Manually Annotated Filaments from GONG H-Alpha Observations},
  journal={Scientific Data},
  year={2024},
  doi={10.1038/s41597-024-03876-y}
}

@misc{filament_competition_2026,
  title={Solar Filament Segmentation Challenge 2026},
  author={Ahmadzadeh, Azim and Kempton, Dustin J. and Li, Qin and Pevtsov, Alexei A.},
  year={2026},
  howpublished={\url{https://kaggle.com/competitions/filament-segmentation-2026}}
}
```

---

## 📄 License
This repository is licensed under the [Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/) license in compliance with competition data guidelines.
