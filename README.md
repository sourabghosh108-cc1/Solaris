# ☀️ Solar Filament Segmentation Challenge 2026 (MAGFiLO 1.0)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics YOLO11](https://img.shields.io/badge/Ultralytics-YOLO11-00FFFF.svg)](https://docs.ultralytics.com/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

Official solution repository for the **Solar Filament Segmentation Challenge 2026**, hosted by the **Earth-Space AI Research (ESAIR) Lab**, sponsored by the **U.S. National Science Foundation (NSF)** and the **National Solar Observatory (NSO)** in conjunction with **IEEE BigData 2026**.

- **Public Leaderboard Score:** `0.3700`
- **Validation Panoptic Quality (PQ):** `0.4145` (DQ: `0.6093`, SQ: `0.6802`, Best Crop Dice: `0.8433`)
- **Human Inter-Annotator Agreement Benchmark:** $\approx 0.3600$

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
                   │ Stage 1: YOLO11m Candidate Detector       │
                   │ • Trained @ 1280x1280, AdamW lr=0.00087   │
                   │ • Inference @ 1024x1024 (conf >= 0.05)    │
                   └─────────────────────┬─────────────────────┘
                                         │ Bounding Box Proposals
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │ Stage 2: UNet++ (EfficientNet-B4 Backbone)│
                   │ • 512x512 Contextual Padded Crops (20%)   │
                   │ • BCETverskyLoss (0.5 BCE + 0.5 Tversky)  │
                   │ • α=0.3, β=0.7 (penalizes false negatives)│
                   └─────────────────────┬─────────────────────┘
                                         │ Sigmoid Probabilities
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │ Panoptic Post-Processing & RLE Assembly   │
                   │ • Optimal Thresholds: conf=0.35, mask=0.60│
                   │ • Greedy Sequential Disjoint Painter      │
                   │ • Non-Overlapping Instance Masks          │
                   └─────────────────────┬─────────────────────┘
                                         │
                                         ▼
                            submission.csv (COCO RLE)
```

---

## 🚀 Key Methodological Features

1. **Two-Stage Decoupled Framework**: Candidate localization via YOLO11m separates the spatial search problem from high-resolution boundary delineation.
2. **Contextually Padded UNet++**: Bounding boxes are padded with a $20\%$ margin to ensure full capture of thin protruding barbs.
3. **BCETverskyLoss Formulation**: Weighted Tversky loss ($\alpha=0.3, \beta=0.7$) forces the segmenter to prioritize recall on delicate filament boundary threads.
4. **Greedy Panoptic Painter**: Claims pixels sequentially in descending order of detector confidence, strictly guaranteeing non-overlapping masks ($M_i \cap M_j = \emptyset$) with no fragmentation penalties.

---

## 📂 Repository Structure

```
├── sfscs.ipynb           # End-to-end training, validation, & submission pipeline notebook
├── requirements.txt      # Python dependencies with pinned minimum versions
├── main.tex              # ACM LaTeX technical report source (4 pages)
├── main.bib              # BibTeX citation database
├── technical_report.md   # Markdown version of technical report
└── README.md             # Solution overview, setup, and reproduction guide
```

---

## ⚙️ Environment Setup & Installation

### 1. Prerequisites
- Python 3.10+
- CUDA 11.8+ / 12.x enabled GPU (e.g., NVIDIA Tesla T4, P100, RTX 3090/4090 with $\ge 15\text{ GB}$ VRAM)

### 2. Install Dependencies
```bash
# Clone repository
git clone https://github.com/sourabghosh108-cc1/Solaris
cd Solaris

# Install required packages
pip install -r requirements.txt
```

---

## 🔄 Reproduction Guide

### Running on Kaggle (Recommended)
1. Attach the official dataset:
   `competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026`
2. Open [`sfscs.ipynb`](sfscs.ipynb) on Kaggle.
3. Set Accelerator to **GPU T4 x2** or **GPU P100**.
4. Click **"Run All"** / **"Save & Run All (Commit)"**.
5. Generated files in `/kaggle/working/`:
   - `submission.csv` (1,152 predicted instances, 0 overlaps)
   - `best_seg.pth` (Trained UNet++ weights)
   - `seg_history.csv` & `threshold_sweep.csv` (Validation logs)

---

## 📊 Evaluation & Metrics Summary

$$\text{PQ} = \text{DQ} \times \text{SQ} = \frac{\sum_{(p, g) \in \text{TP}} \text{IoU}(p, g)}{|\text{TP}| + \frac{1}{2}|\text{FP}| + \frac{1}{2}|\text{FN}|}$$

| Run | Detection mAP50 | Best Crop Dice | Val DQ | Val SQ | Val PQ | Public Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Human Inter-Annotator Benchmark | — | — | — | — | $\approx 0.3600$ | — |
| **V1 Baseline (Our Model)** | **0.6369** | **0.8433** | **0.6093** | **0.6802** | **0.4145** | **0.3700** |

---

## 📜 Citations & References

```bibtex
@article{magfilo2024,
  title={A dataset of manually annotated filaments from H-alpha observations},
  author={Ahmadzadeh, Azim and others},
  journal={Nature Scientific Data},
  volume={11},
  pages={1031},
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
This repository is licensed under the [Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/) license.
