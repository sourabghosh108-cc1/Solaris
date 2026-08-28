# A Two-Stage Deep Learning Framework for Automated Solar Filament Segmentation from Synoptic H-Alpha Observations

**Research Paper & Technical Report for the Solar Filament Segmentation Challenge 2026 (MAGFiLO 1.0)**  
*IEEE BigData 2026 BigDataCup / National Solar Observatory (NSO) / NSF*

**Author:** Sourab Ghosh  
**Affiliation:** Independent Space AI Research / Competition Team  
**Kaggle Notebook:** `sfscs` (Version 1) | **Public Leaderboard Score:** `0.3700`

---

## Abstract

Accurate automated segmentation of solar filaments from ground-based H-$\alpha$ observations is fundamental for modeling coronal magnetic topology and predicting space weather hazards such as Coronal Mass Ejections (CMEs) and solar flares. However, solar filament morphology exhibits complex, thread-like structures (barbs), high spatial variance, and observational noise across synoptic telescope stations. In this work, we present a two-stage deep learning framework tailored for the MAGFiLO 1.0 benchmark dataset. **Stage 1** deploys an Ultralytics YOLO11m detector for candidate localization of filament bounding boxes across full-disk solar images. **Stage 2** applies a deep UNet++ architecture with an EfficientNet-B4 encoder on contextually padded ($20\%$) filament crops resized to $512 \times 512$. To mitigate severe class imbalance and prioritize faint boundary features, we train the segmenter using a composite **BCETverskyLoss** ($\alpha=0.3, \beta=0.7$). A confidence-ranked greedy panoptic painter enforces strictly disjoint, non-overlapping instance masks in compliance with competition requirements. On the validation set, our pipeline achieves a Panoptic Quality ($PQ$) of **$0.4145$** (Detection Quality $DQ = 0.6093$, Segmentation Quality $SQ = 0.6802$, best crop Dice $= 0.8433$), securing a public leaderboard score of **$0.3700$** and surpassing the human annotator agreement benchmark ($\approx 0.36$).

**Keywords:** Solar Filaments, MAGFiLO 1.0, GONG H-Alpha, Two-Stage Segmentation, YOLO11, UNet++, BCETverskyLoss, Panoptic Quality.

---

## 1. Introduction and Background

Solar filaments are dense, cool plasma structures suspended by helical magnetic field lines in the solar corona above photospheric polarity inversion lines (PILs). When these magnetic channels destabilize, they erupt violently, producing Coronal Mass Ejections (CMEs) that pose severe geomagnetic threats to satellites, aviation, and power distribution grids.

Automated delineation of filaments from ground-based Global Oscillations Network Group (GONG) H-$\alpha$ synoptic observations ($656.3\text{ nm}$) is critical for space weather forecasting. However, automated filament segmentation poses key domain challenges:
1. **Fine-Scale Barb Delineation**: Thin, lateral barbs with faint contrast branch from primary filament spines.
2. **Observational Artifacts & Limb Darkening**: Varying atmospheric turbulence and non-uniform solar disk illumination across ground stations.
3. **Fragmentation and Over-Merging**: Algorithms risk fragmenting contiguous filaments (one-to-many penalty) or merging neighboring distinct filaments (many-to-one penalty).
4. **Annotator Ambiguity**: Diffuse boundaries cause human inter-annotator agreement to average only $\approx 0.36$.

---

## 2. System Architecture

![Two-Stage Pipeline Architecture](figures/pipeline_architecture.png)

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

## 3. Jupyter Notebook Implementation Code

### 3.1 Pipeline Configuration
```python
class CFG:
    SEED = 42

    # ---------- Stage 1: Detector ----------
    DETECTOR      = 'yolo11m.pt'   # 'yolo11s' faster · 'yolo11l/x' requires BATCH=2-4 or will OOM on T4
    DET_IMGSZ     = 1280
    DET_BATCH     = 8              # 16 with an 'x' model at 1280 does NOT fit in 16 GB
    DET_EPOCHS    = 40
    DET_PATIENCE  = 8
    DET_LR        = 0.00087

    # ---------- Stage 2: Crop Segmenter ----------
    ENCODER     = 'efficientnet-b4'
    CROP_SIZE   = 512
    PAD_RATIO   = 0.20
    BOX_JITTER  = 0.00             # 0 in V1 baseline
    SEG_EPOCHS  = 12
    SEG_BATCH   = 8
    SEG_PATIENCE = 7
    SEG_LR      = 2e-4
    TVERSKY_ALPHA, TVERSKY_BETA = 0.3, 0.7

    # ---------- Inference ----------
    INFER_IMGSZ = 1024             # V1 inference size
    CONF_MIN    = 0.05             # floor for candidate caching
    MAX_DET     = 60
    MIN_AREA    = 0                # 0 = disabled in V1
    CONF_GRID   = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    MASK_GRID   = [0.40, 0.50, 0.60]
    SWEEP_PHOTOS = 40

    VAL_FRACTION = 0.15

RUN_NAME = 'V1_baseline'
```

### 3.2 Loss Formulation (BCETverskyLoss)
```python
class TverskyLoss(nn.Module):
    # beta > alpha penalizes false negatives more: targets missing barbs/filaments
    def __init__(self, alpha=CFG.TVERSKY_ALPHA, beta=CFG.TVERSKY_BETA, smooth=1e-6):
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth

    def forward(self, logits, targets):
        p = torch.sigmoid(logits).view(-1); t = targets.view(-1)
        tp = (p * t).sum(); fp = (p * (1 - t)).sum(); fn = ((1 - p) * t).sum()
        return 1 - (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)


class BCETverskyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce, self.tv = nn.BCEWithLogitsLoss(), TverskyLoss()

    def forward(self, logits, targets):
        return 0.5 * self.bce(logits, targets) + 0.5 * self.tv(logits, targets)
```

### 3.3 Greedy Panoptic Painter (Disjoint Instances)
```python
def paint_panoptic(cached, shape, conf_t, mask_t, min_area=CFG.MIN_AREA):
    # In descending confidence order: each pixel goes to the first filament claiming it -> disjoint instances
    h, w = shape
    items = sorted([it for it in cached if it['conf'] >= conf_t], key=lambda d: -d['conf'])
    occupied = np.zeros((h, w), dtype=bool)
    masks = []
    for it in items:
        cx1, cy1, cx2, cy2 = it['box']
        m = np.zeros((h, w), dtype=bool)
        m[cy1:cy2, cx1:cx2] = it['probs'].astype('float32') > mask_t
        m &= ~occupied
        if m.sum() <= min_area:
            continue
        occupied |= m
        masks.append(m)
    return masks
```

---

## 4. Experimental Results and Training Curves

### 4.1 Training Convergence History
![Training Loss and Dice Progression](figures/loss_curve.png)

The UNet++ segmenter achieved a peak validation crop Dice score of **$0.8433$** at epoch 10/12.

### 4.2 Step-by-Step Qualitative Delineation
![Qualitative Predictions](figures/qualitative_predictions.png)

### 4.3 Quantitative Metric Comparison
![Metric Comparison](figures/metric_comparison.png)

| Evaluation Metric | Score / Value |
| :--- | :---: |
| **Stage 1 Detector $\text{mAP}_{50}$** | **$0.6369$** |
| **Stage 1 Detector Precision / Recall** | $0.6154$ / $0.6346$ |
| **Stage 2 Best Crop Validation Dice** | **$0.8433$** (Epoch 10) |
| **Detection Quality ($\text{DQ}$)** | **$0.6093$** ($\text{TP}=815, \text{FP}=466, \text{FN}=579$) |
| **Segmentation Quality ($\text{SQ}$)** | **$0.6802$** |
| **Full Validation Panoptic Quality ($\text{PQ}$)** | **$0.4145$** |
| **Public Leaderboard Score** | **$0.3700$** ($1,152$ instances) |
| **Human Inter-Annotator Agreement Benchmark** | $\approx 0.3600$ |

---

## 5. Conclusion

We presented an effective two-stage deep learning pipeline combining a YOLO11m detector with a UNet++ (EfficientNet-B4) segmenter trained under BCETversky loss. By coupling candidate localization with greedy panoptic assembly, our solution strictly enforces disjoint instance masks, achieves a validation PQ of $0.4145$, and scores $0.3700$ on the Kaggle public leaderboard, surpassing human annotator agreement benchmarks.

---

## References

1. A. Ahmadzadeh, et al., "A dataset of manually annotated filaments from H-alpha observations," *Nature Scientific Data*, vol. 11, p. 1031, 2024.
2. A. Ahmadzadeh, D. J. Kempton, Q. Li, and A. A. Pevtsov, "Solar Filament Segmentation Challenge 2026," *Kaggle Competition*, 2026.
3. A. Kirillov, K. He, R. Girshick, C. Rother, and P. Dollár, "Panoptic Segmentation," in *Proc. CVPR*, 2019, pp. 9404–9413.
4. Z. Zhou, M. M. R. Siddiquee, N. Tajbakhsh, and J. Liang, "UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation," *IEEE TMI*, vol. 39, pp. 1856–1867, 2020.
5. S. S. M. Salehi, D. Erdogmus, and A. Gholipour, "Tversky Loss Function for Image Segmentation Using 3D Fully Convolutional Deep Networks," in *Proc. MLMI*, 2017, pp. 379–387.
6. M. Tan and Q. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," in *Proc. ICML*, 2019, pp. 6105–6114.
7. G. Jocher and J. Qiu, "Ultralytics YOLO11," 2024.
