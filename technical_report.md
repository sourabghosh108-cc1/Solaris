# A Two-Stage Deep Learning Framework for Automated Solar Filament Segmentation from Synoptic H-Alpha Observations

**Technical Report for the Solar Filament Segmentation Challenge 2026 (MAGFiLO 1.0)**  
*IEEE BigData 2026 BigDataCup / National Solar Observatory (NSO) / NSF*

**Authors:** Sourab Ghosh, et al.  
**Affiliation:** Independent Space AI Research / Competition Team  
**Kaggle Notebook:** `sfscs` (Version 1) | **Public Leaderboard Score:** `0.3700`

---

## Abstract

Accurate automated segmentation of solar filaments from ground-based H-$\alpha$ observations is fundamental for modeling coronal magnetic topology and predicting space weather hazards such as Coronal Mass Ejections (CMEs) and solar flares. However, solar filament morphology exhibits complex, thread-like structures (barbs), high spatial variance, and observational noise across synoptic telescope stations. In this work, we present a two-stage deep learning framework tailored for the MAGFiLO 1.0 benchmark dataset. **Stage 1** deploys an Ultralytics YOLO11m detector for robust candidate localization of filament bounding boxes across full-disk solar images. **Stage 2** applies a deep UNet++ architecture with an EfficientNet-B4 encoder on contextually padded ($20\%$) filament crops resized to $512 \times 512$. To mitigate severe class imbalance and prioritize faint boundary features, we train the segmenter using a composite **BCETverskyLoss** ($\alpha=0.3, \beta=0.7$). A confidence-ranked greedy panoptic painter enforces strictly disjoint, non-overlapping instance masks in compliance with competition requirements. On the validation set, our pipeline achieves a Panoptic Quality ($PQ$) of **$0.4145$** (Detection Quality $DQ = 0.6093$, Segmentation Quality $SQ = 0.6802$, best crop Dice $= 0.8433$), securing a public leaderboard score of **$0.3700$** and surpassing the human annotator agreement benchmark ($\approx 0.36$).

**Keywords:** Solar Filaments, MAGFiLO 1.0, GONG H-Alpha, Two-Stage Segmentation, YOLO11, UNet++, BCETverskyLoss, Panoptic Quality.

---

## 1. Introduction and Background

Solar filaments are dense, cool plasma structures suspended by helical magnetic field lines in the solar corona above photospheric polarity inversion lines (PILs). When these magnetic channels destabilize, they erupt violently, producing Coronal Mass Ejections (CMEs) that pose severe geomagnetic threats to satellites, aviation, and power distribution grids.

Automated delineation of filaments from ground-based Global Oscillations Network Group (GONG) H-$\alpha$ synoptic observations ($656.3\text{ nm}$) is critical for space weather forecasting. However, automated filament segmentation poses key domain challenges:
1. **Fine-Scale Barb Delineation**: Thin, lateral barbs with faint contrast branch from primary filament spines.
2. **Observational Artifacts & Limb Darkening**: Varying atmospheric turbulence and non-uniform solar disk illumination across ground stations.
3. **Fragmentation and Over-Merging**: Algorithms risk fragmenting contiguous filaments (one-to-many penalty) or merging neighboring distinct filaments (many-to-one penalty).
4. **Annotator Ambiguity**: Diffuse boundaries cause human inter-annotator agreement to average only $\approx 0.36$.

To tackle these challenges, we develop a decoupled, two-stage deep neural network architecture optimizing the **Panoptic Quality (PQ)** metric.

---

## 2. Dataset and Problem Formulation

The **MAGFiLO 1.0** dataset contains $2048 \times 2048$ single-channel grayscale GONG observations. The training set provides $707$ unique solar images ($600$ train / $107$ validation) with $976$ multi-annotator record sets comprising $8,199$ filament instances. The test set comprises $180$ full-disk observations.

### Evaluation Metric
Performance is measured via **Panoptic Quality (PQ)**:

$$\text{PQ} = \text{DQ} \times \text{SQ} = \left( \frac{|\text{TP}|}{|\text{TP}| + \frac{1}{2}|\text{FP}| + \frac{1}{2}|\text{FN}|} \right) \times \left( \frac{\sum_{(p, g) \in \text{TP}} \text{IoU}(p, g)}{|\text{TP}|} \right)$$

where a predicted mask $p$ and ground truth mask $g$ form a True Positive ($\text{TP}$) if $\text{IoU}(p, g) > 0.5$. Non-overlapping instance disjointness is strictly required ($p_i \cap p_j = \emptyset$).

---

## 3. Proposed Methodology

```
+-----------------------------------------------------------------------------+
|                          System Architecture Flow                           |
+-----------------------------------------------------------------------------+
|                                                                             |
|  [ Full Disk Image (2048x2048) ]                                            |
|         │                                                                   |
|         ▼                                                                   |
|  [ Stage 1: YOLO11m Detector ]                                              |
|         │   • Trained @ 1280x1280, AdamW lr=0.00087, 40 Epochs              |
|         │   • Inference @ 1024x1024, Candidate Box Extraction (conf >= 0.05)|
|         ▼                                                                   |
|  [ Bounding Box Proposals + 20% Contextual Margin ]                         |
|         │                                                                   |
|         ▼                                                                   |
|  [ Stage 2: UNet++ (EfficientNet-B4) @ 512x512 ]                            |
|         │   • Single-channel grayscale input                                |
|         │   • BCETverskyLoss (0.5 BCE + 0.5 Tversky with α=0.3, β=0.7)      |
|         │   • Batch size 8, Cosine LR scheduling, 12 Epochs                 |
|         ▼                                                                   |
|  [ Greedy Panoptic Assembly & Disjoint Mask Assignment ]                    |
|         │   • 2D Threshold Sweep on Validation Subset                       |
|         │   • Optimal Parameters: conf = 0.35, mask_thr = 0.60              |
|         │   • Sequential Pixel Claiming by Confidence Score                 |
|         ▼                                                                   |
|  [ Non-Overlapping Instance Masks -> COCO RLE (submission.csv) ]            |
+-----------------------------------------------------------------------------+
```

### 3.1 Stage 1: Candidate Detection (YOLO11m)
We train an Ultralytics **YOLO11m** detector on $1280 \times 1280$ images for $40$ epochs with AdamW ($\text{lr}_0 = 8.7 \times 10^{-4}$, batch size $8$). The detector extracts candidate bounding boxes covering filament spines and barbs. At inference, candidate boxes are extracted at $\text{conf}_{\text{min}} = 0.05$ and $\text{max\_det} = 60$.

Validation detector performance:
- $\text{mAP}_{50}$: **$0.6369$**
- $\text{mAP}_{50-95}$: **$0.3760$**
- Precision: **$0.6154$** | Recall: **$0.6346$**

### 3.2 Stage 2: Crop Segmentation (UNet++ with EfficientNet-B4)
Candidate bounding boxes are expanded with a $20\%$ contextual margin ($\text{PAD\_RATIO} = 0.20$) and resized to $512 \times 512$. 

The model uses a **UNet++** architecture with an ImageNet-pretrained **EfficientNet-B4** encoder adapted for single-channel grayscale images. Augmentations include horizontal/vertical flips, $90^\circ$ rotations, and brightness/contrast adjustments.

### 3.3 Loss Formulation: BCETverskyLoss
To address severe class imbalance and penalize false negatives on thin filament barbs, we use a composite **BCETverskyLoss**:

$$\mathcal{L}_{\text{total}} = 0.5 \cdot \mathcal{L}_{\text{BCE}} + 0.5 \cdot \mathcal{L}_{\text{Tversky}}$$

$$\mathcal{L}_{\text{Tversky}} = 1 - \frac{\text{TP} + \epsilon}{\text{TP} + \alpha \text{FP} + \beta \text{FN} + \epsilon}$$

with $\alpha = 0.30$ and $\beta = 0.70$ ($\beta > \alpha$).

The segmenter achieves a best validation crop Dice of **$0.8433$** (epoch 10 of 12).

### 3.4 Panoptic Assembly and Inference
1. Padded crops are processed through UNet++ in batched forward passes to generate sigmoid probability maps.
2. Probability maps are resized to full-image coordinate space.
3. **Threshold Optimization**: A grid search over validation images identifies optimal hyperparameters: $\text{conf} = 0.35, \text{mask\_thr} = 0.60$.
4. **Greedy Disjoint Assignment**: Predictions are sorted by confidence; pixels are claimed by the highest-confidence filament, guaranteeing $M_i \cap M_j = \emptyset$.
5. Validated binary masks are encoded into compressed COCO RLE format.

---

## 4. Experimental Results

### 4.1 Quantitative Performance

| Method | Validation PQ | Validation DQ | Validation SQ | Crop Dice | Public Leaderboard |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Human Inter-Annotator Agreement | $\approx 0.3600$ | — | — | — | — |
| **Our Two-Stage Pipeline (V1)** | **0.4145** | **0.6093** | **0.6802** | **0.8433** | **0.3700** |

Detailed validation breakdown ($107$ validation photos):
- True Positives ($\text{TP}$): $815$
- False Positives ($\text{FP}$): $466$
- False Negatives ($\text{FN}$): $579$
- **Detection Quality ($\text{DQ}$):** $0.6093$
- **Segmentation Quality ($\text{SQ}$):** $0.6802$
- **Panoptic Quality ($\text{PQ}$):** $0.4145$
- **Public Leaderboard Score:** **$0.3700$** ($1,152$ predicted instances across $177/180$ test images)

Our solution outperforms the human inter-annotator agreement baseline ($\approx 0.36$), demonstrating the effectiveness of the two-stage detection and crop segmentation pipeline.

---

## 5. Conclusion

We presented an effective two-stage deep learning pipeline for automated solar filament segmentation on the MAGFiLO 1.0 dataset. By coupling a YOLO11m detector with a UNet++ (EfficientNet-B4) crop segmenter trained under BCETversky loss, combined with greedy panoptic disjoint assignment, our pipeline achieves strong performance surpassing human annotator agreement benchmarks.

---

## References

1. A. Ahmadzadeh, et al., "A dataset of manually annotated filaments from H-alpha observations," *Nature Scientific Data*, vol. 11, no. 1, p. 1031, 2024.
2. A. Ahmadzadeh, D. J. Kempton, Q. Li, and A. A. Pevtsov, "Solar Filament Segmentation Challenge 2026," *Kaggle Competition*, 2026.
3. A. Kirillov, K. He, R. Girshick, C. Rother, and P. Dollár, "Panoptic Segmentation," in *Proc. CVPR*, 2019, pp. 9404–9413.
4. Z. Zhou, M. M. R. Siddiquee, N. Tajbakhsh, and J. Liang, "UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation," *IEEE TMI*, vol. 39, no. 6, pp. 1856–1867, 2020.
5. S. S. M. Salehi, D. Erdogmus, and A. Gholipour, "Tversky Loss Function for Image Segmentation Using 3D Fully Convolutional Deep Networks," in *Proc. MLMI*, 2017, pp. 379–387.
6. M. Tan and Q. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," in *Proc. ICML*, 2019, pp. 6105–6114.
7. G. Jocher and J. Qiu, "Ultralytics YOLO11," 2024. [Online]. Available: https://github.com/ultralytics/ultralytics.
