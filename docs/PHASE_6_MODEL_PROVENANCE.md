# HOUND — Phase 6: Visual Model Provenance & Place Recognition Architecture Audit

**Document ID:** `HOUND-PROVENANCE-PHASE-6`  
**Classification:** Scientific Specification & Architectural Audit  
**Author:** AIwolfie  
**Status:** Approved & Verified  
**Date:** November 12, 2026  
**Hardware Tested:** Intel Core i5-12450H, 16 GB DDR4 RAM, NVIDIA GeForce RTX 2050 (4 GB VRAM, CUDA 11.8)  

---

## 1. Overview & Objective

Phase 6 audits the provenance, mathematical properties, inference characteristics, and failure modes of all visual representation and place recognition algorithms in HOUND.

A core finding of Phase 5.1 and Phase 6 is that **general-purpose computer vision backbones pretrained on object classification (ImageNet-1k) are fundamentally distinct from domain-specific Visual Place Recognition (VPR) models**. This document provides an unsparing analysis of each visual component currently in HOUND, measures its real-world performance on host hardware, and evaluates modern VPR architectures for future integration.

---

## 2. Inventory of Current Visual Models & Matchers

| Method / Component | Architecture / Algorithm | Weights Provenance & Dataset | Live Agent Usage | Execution Device | Memory Footprint | Per-Sample Latency |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| **`HybridColorSpatialDescriptor`** | 1224-d L2 vector ($3\times 3$ HSV pyramid + steerable Sobel) | Handcrafted algorithm; zero learned weights | **Active Default** in `visual_place_search` | CPU | $< 1\text{ MB}$ | **$3.10\text{ ms}$** |
| **`NeuralGlobalDescriptor (ResNet-18)`** | PyTorch CNN backbone, pooled 512-d feature vector | ImageNet-1k classification (`resnet18-f37072fd.pth`, 44.7 MB) | Standby / Optional | CPU / **CUDA** | $\sim 180\text{ MB}$ | **$6.99\text{ ms}$ (CUDA)**<br>$31.70\text{ ms}$ (CPU) |
| **`NeuralGlobalDescriptor (ResNet-50)`** | PyTorch CNN backbone, pooled 2048-d feature vector | ImageNet-1k classification (`resnet50-11ad3fa6.pth`, 97.8 MB) | Standby / Optional | CPU / **CUDA** | $\sim 360\text{ MB}$ | **$10.80\text{ ms}$ (CUDA)**<br>$68.40\text{ ms}$ (CPU) |
| **`LocalFeatureMatcher (ORB)`** | FAST keypoint detector + BRIEF binary descriptor (1000 features) | Algorithmic; zero weights; Hamming distance | Active in geometric verification | CPU | $< 2\text{ MB}$ | **$3.95\text{ ms}$** |
| **`LocalFeatureMatcher (SIFT)`** | Difference-of-Gaussians + gradient orientation histogram | Algorithmic; zero weights; L2 distance | Optional fallback | CPU | $< 5\text{ MB}$ | **$82.15\text{ ms}$** |
| **`RANSAC Homography`** | Random Sample Consensus projective transformation matrix | Geometric optimizer ($\ge 10$ inliers, $\ge 0.25$ ratio) | Active in geometric verification | CPU | $< 0.1\text{ MB}$ | **$< 0.5\text{ ms}$** |

---

## 3. Deep Architectural Teardown & Provenance

### 3.1. `HybridColorSpatialDescriptor`
- **Implementation File:** [`core/visual_place_recognition.py`](file:///D:/project/Hound/core/visual_place_recognition.py#L49-L110)
- **Mathematical Formulation:**
  1. Input frame normalized to $384 \times 256$ RGB.
  2. Image divided into a $3\times 3$ spatial grid (9 cells of $128 \times 85$ pixels).
  3. For each cell, computes a 3D HSV color histogram with 8 Hue bins, 4 Saturation bins, and 4 Value bins ($8 \times 4 \times 4 = 128$ dimensions).
  4. Computes steerable Sobel gradients ($G_x, G_y$) and aggregates gradient magnitudes into an 8-bin angular orientation histogram ($0^\circ\text{--}360^\circ$).
  5. Concatenates features into a single composite vector:
     $$\mathbf{v} \in \mathbb{R}^{9 \times (128 + 8)} = \mathbb{R}^{1224}$$
  6. Applies Euclidean L2 normalization: $\mathbf{\hat{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$.
- **Strengths:** Completely deterministic, zero network dependency, runs in $3.10\text{ ms}$ on CPU, effective at capturing coarse horizon splits (sky top, terrain bottom, coastal blue).
- **Failure Conditions:** Highly sensitive to illumination shifts (brightness $\pm 40\%$ drops cosine similarity from $1.00$ to $0.30$), seasonal color transitions (snow on ground drops similarity to $0.45$), and camera exposure variations.

### 3.2. `NeuralGlobalDescriptor` (ResNet-18 & ResNet-50)
- **Implementation File:** [`core/visual_place_recognition.py`](file:///D:/project/Hound/core/visual_place_recognition.py#L112-L175)
- **Model Checkpoint Provenance:**
  - `ResNet-18`: Downloaded from `https://download.pytorch.org/models/resnet18-f37072fd.pth` via `torchvision.models.resnet18(weights="DEFAULT")`. File size: 44.7 MB.
  - `ResNet-50`: Downloaded from `https://download.pytorch.org/models/resnet50-11ad3fa6.pth` via `torchvision.models.resnet50(weights="DEFAULT")`. File size: 97.8 MB.
  - License: BSD 3-Clause.
  - Checkpoint Directory: `C:\Users\mayan\.cache\torch\hub\checkpoints/`.
- **Training Dataset:** ImageNet-1k (ILSVRC 2012, 1.28 million images across 1,000 object categories).
- **Preprocessing Pipeline:**
  1. Input frame converted to PIL RGB.
  2. Resized to $224 \times 224$ pixels.
  3. Scaled to $[0.0, 1.0]$ tensor.
  4. Standard ImageNet channel normalization:
     $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
  5. Feature pooling: Classification head (`model.fc`) replaced with `torch.nn.Identity()`. Features extracted from global average pooling layer.
- **Hardware Acceleration:**
  - Auto-selects CUDA if available: runs on host `NVIDIA GeForce RTX 2050` (Device 0).
  - ResNet-18 inference latency: **$6.99\text{ ms}$** per frame (warm).
  - ResNet-50 inference latency: **$10.80\text{ ms}$** per frame (warm).
  - Automatic CPU fallback: verified functional without crashing when CUDA is disabled ($31.70\text{ ms}$ and $68.40\text{ ms}$).

### 3.3. Why ImageNet-Pretrained ResNet is NOT a Visual Place Recognition Model

In computer vision literature, using standard ImageNet CNN weights for visual geolocalization is known to suffer from severe domain mismatch:
1. **Semantic Invariance vs Instance Specificity:**
   ImageNet models are trained to achieve *class invariance* (e.g. recognizing any dog or car regardless of background). Visual place recognition requires the exact opposite: *instance-level discrimination* (distinguishing this specific Eiffel Tower or this specific mountain pass from other towers and mountains, while ignoring transient objects like cars, people, and weather).
2. **Feature Pooling Collapse:**
   Standard average pooling weights all spatial regions equally. In geolocalization, sky regions and moving foreground objects (buses, pedestrians) corrupt global descriptors. Dedicated VPR architectures employ **Generalized Mean (GeM) pooling** or **NetVLAD cluster pooling** with learnable attention masks to downweight transient features.
3. **Loss Function Objective:**
   ImageNet backbones are trained with cross-entropy loss over categorical labels. State-of-the-art geolocalization models are trained with contrastive, triplet, or multi-similarity margin losses over metric geographic embeddings (e.g., pulling images taken within $25\text{ m}$ together while pushing images taken $>500\text{ m}$ apart away).

---

## 4. Modern Visual Place Recognition (VPR) Literature Review

To advance HOUND toward true open-world visual place recognition, we evaluated four dedicated geolocalization architectures from recent computer vision literature:

| Architecture | Backbone & Pooling | Training Dataset | Model Size | License | Suitability for HOUND |
|:---|:---|:---|:---:|:---:|:---|
| **AnyLoc (2023)** | DINOv2 (ViT-S / ViT-B / ViT-G) + GeM / VLAD | Zero-shot self-supervised (LVD-142M) | 88 MB (ViT-S) to 1.1 GB (ViT-L) | Apache 2.0 | **Highest.** Universal zero-shot descriptor across indoor, urban, and aerial domains without site-specific training. |
| **CosPlace (2022)** | ResNet-50 / ConvNeXt + GeM pooling | San Francisco eXtreme + Pittsburgh 250k (Large Margin Cosine) | 98 MB | MIT | **High.** Lightweight, extremely fast GPU inference ($<8\text{ ms}$), high recall on urban street views. |
| **MixVPR (2023)** | ResNet-50 + Feature Mixer MLP blocks | GSV-Cities (560,000 georeferenced panoramic images) | 105 MB | MIT | **High.** State-of-the-art city retrieval accuracy on Pittsburgh, Tokyo TM, and MSLS benchmarks. |
| **NetVLAD (2016)** | VGG-16 / ResNet + Soft assignment cluster centroids | Pittsburgh 250k / Tokyo 24/7 | 120 MB | MIT | **Moderate.** Classical standard, robust across day-to-night transitions, but larger memory footprint and slower inference. |

### Recommendation for Phase 7
Integrate **AnyLoc (DINOv2-ViT-S)** or **CosPlace (ResNet-50)** as an optional neural engine plugin:
- Both operate under permissive licenses (Apache 2.0 / MIT).
- Model weights fit comfortably within the 4 GB VRAM ceiling of the host RTX 2050.
- Preserves 100% CPU fallback using the deterministic `HybridColorSpatialDescriptor`.

---

## 5. Local Geometric Feature Verification: ORB vs SIFT

Local feature matching provides the critical second-stage verification filter after global descriptor retrieval:

```text
Query Image (1024x768) ───► Keypoint Detection (SIFT or ORB)
                                      │
Reference Image        ───► Feature Extraction (128-d float or 32-byte binary)
                                      │
                             KNN Matcher (k=2)
                                      │
                             Lowe's Ratio Test (threshold = 0.75 / 0.85)
                                      │
                             RANSAC Homography Estimation (reproj_thresh = 5.0)
                                      │
                             Verified if: inliers >= 10 AND inlier_ratio >= 0.25
```

### Empirical Latency & Inlier Benchmarks on Real Photographs

| Test Pair | Metric | ORB Matcher (CPU) | SIFT Matcher (CPU) | Verified Status |
|:---|:---|:---:|:---:|:---:|
| **Eiffel Tower (Real Photograph vs Prototype)** | Inliers / Ratio | $18\text{ inliers}$ / $0.32$ | $12\text{ inliers}$ / $0.28$ | **Verified** |
| **Eiffel Tower (Brightened +40%)** | Inliers / Ratio | $142\text{ inliers}$ / $0.99$ | $41\text{ inliers}$ / $1.00$ | **Verified** |
| **Eiffel Tower (Rotated 15°)** | Inliers / Ratio | $53\text{ inliers}$ / $0.71$ | $22\text{ inliers}$ / $0.85$ | **Verified** |
| **Eiffel Tower (Rotated 60°)** | Inliers / Ratio | $18\text{ inliers}$ / $0.62$ | $8\text{ inliers}$ / $0.50$ | **Verified (ORB)** |
| **Unrelated Scene (Shinjuku vs Eiffel)** | Inliers / Ratio | **$0\text{ inliers}$ / $0.00$** | **$0\text{ inliers}$ / $0.00$** | **False Match Rejected** |
| **Execution Latency per Verification** | Milliseconds | **$3.95\text{ ms}$** | **$82.15\text{ ms}$** | **ORB is 20.8x faster** |

**Conclusion:** ORB is selected as the primary geometric verification engine for live investigations due to its sub-4ms execution speed and rotation invariance. SIFT is preserved for offline or deep-forensic analysis when maximum sub-pixel keypoint accuracy is required.
