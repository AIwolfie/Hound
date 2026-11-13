# HOUND — Phase 7: Model Provenance & Legal Licensing Audit

**Document ID:** `HOUND-PROVENANCE-PHASE-7`  
**Classification:** Academic & Legal Provenance Register  
**Author:** AIwolfie  
**Status:** Complete & Approved  
**Date:** November 13, 2026  
**System:** HOUND Visual Place Recognition Matrix  

---

## 1. Provenance Register of Integrated VPR Models

| Model Attribute | CosPlace | AnyLoc | MixVPR | ResNet-18 Baseline | Hybrid Color-Spatial |
|:---|:---|:---|:---|:---|:---|
| **Model Name** | `CosPlace_ResNet18_512` | `AnyLoc_dinov2_vits14` | `MixVPR_resnet50_512` | `ResNet18_ImageNet_Baseline` | `Hybrid_Color_Spatial_1224` |
| **Full Paper Title** | *Rethinking Visual Geo-localization for Large-Scale Applications* | *AnyLoc: Towards Universal Visual Place Recognition* | *MixVPR: Feature Mixing for Visual Place Recognition* | *Deep Residual Learning for Image Recognition* | *HOUND Spatial Color Pyramid & Steerable Edge Descriptor* |
| **Authors** | Gabriele Berton, Carlo Masone, Barbara Caputo | Nikhil Keetha, Avneesh Mishra, Jay Karhade, Kshitij Minhas, Sourav Garg, Sebastian Scherer | Amar Ali-bey, Brahim Chaib-draa, Philippe Giguere | Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun | HOUND Core Engineering Team |
| **Conference / Journal** | IEEE/CVF CVPR 2022 | IEEE RA-L / RSS 2023 | IEEE/CVF WACV 2023 | IEEE/CVF CVPR 2016 | HOUND Internal Specification (2026) |
| **Official Repository** | [github.com/gmberton/CosPlace](https://github.com/gmberton/CosPlace) | [github.com/AnyLoc/AnyLoc](https://github.com/AnyLoc/AnyLoc) | [github.com/AmarAliBey/MixVPR](https://github.com/AmarAliBey/MixVPR) | [github.com/pytorch/vision](https://github.com/pytorch/vision) | [github.com/AIwolfie/Hound](https://github.com/AIwolfie/Hound) |
| **License** | **MIT License** | **Apache 2.0** | **MIT License** | **BSD-3-Clause** | **MIT License** |
| **Training Dataset** | SF-XL (41.2M images, 5M cells) | LVD-142M (Foundation data) | GSV-Cities (560k images, 67k places) | ImageNet-1k (1.28M images, 1k classes) | Zero (Algorithmic filter) |
| **Embedding Dimension** | $512\text{ dimensions}$ | $384\text{ dimensions}$ | $512\text{ dimensions}$ | $512\text{ dimensions}$ | $1224\text{ dimensions}$ |
| **Weights Source** | Official GitHub Releases | Official Meta AI DINOv2 S3 | MixVPR Official Releases | PyTorch Torchvision Weights | Algorithmic (No weights) |
| **Weights File / Cache** | `ResNet18_512_cosplace.pth` | `dinov2_vits14_pretrain.pth` | Dynamic PyTorch Architecture | `resnet18-f37072fd.pth` | Code only |
| **Local Checkpoint Size** | **$43.7\text{ MB}$** | **$84.2\text{ MB}$** | **$97.8\text{ MB}$** | **$44.7\text{ MB}$** | **$0\text{ MB}$** |
| **Inference Device** | CUDA / CPU Fallback | CUDA / CPU Fallback | CUDA / CPU Fallback | CUDA / CPU Fallback | CPU (Vectorized NumPy) |

---

## 2. Licensing Compatibility Audit

All integrated models and libraries were audited against HOUND's **MIT License**:

1. **CosPlace (MIT License):** Fully permissive. Commercial, modification, distribution, and private use allowed without copyleft obligations.
2. **AnyLoc / DINOv2 (Apache 2.0):** Permissive open-source license with patent grant. Compatible with MIT-licensed downstream applications.
3. **MixVPR (MIT License):** Fully permissive.
4. **PyTorch & Torchvision (BSD-3-Clause):** Fully compatible with MIT license.
5. **FAISS (MIT License):** Fully permissive open-source license.
6. **OpenCV (Apache 2.0):** SIFT keypoint patents expired in 2020; SIFT is now standard Apache 2.0 licensed in OpenCV.

**Audit Finding:** Zero proprietary, non-commercial (CC BY-NC), or copyleft (GPL) restrictions exist in HOUND Phase 7. The entire visual place recognition pipeline is legally deployable in open-source, humanitarian, and authorized security contexts.
