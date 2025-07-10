# RPFNet
Source code of the paper ***"Residual Prior-driven Frequency-aware Network for Image Fusion"*** which has been accepted by ***ACM MM 2025***.
- [[Arxive](https://arxiv.org/abs/2507.06735)]
- Guan Zheng, Xue Wang, Wenhua Qian, Peng Liu, RunZhuo Ma

# Abstract
Image fusion aims to integrate complementary information across modalities to generate high-quality fused images, thereby enhancing the performance of high-level vision tasks. While global spatial modeling mechanisms show promising results, constructing long-range feature dependencies in the spatial domain incurs substantial computational costs. Additionally, the absence of ground-truth exacerbates the difficulty of capturing complementary features effectively. To tackle these challenges, we propose a Residual Prior-driven Frequency-aware Network, termed as RPFNet. Specifically, RPFNet employs a dual-branch feature extraction framework: the Residual Prior Module (RPM) extracts modality-specific difference information from residual maps, thereby providing complementary priors for fusion; the Frequency Domain Fusion Module (FDFM) achieves efficient global feature modeling and integration through frequency-domain convolution. Additionally, the Cross Promotion Module (CPM) enhances the synergistic perception of local details and global structures through bidirectional feature interaction. During training, we incorporate an auxiliary decoder and saliency structure loss to strengthen the model’s sensitivity to modality-specific differences. Furthermore, a combination of adaptive weight-based frequency contrastive loss and SSIM loss effectively constrains the solution space, facilitating the joint capture of local details and global features while ensuring the retention of complementary information. Extensive experiments validate the fusion performance of RPFNet, which effectively integrates discriminative features, enhances texture details and salient objects, and can effectively facilitate the deployment of the high-level vision task.

# :triangular_flag_on_post: Testing
If you want to infer with our RPFNet and obtain the fusion results in our paper, please run ```test.py```.
Then, the fused results will be saved in the ```'./Fused/'``` folder.

# :triangular_flag_on_post: Training
You can change your own data address in ```dataset.py``` and use ```train.py``` to retrain the method.
