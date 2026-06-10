# HistDiT: A Structure-Aware Latent Conditional Diffusion Model for High-Fidelity Virtual Staining in Histopathology

[![ICPR 2026](https://img.shields.io/badge/ICPR-2026-blue.svg)](#)
[![arXiv](https://img.shields.io/badge/arXiv-2604.08305-b31b1b.svg)](https://arxiv.org/abs/2604.08305)
[![PyTorch](https://img.shields.io/badge/PyTorch-Implementation-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#)
[![RRPR Badge Eligible](https://img.shields.io/badge/IAPR_TC22-RRPR_Badge-gold.svg)](#)

**Official PyTorch implementation of the paper: "HistDiT: A Structure-Aware Latent Conditional Diffusion Model for High-Fidelity Virtual Staining in Histopathology"**

**Authors:** Aasim Bin Saleem, Amr Ahmed, Ardhendu Behera, Hafeezullah Amin, Iman Yi Liao, Mahmoud Khattab, Pan Jia Wern, Haslina Makmur

**Institution:** Edge Hill University & Collaborating Institutions

---

## Abstract

Immunohistochemistry (IHC) is essential for assessing specific immune biomarkers like Human Epidermal growth-factor Receptor 2 (HER2) in breast cancer. However, the traditional protocols of obtaining IHC stains are resource-intensive, time-consuming, and prone to structural damages. Virtual staining has emerged as a scalable alternative, but it faces significant challenges in preserving fine-grained cellular structures while accurately translating biochemical expressions. Current state-of-the-art methods still rely on Generative Adversarial Networks (GANs) or standard convolutional U-Net diffusion models that often struggle with "structure and staining trade-offs". The generated samples are either structurally relevant but blurry, or texturally realistic but have artifacts that compromise their diagnostic use. Our model outperforms existing baselines, as demonstrated through rigorous quantitative and qualitative evaluations.

## Key Contributions

* **Dual-Stream Conditioning Strategy:** Explicitly maintains a balance between spatial constraints via VAE-encoded latents and semantic phenotype guidance via UNI Foundation Model embeddings.
* **Multi-Objective Loss Function:** Combines an auxiliary L1 term with the standard MSE to produce sharper images with clear morphological structures, mitigating the blurring effects caused by imperfect serial section registrations.
* **Structural Correlation Metric (SCM):** Utilizes SCM to focus purely on the core morphological structure (correlation of variance), effectively correcting the luminance bias inherent in standard SSIM for bright-field microscopy.
* **State-of-the-Art Performance:** Outperforms existing GAN and diffusion baselines on both the BCI and MIST benchmarks, validated by both quantitative metrics and expert qualitative assessments.

## Method Overview

![HistDiT Architecture](assets/Proposed_HistDiT_Architecture.png)

1. **Latent Encoding:** H&E and IHC images are compressed into a spatial latent representation using a frozen AutoencoderKL.
2. **Semantic Extraction:** A pre-trained foundation model (UNI) extracts robust, patch-level semantic embeddings from the H&E input.
3. **Conditioned Generation:** The Diffusion Transformer (HistDiT) iteratively denoises pure Gaussian noise, guided by both the spatial latents (via Cross-Attention) and semantic embeddings (via adaLN).
4. **Decoding:** The denoised latents are decoded back into the pixel space, resulting the final high-fidelity virtual IHC stain.

## Installation

We recommend using Anaconda to manage the environment. The code has been tested with **Python 3.11.13** and **CUDA 12.8**. 

Create and activate the environment:
```bash
conda create -n histdit python=3.11.13 -y
conda activate histdit
```
Install PyTorch and CUDA dependencies:

```bash
conda install -c nvidia -c pytorch cuda-toolkit=12.8 -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
Install the remaining dependencies:
```bash
pip install -r requirements.txt
```

### Direct Pip Installation (For Pre-configured Clusters like RunAI):

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
(Note: You will also need a valid HuggingFace access token to download the UNI model weights during inference).

## Datasets
HistDiT is evaluated on the public BCI benchmark and the MIST dataset. You can access and download the datasets from their official repository:

[Breast Cancer Immunohistochmeical (BCI) Benchmark](https://bupt-ai-cz.github.io/BCI/)

[Multi-Immunohistochemical Stain Transfer (MIST) Dataset](https://drive.google.com/drive/folders/146V99Zv1LzoHFYlXvSDhKmflIL-joo6p)

Unlike PyramidPix2Pix implementation that require concatenated {A,B} image pairs, HistDiT expects the H&E and IHC images to be in separate directories. Please structure your dataset as follows:
- File structure
  ```
  HistDiT
    ├──BCI_dataset
    |    ├── IHC
    |          ├──train
    |          |    ├── 00000_train_1+.png
    |          |    ├── 00001_train_3+.png
    |          |    └── ...
    |          └──test
    |               ├── 00000_test_1+.png
    |               ├── 00001_test_2+.png
    |               └── ...
    ├──MIST_dataset
         ├── HER2
               ├──train
               |    ├── HE
               |    |    ├── 14M2102785_23_16.jpg
               |    |    ├── 14M2102785_31_31.jpg
               |    |    └── ...
               |    ├── IHC
               |         ├── 14M2102785_23_16.jpg
               |         ├── 14M2102785_31_31.jpg
               |         └── ...
               └──test
                    ├── HE
                    |    ├── 15M2102762_21_24.jpg
                    |    └── ...
                    └── IHC
                         ├── 15M2102762_21_24.jpg
                         └── ...
  ```
  A small subset of images is provided in ./sample_data to allow for immediate testing.

## Reproducing Results
1. Download Pre-trained Weights
   Due to file size constraints, the model checkpoints are hosted externally. Download both model.pt and model_ema.pt from [Trained on BCI](https://drive.google.com/drive/folders/1KDbXsPQOXL9m44YjLWuNZLaiU-9-l91G?usp=drive_link) and [Trained on MIST](https://drive.google.com/drive/folders/1KDbXsPQOXL9m44YjLWuNZLaiU-9-l91G?usp=drive_link).
   
   Place both model files inside the ./weights/ directory.
   
3. Run Inference
   The inference.py script automatically loads the test data, applies the dual-conditioning, generates the virtual stains, and neatly sorts the outputs (Generated, EMA Generated, H&E Ground Truth, and IHC Ground Truth) into the ./test_results directory.
   ```bash
   python inference.py
   ```
   (Note: You will also need to insert a valid HuggingFace access token to use the UNI model weights during inference).

4. Evaluate Metrics
   Once inference is complete, calculate standard image translation metrics (LPIPS, FID, PSNR, MSE, and MS-SSIM) by running:
   ```bash
   python evaluate.py
   ```
   Results will be printed to the console and saved in ./test_results/EVALUATION_METRICS.txt.

## Training
The training scripts and full data processing pipelines are currently being packaged for an extended journal submission and will be released in this repository shortly.

## Results Snapshot

HistDiT establishes a new state-of-the-art for virtual IHC staining across two major histopathology benchmarks: the BCI dataset and the MIST dataset.

### Quantitative Results

| Dataset | MSE (↓) | PSNR(dB) (↑) | SSIM (↑) | SCM (↑) | LPIPS (↓) | FID (↓) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BCI Benchmark** | 891.53 | 21.43 | 0.4769 | 0.540 | 0.412 | 49.15 |
| **MIST Dataset** | 3396.88 | 14.26 | 0.211 | 0.302 | 0.489 | 59.30 |

*(For full performance breakdowns, baseline comparisons, and level-wise HER2 expression analysis, please refer to the main paper.)*

### Visual Results

#### BCI Dataset (HER2 Expression Levels 0 to 3+)
![BCI Results](assets/BCI_results.png)
*Qualitative comparison on the BCI Dataset. HistDiT demonstrates higher fidelity and accurate stain intensity, particularly in high-grade regions (2+, 3+).*

#### MIST Dataset
![MIST Results](assets/MIST_results.png)
*Visual comparison on the MIST dataset. HistDiT successfully restores complex staining patterns and sharp morphological details even under unconstrained conditions.*

#### Ablation on Objective Functions
![Ablation Results](assets/Ablation_results.jpg)
*Visual samples demonstrating that our combined hybrid objective (MSE + L1) produces sharper cellular structures compared to the smoothing artifacts seen when using MSE alone.*

## Acknowledgements
We thank the authors of the BCI Dataset and the MIST dataset for providing the public histopathology paired images. The code utilizes components from the HuggingFace Diffusers library and the timm library.

## Citation
If you find this code or research useful in your work, please consider citing our paper:
```
@misc{aasim2026histdit,
      title={HistDiT: A Structure-Aware Latent Conditional Diffusion Model for High-Fidelity Virtual Staining in Histopathology}, 
      author={Aasim Bin Saleem and Amr Ahmed and Ardhendu Behera and Hafeezullah Amin and Iman Yi Liao and Mahmoud Khattab and Pan Jia Wern and Haslina Makmur},
      year={2026},
      eprint={2604.08305},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
