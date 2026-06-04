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

[cite_start]Immunohistochemistry (IHC) is essential for assessing specific immune biomarkers like Human Epidermal growth-factor Receptor 2 (HER2) in breast cancer[cite: 7]. [cite_start]However, the traditional protocols of obtaining IHC stains are resource-intensive, time-consuming, and prone to structural damages[cite: 8]. [cite_start]Virtual staining has emerged as a scalable alternative, but it faces significant challenges in preserving fine-grained cellular structures while accurately translating biochemical expressions[cite: 9]. [cite_start]Current state-of-the-art methods still rely on Generative Adversarial Networks (GANs) or standard convolutional U-Net diffusion models that often struggle with "structure and staining trade-offs"[cite: 10]. [cite_start]The generated samples are either structurally relevant but blurry, or texturally realistic but have artifacts that compromise their diagnostic use[cite: 11]. 

[cite_start]In this paper, we introduce HistDiT, a novel latent conditional Diffusion Transformer (DiT) architecture that establishes a new benchmark for visual fidelity in virtual histological staining[cite: 12]. [cite_start]The novelty introduced in this work is, a) the Dual-Stream Conditioning strategy that explicitly maintains a balance between spatial constraints via VAE-encoded latents and semantic phenotype guidance via UNI embeddings [cite: 13][cite_start]; b) the multi-objective loss function that contributes to sharper images with clear morphological structure [cite: 14][cite_start]; and c) the use of the Structural Correlation Metric (SCM) to focus on the core morphological structure for precise assessment of sample quality[cite: 15]. [cite_start]Consequently, our model outperforms existing baselines, as demonstrated through rigorous quantitative and qualitative evaluations[cite: 16].

## Key Contributions

* [cite_start]**Dual-Stream Conditioning Strategy:** Explicitly maintains a balance between spatial constraints via VAE-encoded latents and semantic phenotype guidance via UNI Foundation Model embeddings[cite: 13, 140, 144].
* [cite_start]**Multi-Objective Loss Function:** Combines an auxiliary L1 term with the standard MSE to produce sharper images with clear morphological structures, mitigating the blurring effects caused by imperfect serial section registrations[cite: 14, 43, 44].
* [cite_start]**Structural Correlation Metric (SCM):** Utilizes SCM to focus purely on the core morphological structure (correlation of variance), effectively correcting the luminance bias inherent in standard SSIM for bright-field microscopy[cite: 15, 45, 46, 202].
* [cite_start]**State-of-the-Art Performance:** Outperforms existing GAN and diffusion baselines on both the BCI and MIST benchmarks, validated by both quantitative metrics and expert qualitative assessments[cite: 16, 47, 48].

## Method Overview

![HistDiT Architecture](assets/Proposed_DiT_Architecture.png)

1. [cite_start]**Latent Encoding:** H&E images are compressed into a spatial latent representation using a frozen AutoencoderKL[cite: 136, 137].
2. [cite_start]**Semantic Extraction:** A pre-trained foundation model (UNI) extracts robust, patch-level semantic embeddings from the H&E input[cite: 144, 145].
3. [cite_start]**Conditioned Generation:** The Diffusion Transformer (HistDiT) iteratively denoises pure Gaussian noise, heavily guided by both the spatial latents (via Cross-Attention) and semantic embeddings (via adaLN)[cite: 140, 146, 155].
4. [cite_start]**Decoding:** The denoised latents are decoded back into the pixel space, yielding the final high-fidelity virtual IHC stain[cite: 138].

## Installation

We recommend using Anaconda to manage the environment. The code has been tested with **Python 3.11.13** and **CUDA 12.8**. 

Create and activate the environment:
```bash
conda create -n histdit python=3.11.13 -y
conda activate histdit
Install PyTorch and CUDA dependencies:Bashconda install -c nvidia -c pytorch cuda-toolkit=12.8 -y
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu128](https://download.pytorch.org/whl/cu128)
Install Mamba and Causal-Conv1d requirements:Bashconda install -c conda-forge git -y
python -m pip install git+[https://github.com/Dao-AILab/causal-conv1d.git](https://github.com/Dao-AILab/causal-conv1d.git)
python -m pip install git+[https://github.com/state-spaces/mamba.git](https://github.com/state-spaces/mamba.git)
Install the remaining dependencies:Bashpip install -r requirements.txt
(Note: You will also need a valid HuggingFace access token to download the UNI model weights during inference).DatasetsHistDiT is evaluated on the public BCI (Breast Cancer Immunohistochemical) dataset and the MIST dataset.  You can download the BCI dataset from the Official BCI Homepage.Unlike Pix2Pix implementations that require concatenated {A,B} image pairs, HistDiT expects the H&E and IHC images to be in separate directories. Please structure your dataset as follows:Plaintext<root_path>/BCI_dataset/HE/test/00000_test_1+.png
<root_path>/BCI_dataset/IHC/test/00000_test_1+.png
A small subset of images is provided in ./sample_data to allow for immediate testing.Reproducing Results1. Download Pre-trained WeightsDue to file size constraints, the model checkpoints are hosted externally.Download model.ckpt and model_ema.ckpt from: [Insert Google Drive / Zenodo Link Here]Place both files inside the ./weights/ directory.2. Run InferenceThe inference.py script automatically loads the test data, applies the dual-conditioning, generates the virtual stains, and neatly sorts the outputs (Generated, EMA Generated, H&E Ground Truth, and IHC Ground Truth) into the ./test_results directory.Bashpython inference.py
3. Evaluate MetricsOnce inference is complete, calculate standard image translation metrics (LPIPS, FID, PSNR, MSE, and MS-SSIM) by running:Bashpython evaluate.py
Results will be printed to the console and saved in ./test_results/EVALUATION_METRICS.txt.TrainingThe training scripts and full data processing pipelines are currently being packaged for an extended journal submission and will be released in this repository shortly.AcknowledgementsWe thank the authors of the BCI Dataset and the MIST dataset for providing the public histopathology paired images.This codebase utilizes components from the HuggingFace Diffusers library and the timm library.CitationIf you find this code or research useful in your work, please consider citing our paper:Code snippet@misc{saleem2026histdit,
      title={HistDiT: A Structure-Aware Latent Conditional Diffusion Model for High-Fidelity Virtual Staining in Histopathology}, 
      author={Aasim Bin Saleem and Amr Ahmed and Ardhendu Behera and Hafeezullah Amin and Iman Yi Liao and Mahmoud Khattab and Pan Jia Wern and Haslina Makmur},
      year={2026},
      eprint={2604.08305},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
