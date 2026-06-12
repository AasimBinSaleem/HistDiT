import os
import copy
import timm
import torch
import logging
import numpy as np
from tqdm import tqdm
from PIL import Image
from models.utils import *
#from dataset import bcidatasets
from dataset import mistdatasets
from torchvision import transforms
from models.utils import pil_loader
from timm.layers import SwiGLUPacked
import torchvision.transforms.v2 as TF
from timm.data import resolve_data_config
from diffusers import DDPMScheduler, AutoencoderKL
from timm.data.transforms_factory import create_transform

# Custom modules (Ensure these are in your models/ directory)
from models.backbone import HistoDiT
from models.utils import save_single_image, count_parameters

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

# Note: Users will need their own HuggingFace token to access the UNI model.
from huggingface_hub import login
login(token="YOUR_HF_TOKEN")

def sample_images(model, c_spatial, c_semantic, device, noise_scheduler, cfg_scale, strength):
    logging.info(f"Sampling {c_spatial.shape[0]} new images with strength={strength}")
    model.eval()
    with torch.no_grad():
        num_steps = noise_scheduler.config.num_train_timesteps
        t_start = int(strength * num_steps)
        t_start = max(1, t_start)
        
        x = torch.randn_like(c_spatial).to(device)
        
        for t in tqdm(reversed(range(1, t_start)), desc="Sampling Progress", position=0):
            t_tensor = torch.full((x.shape[0],), t, device=device, dtype=torch.long)
            predicted_noise = model(x, t_tensor, c_spatial, c_semantic)
            
            if cfg_scale > 0:
                uncond_predicted_noise = model(x, t_tensor, None, None)
                predicted_noise = (1 + cfg_scale) * predicted_noise - cfg_scale * uncond_predicted_noise
                
            step_output = noise_scheduler.step(predicted_noise, t_tensor[0].item(), x)
            x = step_output.prev_sample
            
    model.train()
    return x

def get_test_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(args.image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    #dataset = bcidatasets.BCIDataset(x_img_dir=args.x_image_dir_path, y_img_dir=args.y_image_dir_path, num_pairs=args.num_pairs, transforms=transforms)
    dataset = mistdatasets.MISTDataset(x_img_dir=args.x_image_dir_path, y_img_dir=args.y_image_dir_path, num_pairs=args.num_pairs, transforms=transforms)
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=min(4, os.cpu_count()), pin_memory=False, drop_last=False)
    return dataloader

def test(args):
    # --- AUTOMATED FOLDER CREATION ---
    output_dir = args.output_dir
    dir_normal = os.path.join(output_dir, "NORMAL")
    dir_ema = os.path.join(output_dir, "EMA")
    dir_ihc = os.path.join(output_dir, "IHC_GT")
    dir_he = os.path.join(output_dir, "HE_GT")
    
    os.makedirs(dir_normal, exist_ok=True)
    os.makedirs(dir_ema, exist_ok=True)
    os.makedirs(dir_ihc, exist_ok=True)
    os.makedirs(dir_he, exist_ok=True)
    
    device = args.device
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.requires_grad_(False)
    
    # Load UNI model for conditioning
    timm_kwargs = {
        'img_size': 224, 'patch_size': 14, 'depth': 24, 'num_heads': 24,
        'init_values': 1e-5, 'embed_dim': 1536, 'mlp_ratio': 2.66667*2,
        'num_classes': 0, 'no_embed_class': True, 'mlp_layer': SwiGLUPacked,
        'act_layer': torch.nn.SiLU, 'reg_tokens': 8, 'dynamic_img_size': True
    }
    uni_model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs).to(device)
    uni_model.eval()
    uni_model.requires_grad_(False)
    
    uni_config = resolve_data_config(uni_model.pretrained_cfg, model=uni_model)
    uni_transform_tensor = torch.nn.Sequential(
        TF.Resize(size=uni_config['input_size'][1:], antialias=True), 
        TF.Normalize(mean=uni_config['mean'], std=uni_config['std'])
    ).to(device)
    
    @torch.no_grad()
    def compute_uni_embeddings_batched(he_images):
        he_images_0_1 = (he_images + 1) / 2
        uni_input = uni_transform_tensor(he_images_0_1)
        return uni_model(uni_input)
    
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02,
        beta_schedule="scaled_linear", prediction_type="epsilon", clip_sample=False, steps_offset=1
    )
    
    model = HistoDiT(
        img_size=args.image_size // 8, patch_size=2, in_chans=4, embed_dim=768, 
        num_heads=12, depth=12, semantic_embed_dim=1536
    ).to(device)
    ema_model = copy.deepcopy(model).to(device)
   
    logging.info('Loading model weights...')
    model.load_state_dict(torch.load(args.model_dir, map_location=device, weights_only=True))
    ema_model.load_state_dict(torch.load(args.EMA_model_dir, map_location=device, weights_only=True))
    # Calculate and print parameters
    param_summary = count_parameters(model)
    print(f"\n[HistDiT] {param_summary}")
    
    dataloader = get_test_data(args)
    
    pbar = tqdm(dataloader)
    
    for i, (gt_IHC_images, cond_test_HE_images, filenames) in enumerate(pbar):
        cond_test_HE_images = cond_test_HE_images.to(device)
        with torch.no_grad():
            conditional_latents = vae.encode(cond_test_HE_images).latent_dist.sample() * args.latent_scale
            c_semantic = compute_uni_embeddings_batched(cond_test_HE_images)
                
            generated_latents = sample_images(model, conditional_latents, c_semantic, device, noise_scheduler, args.guidance, args.strength)
            generated_images = vae.decode(generated_latents/args.latent_scale).sample
            sampled_images = ((generated_images.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)
                
            ema_generated_latents = sample_images(ema_model, conditional_latents, c_semantic, device, noise_scheduler, args.guidance, args.strength)
            ema_generated_images = vae.decode(ema_generated_latents/args.latent_scale).sample
            ema_sampled_images = ((ema_generated_images.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)
        
        gt_IHC_images = ((gt_IHC_images.to(device).clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)
        cond_test_HE_images = ((cond_test_HE_images.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)
        
        for j in range(sampled_images.size(0)):
            base = str(filenames[j])
            # Save properly sorted into their respective folders
            save_single_image(sampled_images[j], os.path.join(dir_normal, base))
            save_single_image(ema_sampled_images[j], os.path.join(dir_ema, base))
            save_single_image(gt_IHC_images[j], os.path.join(dir_ihc, base))
            save_single_image(cond_test_HE_images[j], os.path.join(dir_he, base))

def launch():
    class Args:
        # Standardized to point to the sample data by default for reviewers
        x_image_dir_path = './sample_data/MIST_Dataset/HE'
        y_image_dir_path = './sample_data/MIST_Dataset/IHC'
        output_dir = './test_results'
        num_pairs  = 8 # Adjust to match your sample size (For BCI it 977 and for MIST it is 1000)
        batch_size = 8
        image_size = 512
        latent_scale = 0.18285
        guidance = 3
        strength = 1
        
        # Standardized Model Names
        model_dir  = './weights/model_mist.pt'
        EMA_model_dir = './weights/model_ema_mist.pt'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = Args()
    test(args)

if __name__ == "__main__":
    launch()
