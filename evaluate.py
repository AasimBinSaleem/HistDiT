import os
import torch
import lpips
import numpy as np
from cleanfid import fid
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from PIL import Image
# Ensure utils is in your path or models folder
from models.utils import SSIM_decomposed

def load_image(image_path):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
    return np.array(image, dtype=np.uint8)

def evaluate_images(gt_dir, generated_dir, result_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_loss = lpips.LPIPS(net='alex').to(device)
    
    gt_images = sorted(os.listdir(gt_dir))
    generated_images = sorted(os.listdir(generated_dir))
    
    mse_list, psnr_list, ms_ssim_list, scm_list, lpips_list = [], [], [], [], []
    
    with open(result_file, "w") as f:
        f.write("Image_Name, MSE, PSNR, Ms-SSIM, SCM, LPIPS \n")
        
        for gt_img_name, gen_img_name in zip(gt_images, generated_images):
            if not gt_img_name.endswith('.png'): continue
            
            gt_path = os.path.join(gt_dir, gt_img_name)
            gen_path = os.path.join(generated_dir, gen_img_name)

            gt_img = load_image(gt_path)
            gen_img = load_image(gen_path)
            
            mse = mean_squared_error(gt_img, gen_img)
            psnr = peak_signal_noise_ratio(gt_img, gen_img, data_range=255)
            ssim = structural_similarity(gt_img, gen_img, channel_axis=-1)
            
            gt_tensor = torch.from_numpy(gt_img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            gen_tensor = torch.from_numpy(gen_img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
          
            ms_ssim, _, scm = SSIM_decomposed(gt_tensor, gen_tensor, val_range=1.0, size_average=True)
            
            gt_tensor_lpips = (gt_tensor * 2.0) - 1.0 
            gen_tensor_lpips = (gen_tensor * 2.0) - 1.0
            
            with torch.no_grad():
                lpips_val = lpips_loss(gt_tensor_lpips, gen_tensor_lpips).item()
            
            mse_list.append(mse)
            psnr_list.append(psnr)
            ms_ssim_list.append(ms_ssim.item())
            scm_list.append(scm.item())
            lpips_list.append(lpips_val)
            
            f.write(f"{gt_img_name}, {mse:.2f}, {psnr:.2f}, {ms_ssim.item():.4f}, {scm.item():.4f}, {lpips_val:.4f}\n")
    
    # Compute FID
    fid_score = fid.compute_fid(gt_dir, generated_dir, mode="clean", device=device, num_workers=0)
    
    print(f"\nEvaluation Complete! Results saved in: {result_file}")
    print(f"Average MSE: {np.mean(mse_list):.2f} (Lower is better)")
    print(f"Average PSNR: {np.mean(psnr_list):.2f}")
    print(f"Average Ms-SSIM: {np.mean(ms_ssim_list):.4f}")
    print(f"SCM: {np.mean(scm_list):.4f}")
    print(f"Average LPIPS: {np.mean(lpips_list):.4f} (Lower is better)")
    print(f"FID Score: {fid_score:.2f} (Lower is better)")

if __name__ == "__main__":
    gt_dir = './test_results/IHC_GT'
    generated_dir = './test_results/NORMAL' # Or change to EMA if evaluating EMA results
    result_file = './test_results/EVALUATION_METRICS.txt'
    
    evaluate_images(gt_dir, generated_dir, result_file)