import math
import torch
import torch.nn as nn

def get_timestep_embedding(timesteps, embedding_dim):
    """
    Builds a sinusoidal timestep embedding.
    """
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Modulate(nn.Module):
    """
    A simple modulation layer that predicts scale and shift from a
    combined conditioning embedding.
    """
    def __init__(self, latent_dim, num_features):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(latent_dim, 2 * num_features), # Predicts scale and shift
        )

    def forward(self, x, c_combined):
        mod_params = self.adaLN_modulation(c_combined).unsqueeze(1) # [B, 1, 2*F]
        scale, shift = mod_params.chunk(2, dim=2) # [B, 1, F], [B, 1, F]
        return (scale * x) + shift


class Mlp(nn.Module):
    """A standard feed-forward network (MLP) block."""
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
    
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding.
    """
    def __init__(self, img_size=64, patch_size=2, in_chans=4, embed_dim=768):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x) # Shape: [B, E, H/P, W/P]
        x = x.flatten(2) # Shape: [B, E, N]
        x = x.transpose(1, 2) # Shape: [B, N, E]
        return x

class HistoDiTBlock(nn.Module):
    """
    The core HistoDiT block.
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.mod1 = Modulate(embed_dim, embed_dim)
        
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.mod2 = Modulate(embed_dim, embed_dim)

        self.norm3 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.mod3 = Modulate(embed_dim, embed_dim)
        
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        hidden_features = int(embed_dim * mlp_ratio)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=hidden_features)

    def forward(self, x, c_spatial, c_combined):
        """
        x: Input IHC tokens [B, N, E]
        c_spatial: H&E spatial condition tokens [B, N_cond, E]
        c_combined: Combined (t + semantic) embedding [B, E]
        """
        
        # 1. Self-Attention with adaLN
        x_norm = self.norm1(x)
        x_modulated = self.mod1(x_norm, c_combined)
        attn_out, _ = self.self_attn(x_modulated, x_modulated, x_modulated)
        x = x + attn_out
        
        # 2. Cross-Attention with adaLN
        x_norm = self.norm2(x)
        x_modulated = self.mod2(x_norm, c_combined)
        attn_out, _ = self.cross_attn(
            query=x_modulated, 
            key=c_spatial, 
            value=c_spatial
        )
        x = x + attn_out
        
        # 3. MLP with adaLN
        x_norm = self.norm3(x)
        x_modulated = self.mod3(x_norm, c_combined)
        mlp_out = self.mlp(x_modulated)
        x = x + mlp_out
        
        return x


# --- Main Model (UPDATED FOR CFG) ---

class HistoDiT(nn.Module):
    """
    The full HistoDiT model, now with CFG support.
    """
    def __init__(self,
                 img_size=64,
                 patch_size=2,
                 in_chans=4,
                 embed_dim=768,
                 num_heads=12,
                 depth=12,
                 semantic_embed_dim=1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.semantic_embed_dim = semantic_embed_dim

        # --- Input Embedders ---
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        self.cond_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_cond_patches = self.cond_embed.num_patches
        self.cond_pos_embed = nn.Parameter(torch.zeros(1, num_cond_patches, embed_dim))

        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Renamed to clarify its role: projects semantic dim to embed_dim
        self.label_embed_projector = nn.Sequential(
            nn.Linear(semantic_embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # --- CFG NULL EMBEDDINGS (NEW) ---
        # Trainable embeddings for the unconditional case
        self.null_semantic_embed = nn.Parameter(torch.zeros(1, self.semantic_embed_dim))
        self.null_spatial_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        # --- Core Backbone ---
        self.blocks = nn.ModuleList([
            HistoDiTBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        
        # --- Output Layers ---
        self.norm_out = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.mod_out = Modulate(embed_dim, embed_dim)
        self.linear_out = nn.Linear(embed_dim, patch_size * patch_size * in_chans)

    def forward(self, z_t, t, c_spatial, c_semantic):
        """
        z_t: Noised IHC latent [B, C, H, W]
        t: Timesteps [B]
        c_spatial: H&E latent [B, C, H, W] or None
        c_semantic: H&E semantic embedding [B, D_sem] or None
        """
        
        # --- Embed all inputs ---
        
        # 1. Embed z_t (main data)
        x = self.patch_embed(z_t) + self.pos_embed # [B, N, E]
        
        # 2. Embed c_spatial (H&E structure) - (UPDATED FOR CFG)
        if c_spatial is None:
            # Use null spatial embedding
            c_s = self.null_spatial_embed.expand(z_t.shape[0], -1, -1) # [B, N, E]
        else:
            # Use H&E latent
            c_s = self.cond_embed(c_spatial) + self.cond_pos_embed # [B, N_cond, E]
        
        # 3. Embed t and c_semantic (for adaLN) - (UPDATED FOR CFG)
        t_emb = self.time_embed(get_timestep_embedding(t, self.embed_dim)) # [B, E]
        
        if c_semantic is None:
            # Use null semantic embedding
            c_sem_null = self.null_semantic_embed.expand(z_t.shape[0], -1) # [B, D_sem]
            c_sem_emb = self.label_embed_projector(c_sem_null) # [B, E]
        else:
            # Use UNI embedding
            c_sem_emb = self.label_embed_projector(c_semantic) # [B, E]
        
        # Combine t and semantic embedding
        c_combined = t_emb + c_sem_emb # [B, E]

        # --- Run through backbone ---
        for block in self.blocks:
            x = block(x, c_s, c_combined) # [B, N, E]
            
        # --- Final output processing ---
        x_norm = self.norm_out(x)
        x_modulated = self.mod_out(x_norm, c_combined)
        x_out = self.linear_out(x_modulated) # [B, N, P*P*C]
        
        # --- Un-patchify ---
        B, N, _ = x_out.shape
        H_patch = W_patch = int(N**0.5)
        P = self.patch_size
        C = self.in_chans
        
        x_out = x_out.reshape(B, H_patch, W_patch, P, P, C)
        x_out = x_out.permute(0, 5, 1, 3, 2, 4)
        predicted_noise = x_out.reshape(B, C, H_patch * P, W_patch * P)
        
        return predicted_noise

# ============================== Exponential Moving Average ============================
    
class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):

            old_weight, up_weight = ma_params.data, current_params.data

            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        # calculate the value of average gradients
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)

        self.step += 1

    def reset_parameters(self, ema_model, model):
        # reset the weights of the EMA model
        ema_model.load_state_dict(model.state_dict())


if __name__ == '__main__':
    pass