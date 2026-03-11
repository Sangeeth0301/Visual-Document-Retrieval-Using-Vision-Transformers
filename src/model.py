import torch
import torch.nn as nn
import math
from torchvision.models import vit_b_16, ViT_B_16_Weights

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Projection of patches to embeddings using a convolutional layer
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: (Batch, Channels, Height, Width) -> (B, C, H, W)
        x = self.proj(x) # (B, embed_dim, H//patch, W//patch)
        x = x.flatten(2) # (B, embed_dim, num_patches)
        x = x.transpose(1, 2) # (B, num_patches, embed_dim)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        # +1 for the CLS token
        # Learnable 1D positional embeddings properly initialized with truncated normal distribution
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x):
        return x + self.pos_embed

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_drop=0.0, proj_drop=0.0, use_scaling=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_scaling = use_scaling
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        
        # Compute Q, K, V simultaneously
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        # Compute Q, K, V -> shape (B, N, 3 * C)
        qkv = self.qkv(x)
        # Reshape to (B, N, 3, num_heads, head_dim) and permute to (3, B, num_heads, N, head_dim)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # Each is (B, num_heads, N, head_dim)
        
        # Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
        # Q @ K^T
        scaling = (1.0 / math.sqrt(self.head_dim)) if self.use_scaling else 1.0
        attn_scores = (q @ k.transpose(-2, -1)) * scaling
        attn_weights = attn_scores.softmax(dim=-1)
        attn_weights = self.attn_drop(attn_weights)
        
        # Multiply by V
        x = (attn_weights @ v) # (B, num_heads, N, head_dim)
        x = x.transpose(1, 2).reshape(B, N, C) # Concat heads
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_weights

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0, use_scaling=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, attn_drop=attn_drop, proj_drop=drop, use_scaling=use_scaling)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(drop)
        )
        
    def forward(self, x):
        normed_x = self.norm1(x)
        attn_out, attn_weights = self.attn(normed_x)
        x = x + attn_out # Residual connection
        
        x = x + self.mlp(self.norm2(x)) # Residual connection
        return x, attn_weights

class ViTEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, depth=4, num_heads=8, drop_rate=0.0, attn_drop_rate=0.0, use_pos_embed=True, use_scaling=True):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.use_pos_embed = use_pos_embed
        
        # [CLS] token as a learnable parameter initialized correctly
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Positional encoding
        self.pos_embed = PositionalEncoding(self.patch_embed.num_patches, embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, drop=drop_rate, attn_drop=attn_drop_rate, use_scaling=use_scaling) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x, return_attn=False):
        B = x.shape[0]
        # 1. Patch Embedding
        x = self.patch_embed(x)
        
        # 2. Add [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (B, num_patches + 1, embed_dim)
        
        # 3. Add Positional Encoding and Dropout
        if self.use_pos_embed:
            x = self.pos_embed(x)
        x = self.pos_drop(x)
        
        attn_weights = []
        # 4. Pass through Transformer Blocks
        for block in self.blocks:
            x, attn = block(x)
            attn_weights.append(attn)
                
        # 5. Final LayerNorm
        x = self.norm(x)
        
        # 6. Extract the [CLS] token representation for the entire image
        cls_embedding = x[:, 0]
        
        if return_attn:
            return cls_embedding, attn_weights
        return cls_embedding

class BetterViTEncoder(nn.Module):
    def __init__(self, embed_dim=768, pretrained=True):
        super().__init__()
        if pretrained:
            weights = ViT_B_16_Weights.IMAGENET1K_V1
            self.model = vit_b_16(weights=weights)
        else:
            self.model = vit_b_16(weights=None)
            
        # We only need the backbone, not the final classification head
        self.model.heads = nn.Identity()
        
    def forward(self, x, return_attn=False):
        # x: (B, 3, 224, 224)
        if return_attn:
            # Note: torchvision's ViT doesn't easily return attention maps without hooks
            # We'll skip attn for the 'Better' model for simplicity or use a hook later
            return self.model(x), None
        return self.model(x)
