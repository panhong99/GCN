import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.patch_embed(x)  # (batch_size, embed_dim, num_patches)
        x = x.flatten(2).transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim, num_classes, num_layers, num_heads, hidden_dim, dropout):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.num_patches = (image_size // patch_size) ** 2

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(embed_dim, num_heads, hidden_dim, dropout), num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embedding(x)  # (B, num_patches, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, embed_dim)
        x += self.pos_embedding  # (B, num_patches + 1, embed_dim)
        x = self.transformer_encoder(x)  # (B, num_patches + 1, embed_dim)
        x = x[:, 0]  # (B, embed_dim)
        x = self.fc(x)  # (B, num_classes)
        return x

def vit_base_patch16_224(num_classes):
    return VisionTransformer(image_size=224, patch_size=16, in_channels=3, embed_dim=768, num_classes=num_classes, num_layers=12, num_heads=12, hidden_dim=3072, dropout=0.1)
