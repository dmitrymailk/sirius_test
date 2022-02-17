import torch
from torch import nn
from vit_pytorch import ViT

class VisionTransformer(nn.Module):
    def __init__(self, 
                    image_size = 256,
                    patch_size = 32,
                    num_classes = 10,
                    dim = 1024,
                    depth = 6,
                    heads = 16,
                    mlp_dim = 2048,
                    dropout = 0.1,
                    emb_dropout = 0.1):
        super(VisionTransformer, self).__init__()

        self.trans = ViT(
                image_size=image_size,
                patch_size=patch_size,
                num_classes=num_classes,
                dim=dim,
                depth=depth,
                heads=heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                emb_dropout=emb_dropout)
    
    def forward(self, x):
        x = self.trans(x)
        return x 