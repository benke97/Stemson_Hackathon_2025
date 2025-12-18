import torch
import torch.nn as nn
from torchvision import models
import numpy as np

class MetadataEncoder(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=64, output_dim=128, dropout_prob=0.2):
        super(MetadataEncoder, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class ImageEncoder(nn.Module):
    def __init__(self, output_dim=128, pretrained=True, freeze_backbone=False):
        """
        Args:
            output_dim (int): Dimension of the output embedding.
            pretrained (bool): Whether to load ImageNet weights.
            freeze_backbone (bool): If True, freezes the convolutional layers.
        """
        super(ImageEncoder, self).__init__()
        
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.model = models.resnet18(weights=weights)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, output_dim)

    def forward(self, x):
        return self.model(x)

class SigLIPModel(nn.Module):
    """
    Combines Image and Metadata encoders and holds
    Temperature (t) and Bias (b) for Sigmoid Loss.
    """
    def __init__(self, meta_input_dim=14, embed_dim=128, init_temp=20.0, init_bias=-6.24):
        super().__init__()
        self.visual = ImageEncoder(output_dim=embed_dim)
        self.meta = MetadataEncoder(input_dim=meta_input_dim, output_dim=embed_dim)
        
        self.t_prime = nn.Parameter(torch.tensor(np.log(init_temp))) 
        self.b = nn.Parameter(torch.tensor(init_bias))

    def forward(self, img, meta_vec):
        
        img_feat = self.visual(img)
        meta_feat = self.meta(meta_vec)

        #turn embeddings into unit vectors so they're all the same length and cosine sim compares angles only
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        meta_feat = meta_feat / meta_feat.norm(dim=-1, keepdim=True)

        return img_feat, meta_feat, self.t_prime.exp(), self.b

if __name__ == "__main__":
    # Test block
    model = SigLIPModel(meta_input_dim=14)
    # Dummy image must now be 3 channels
    dummy_img = torch.randn(2, 3, 224, 224) 
    dummy_meta = torch.randn(2, 14)
    
    i_emb, m_emb, t, b = model(dummy_img, dummy_meta)
    print(f"Image Emb: {i_emb.shape}, Meta Emb: {m_emb.shape}")
    print(f"Temp: {t.item()}, Bias: {b.item()}")