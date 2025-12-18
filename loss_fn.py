import torch
import torch.nn as nn
import torch.nn.functional as F

class SigmoidCLIPLoss(nn.Module):
    """
    Sigmoid Loss with Multi-Positives.
    If multiple images in the batch share the same metadata, they are ALL 
    treated as positive matches
    """
    def __init__(self, epsilon=1e-6):
        """
        Args:
            epsilon (float): Threshold to consider two metadata vectors identical.
        """
        super().__init__()
        self.epsilon = epsilon

    def forward(self, img_feat, meta_feat, temperature, bias, raw_meta_batch):
        """
        Args:
            img_feat: (B, D) Normalized image embeddings
            meta_feat: (B, D) Normalized metadata embeddings
            temperature: Scalar (learnable)
            bias: Scalar (learnable)
            raw_meta_batch: (B, 14) The actual input metadata vectors (ground truth)
        """
        #Similarities
        logits = (torch.matmul(img_feat, meta_feat.t()) * temperature) + bias
        
        #check gt metadatas and which are the same
        dists = torch.cdist(raw_meta_batch, raw_meta_batch, p=2)
        
        # Create Boolean Mask: True if metadata is effectively identical, False otherwise
        is_positive = dists < self.epsilon
        
        # Convert True/False to Sigmoid Targets (+1/-1)
        labels = torch.where(is_positive, torch.tensor(1.0, device=logits.device), torch.tensor(-1.0, device=logits.device))
        
        # 3. Calculate Sigmoid Loss
        loss = F.softplus(-labels * logits).mean()
        
        return loss