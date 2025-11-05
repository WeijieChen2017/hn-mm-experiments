import torch
import torch.nn as nn
import timm

class CT2DEncoder(nn.Module):
    """
    2D encoder over slices using timm backbones (ImageNet pretrained).
    Produces per-slice embeddings; frozen by default.
    """
    def __init__(self, model_name:str="vit_small_patch16_224", out_dim:int=384, freeze:bool=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool="avg")
        # Ensure single-channel input by repeating channel -> 3
        self.out_dim = self.model.num_features if hasattr(self.model, "num_features") else out_dim
        if freeze:
            for p in self.model.parameters(): p.requires_grad=False

    def encode_slices(self, x:torch.Tensor)->torch.Tensor:
        """
        x: [B,K,1,H,W]
        returns tokens [B,K,D]
        """
        B,K,_,H,W = x.shape
        x = x.repeat(1,1,3,1,1).reshape(B*K, 3, H, W)
        z = self.model(x)  # [B*K, D]
        return z.view(B, K, -1)

# Optional: MedImageInsight stub (implement if you have access to the API and want to use it)
class MIIImageEncoder(nn.Module):
    def __init__(self, out_dim:int):
        super().__init__()
        raise NotImplementedError("Implement MedImageInsight client here and emit per-slice embeddings.")
