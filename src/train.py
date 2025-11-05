import argparse, json, os, math
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.seed import set_seed
from utils.metrics import BinaryMetrics
from utils.logging_utils import save_json
from data.dataset import HNJsonDataset, collate_cases
from models.encoders import CT2DEncoder
from models.adapters import DoseAdapter3D, ContourAdapter3D, NumericAdapter
from models.fusion import AttnPool1D
from models.attention import CrossAttentionBlock
from models.heads import BinaryHead
from utils.losses import make_bce_loss

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=str)
    ap.add_argument("--stage", required=True, type=int, choices=[1,2,3])
    ap.add_argument("--cfg", type=str, default=None)
    ap.add_argument("--out", type=str, default="runs/stageX")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--slices", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--pos_weight", type=float, default=None)
    ap.add_argument("--use_dose_adapter", action="store_true")
    ap.add_argument("--use_contour_adapter", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--ct_encoder", type=str, default="vit_small_patch16_224")
    return ap.parse_args()

class StageModel(nn.Module):
    def __init__(self, stage:int, ct_encoder_name:str, d_numeric:int,
                 use_dose_adapter:bool, use_contour_adapter:bool, d_model:int=384,
                 n_heads:int=6, n_ca_layers:int=2):
        super().__init__()
        # CT slice encoder (frozen)
        self.ct = CT2DEncoder(model_name=ct_encoder_name, out_dim=d_model, freeze=True)
        self.pool = AttnPool1D(d_model)
        # Adapters for Stage 2/3
        self.num_adapter = NumericAdapter(in_dim=d_numeric, out_dim=d_model) if d_numeric>0 else None
        self.dose_adapter = DoseAdapter3D(out_dim=d_model) if use_dose_adapter else None
        self.sdf_adapter = ContourAdapter3D(out_dim=d_model) if use_contour_adapter else None
        # Cross-attention blocks for Stage 3
        self.stage = stage
        if stage==3:
            self.ca_blocks = nn.ModuleList([CrossAttentionBlock(d_model, n_heads) for _ in range(n_ca_layers)])
        # Final head
        # Features concatenated in Stage 2; in Stage 3 we output pooled token
        concat_dim = d_model if stage==3 else d_model + (d_model if d_numeric>0 else 0) + \
                    (d_model if use_dose_adapter else 0) + (d_model if use_contour_adapter else 0)
        self.head = BinaryHead(in_dim=concat_dim, hidden=512)

    def forward(self, batch):
        """
        batch:
          - ct_slices: [B, K, 1, H, W]
          - num_feats: [B, F] (optional)
          - dose_patch: [B, 1, Dz, Hy, Wx] (optional)
          - sdf_patch: [B, 1, Dz, Hy, Wx] (optional)
        """
        B = batch["ct_slices"].shape[0]
        ct_tokens = self.ct.encode_slices(batch["ct_slices"])  # [B, K, D]
        if self.stage==3:
            token_list = [ct_tokens]  # [B, K, D]
            if self.num_adapter is not None and "num_feats" in batch:
                # treat each shell as a token already prepared in data pipeline:
                token_list.append(batch["shell_tokens"])  # [B, S, D] (pre-projected later)
            if self.dose_adapter is not None and "dose_patch" in batch:
                dose_vec = self.dose_adapter(batch["dose_patch"]).unsqueeze(1)  # [B,1,D]
                token_list.append(dose_vec)
            if self.sdf_adapter is not None and "sdf_patch" in batch:
                sdf_vec = self.sdf_adapter(batch["sdf_patch"]).unsqueeze(1)  # [B,1,D]
                token_list.append(sdf_vec)
            tokens = torch.cat(token_list, dim=1)  # [B, T, D]
            # prepare KQVs: CT tokens query others (and themselves)
            # We split tokens so CT is first K entries
            K = ct_tokens.size(1)
            for blk in self.ca_blocks:
                tokens = blk(q=tokens[:, :K, :], kv=tokens)  # returns updated concat (same shape)
            pooled = self.pool(tokens[:, :K, :])  # pool only CT queries
            logits = self.head(pooled)
            return logits
        else:
            pooled = self.pool(ct_tokens)  # [B, D]
            feats = [pooled]
            if self.num_adapter is not None and "num_feats" in batch:
                feats.append(self.num_adapter(batch["num_feats"]))
            if self.dose_adapter is not None and "dose_patch" in batch:
                feats.append(self.dose_adapter(batch["dose_patch"]))
            if self.sdf_adapter is not None and "sdf_patch" in batch:
                feats.append(self.sdf_adapter(batch["sdf_patch"]))
            fused = torch.cat(feats, dim=1)
            logits = self.head(fused)
            return logits

def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    metrics = BinaryMetrics()
    pbar = tqdm(loader, desc="train")
    for batch in pbar:
        for k in batch: batch[k] = batch[k].to(device) if torch.is_tensor(batch[k]) else batch[k]
        y = batch["label"].float().unsqueeze(1)
        logits = model(batch)
        loss = loss_fn(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            prob = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
            metrics.update(prob, y.squeeze(1).detach().cpu().numpy())
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "auc": f"{metrics.auroc():.3f}"})
    return {"loss": float(loss.item()), **metrics.compute()}

@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    metrics = BinaryMetrics()
    total_loss = 0.0; n=0
    for batch in loader:
        for k in batch: batch[k] = batch[k].to(device) if torch.is_tensor(batch[k]) else batch[k]
        y = batch["label"].float().unsqueeze(1)
        logits = model(batch)
        loss = loss_fn(logits, y)
        total_loss += loss.item()*y.size(0); n += y.size(0)
        prob = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
        metrics.update(prob, y.squeeze(1).detach().cpu().numpy())
    return {"loss": total_loss/max(n,1), **metrics.compute()}

def main():
    args = get_args()
    cfg = {}
    if args.cfg and os.path.isfile(args.cfg):
        cfg = yaml.safe_load(open(args.cfg))
    # Merge CLI over YAML defaults
    for k,v in vars(args).items():
        if v is not None: cfg[k] = v
    out = Path(cfg["out"]); out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.get("seed",1337))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets
    train_set = HNJsonDataset(cfg["data"], split="train",
                              num_slices=cfg.get("slices",16),
                              img_size=cfg.get("img_size",224),
                              stage=cfg["stage"],
                              fit_stats=True)
    val_set = HNJsonDataset(cfg["data"], split="val",
                            num_slices=cfg.get("slices",16),
                            img_size=cfg.get("img_size",224),
                            stage=cfg["stage"],
                            stats=train_set.stats)  # reuse standardization
    d_numeric = train_set.numeric_dim

    train_loader = DataLoader(train_set, batch_size=cfg.get("batch_size",2),
                              shuffle=True, num_workers=cfg.get("workers",4),
                              collate_fn=collate_cases)
    val_loader = DataLoader(val_set, batch_size=max(1,cfg.get("batch_size",2)),
                            shuffle=False, num_workers=cfg.get("workers",4),
                            collate_fn=collate_cases)

    model = StageModel(stage=cfg["stage"], ct_encoder_name=cfg.get("ct_encoder","vit_small_patch16_224"),
                       d_numeric=d_numeric,
                       use_dose_adapter=cfg.get("use_dose_adapter", False),
                       use_contour_adapter=cfg.get("use_contour_adapter", False),
                       d_model=384 if "vit_small" in cfg.get("ct_encoder","") else 512).to(device)

    # Loss (with optional positive class weight)
    pos_w = cfg.get("pos_weight", None)
    if pos_w is None:
        # auto from training split
        pos_frac = train_set.pos_fraction
        pos_w = (1-pos_frac)/max(pos_frac,1e-6)
    loss_fn = make_bce_loss(pos_weight=pos_w).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr",3e-4),
                            weight_decay=cfg.get("weight_decay",1e-4))

    best_auc = -1.0; history = {}
    for epoch in range(cfg.get("epochs",20)):
        tr = train_one_epoch(model, train_loader, opt, loss_fn, device)
        va = validate(model, val_loader, loss_fn, device)
        history[epoch] = {"train": tr, "val": va}
        print(f"Epoch {epoch}: val AUC {va['auroc']:.3f} | val F1 {va['f1']:.3f}")
        if va["auroc"] > best_auc:
            best_auc = va["auroc"]
            torch.save(model.state_dict(), out/"best.ckpt")
        torch.save(model.state_dict(), out/"last.ckpt")
        save_json(history, out/"metrics.json")

if __name__ == "__main__":
    main()
