import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from data.dataset import HNJsonDataset, collate_cases
from models.encoders import CT2DEncoder
from models.adapters import DoseAdapter3D, ContourAdapter3D, NumericAdapter
from models.fusion import AttnPool1D
from models.attention import CrossAttentionBlock
from models.heads import BinaryHead
from utils.metrics import BinaryMetrics
import pandas as pd

def build_model(stage, ct_encoder_name, d_numeric, use_dose_adapter, use_contour_adapter):
    # Keep in sync with train.py (simple builder)
    d_model = 384 if "vit_small" in ct_encoder_name else 512
    from train import StageModel
    return StageModel(stage, ct_encoder_name, d_numeric, use_dose_adapter, use_contour_adapter, d_model=d_model)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--ct_encoder", type=str, default="vit_small_patch16_224")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--slices", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--use_dose_adapter", action="store_true")
    ap.add_argument("--use_contour_adapter", action="store_true")
    args = ap.parse_args()

    test_set = HNJsonDataset(args.data, split="test", num_slices=args.slices,
                             img_size=args.img_size, stage=args.stage, fit_stats=False)
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=collate_cases)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.stage, args.ct_encoder, test_set.numeric_dim,
                        args.use_dose_adapter, args.use_contour_adapter).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    metrics = BinaryMetrics()
    rows = []
    with torch.no_grad():
        for batch in loader:
            case_ids = batch["case_id"]
            for k in batch: batch[k] = batch[k].to(device) if torch.is_tensor(batch[k]) else batch[k]
            logits = model(batch)
            prob = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
            y = batch["label"].detach().cpu().numpy()
            metrics.update(prob, y)
            for cid, p, gt in zip(case_ids, prob, y):
                rows.append({"case_id": cid, "prob": float(p), "label": int(gt)})
    print(metrics.summary_str())
    out_csv = Path(args.ckpt).parent/"test_predictions.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

if __name__ == "__main__":
    main()
