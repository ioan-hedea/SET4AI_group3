import json
import os
import time
import csv
import numpy as np
import torch
from torchvision.utils import save_image
from torchvision.models import vgg16
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent


# -----------------------------
# Utility: parse ImageNet prediction
# -----------------------------
def parse_prediction(output, categories):
    probs = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_catid = torch.topk(probs, 1)
    return categories[top_catid], top_prob.item()


# -----------------------------
# Metrics: pixel changes + Linf
# x and x_adv are tensors shaped (1,3,H,W) in [0,1]
# -----------------------------
def perturbation_metrics(x: torch.Tensor, x_adv: torch.Tensor, thresh: float = 1.0 / 255.0):
    """
    Returns:
      num_pixels_changed: count of (H,W) locations where any channel changed more than thresh
      linf: max absolute perturbation (L∞) over all pixels/channels
    """
    # absolute diff per channel
    diff = (x_adv - x).abs()  # (1,3,H,W)
    linf = float(diff.max().item())

    # per-pixel change if ANY channel changed more than thresh
    per_pixel = diff.max(dim=1).values  # (1,H,W)
    num_pixels_changed = int((per_pixel > thresh).sum().item())
    return num_pixels_changed, linf


# ================================================================
# 1. Load JSON file with images + expected human label
# ================================================================
JSON_FILE = "data/image_labels.json"
IMAGE_DIR = "images/"

with open(JSON_FILE, "r") as f:
    items = json.load(f)

# ================================================================
# 2. Load ImageNet labels
# ================================================================
with open("data/imagenet_classes.txt", "r") as f:
    imagenet_labels = [s.strip() for s in f.readlines()]

label_to_index = {label: i for i, label in enumerate(imagenet_labels)}

# ================================================================
# 3. Model
# ================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
net = vgg16(weights="DEFAULT").to(device)
net.eval()

# ================================================================
# 4. Image preprocessing transform
# ================================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # -> [0,1]
])

# ================================================================
# 5. Attack hyperparameters
# ================================================================
EPS = 0.30
PGD_STEPS = 40
PGD_STEP_SIZE = 0.01

# ================================================================
# 6. Output directories
# ================================================================
OUTDIR = "attack_results"
os.makedirs(OUTDIR, exist_ok=True)

METRICS_CSV = os.path.join(OUTDIR, "metrics.csv")

# ================================================================
# 7. Run attacks + collect metrics
# ================================================================
rows = []

for entry in tqdm(items, desc="Running attacks"):
    image_file = entry["image"]
    human_label = entry["label"]  # e.g. "goldfish"

    # Load + preprocess image
    img_path = os.path.join(IMAGE_DIR, image_file)
    img_pil = Image.open(img_path).convert("RGB")
    x = transform(img_pil).unsqueeze(0).to(device)  # (1,3,224,224)

    # Ground truth index
    true_idx = label_to_index.get(human_label, None)
    if true_idx is None:
        print(f" Warning: '{human_label}' not found in ImageNet labels.")

    # Predict clean image
    with torch.no_grad():
        out_clean = net(x)
    pred_clean, prob_clean = parse_prediction(out_clean, imagenet_labels)
    pred_clean_idx = imagenet_labels.index(pred_clean)

    save_image(x, os.path.join(OUTDIR, f"{image_file}_clean.png"))

    clean_correct = (true_idx is not None) and (pred_clean_idx == true_idx)

    print(f"\nImage: {image_file}")
    print(f"Human label: {human_label}")
    print(f"Model prediction (clean): {pred_clean} ({prob_clean:.3f})")

    # -----------------------------
    # FGM Attack
    # -----------------------------
    t0 = time.perf_counter()
    x_fgm = fast_gradient_method(net, x, EPS, np.inf)
    fgm_time = time.perf_counter() - t0

    with torch.no_grad():
        out_fgm = net(x_fgm)
    pred_fgm, prob_fgm = parse_prediction(out_fgm, imagenet_labels)
    pred_fgm_idx = imagenet_labels.index(pred_fgm)

    save_image(x_fgm, os.path.join(OUTDIR, f"{image_file}_fgm.png"))

    fgm_correct = (true_idx is not None) and (pred_fgm_idx == true_idx)
    fgm_success = (true_idx is not None) and (pred_fgm_idx != true_idx)
    fgm_pixels_changed, fgm_linf = perturbation_metrics(x, x_fgm)

    print(f"FGM prediction: {pred_fgm} ({prob_fgm:.3f})")

    # -----------------------------
    # PGD Attack
    # -----------------------------
    t0 = time.perf_counter()
    x_pgd = projected_gradient_descent(net, x, EPS, PGD_STEP_SIZE, PGD_STEPS, np.inf)
    pgd_time = time.perf_counter() - t0

    with torch.no_grad():
        out_pgd = net(x_pgd)
    pred_pgd, prob_pgd = parse_prediction(out_pgd, imagenet_labels)
    pred_pgd_idx = imagenet_labels.index(pred_pgd)

    save_image(x_pgd, os.path.join(OUTDIR, f"{image_file}_pgd.png"))

    pgd_correct = (true_idx is not None) and (pred_pgd_idx == true_idx)
    pgd_success = (true_idx is not None) and (pred_pgd_idx != true_idx)
    pgd_pixels_changed, pgd_linf = perturbation_metrics(x, x_pgd)

    print(f"PGD prediction: {pred_pgd} ({prob_pgd:.3f})")

    # Summary prints (as before)
    if true_idx is not None:
        print("\nCorrect label index:", true_idx)
        print("Clean correct?", clean_correct)
        print("FGM correct?", fgm_correct)
        print("PGD correct?", pgd_correct)

    print("------------------------------------------------------")

    # Record row
    rows.append({
        "image": image_file,
        "label": human_label,
        "clean_top1": pred_clean,
        "clean_prob": prob_clean,
        "clean_correct": int(clean_correct),

        "fgm_top1": pred_fgm,
        "fgm_prob": prob_fgm,
        "fgm_success": int(fgm_success),
        "fgm_pixels_changed": fgm_pixels_changed,
        "fgm_linf": fgm_linf,
        "fgm_time_s": fgm_time,

        "pgd_top1": pred_pgd,
        "pgd_prob": prob_pgd,
        "pgd_success": int(pgd_success),
        "pgd_pixels_changed": pgd_pixels_changed,
        "pgd_linf": pgd_linf,
        "pgd_time_s": pgd_time,
    })

# ================================================================
# 8. Save metrics CSV + print aggregates
# ================================================================
fieldnames = list(rows[0].keys()) if rows else []
with open(METRICS_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

def avg(xs): return float(sum(xs) / max(1, len(xs)))

fgm_success_rate = avg([r["fgm_success"] for r in rows])
pgd_success_rate = avg([r["pgd_success"] for r in rows])

fgm_pixels_avg = avg([r["fgm_pixels_changed"] for r in rows])
pgd_pixels_avg = avg([r["pgd_pixels_changed"] for r in rows])

fgm_linf_max = max(r["fgm_linf"] for r in rows) if rows else 0.0
pgd_linf_max = max(r["pgd_linf"] for r in rows) if rows else 0.0

fgm_time_avg = avg([r["fgm_time_s"] for r in rows])
pgd_time_avg = avg([r["pgd_time_s"] for r in rows])

print("\n=== Aggregate metrics (over dataset) ===")
print(f"FGM success rate: {fgm_success_rate:.2f}")
print(f"PGD success rate: {pgd_success_rate:.2f}")
print(f"FGM avg pixels changed: {fgm_pixels_avg:.1f}")
print(f"PGD avg pixels changed: {pgd_pixels_avg:.1f}")
print(f"FGM max Linf: {fgm_linf_max:.4f} (expected <= {EPS:.2f})")
print(f"PGD max Linf: {pgd_linf_max:.4f} (expected <= {EPS:.2f})")
print(f"FGM avg time/image: {fgm_time_avg:.4f}s")
print(f"PGD avg time/image: {pgd_time_avg:.4f}s")
print(f"\nSaved metrics to: {METRICS_CSV}")