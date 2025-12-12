import json
import os
import numpy as np
import torch
from torchvision.utils import save_image
from torchvision.models import vgg16
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent

# -----------------------------
# Utility: parse Imagenet prediction
# -----------------------------
def parse_prediction(output, categories):
    probs = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_catid = torch.topk(probs, 1)
    return categories[top_catid], top_prob.item()


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
    transforms.ToTensor(),
])


# ================================================================
# 5. Attack hyperparameters
# ================================================================
EPS = 0.30          # This can be tuned
PGD_STEPS = 40
PGD_STEP_SIZE = 0.01

# ================================================================
# 6. Output directory
# ================================================================
OUTDIR = "attack_results"
os.makedirs(OUTDIR, exist_ok=True)

# ================================================================
# 7. Run attacks for every image from the JSON file
# ================================================================
for entry in tqdm(items, desc="Running attacks"):
    image_file = entry["image"]
    human_label = entry["label"]  # e.g. "goldfish"

    # -----------------------------
    # Load + preprocess image
    # -----------------------------
    img_path = os.path.join(IMAGE_DIR, image_file)
    img_pil = Image.open(img_path).convert("RGB")
    x = transform(img_pil).unsqueeze(0).to(device)

    # -----------------------------
    # Ground truth index
    # -----------------------------
    if human_label in label_to_index:
        true_idx = label_to_index[human_label]
    else:
        true_idx = None
        print(f"⚠️ Warning: '{human_label}' not found in ImageNet labels.")

    # -----------------------------
    # Predict clean image
    # -----------------------------
    out_clean = net(x)
    pred_clean, prob_clean = parse_prediction(out_clean, imagenet_labels)

    # Save clean image
    save_image(x, os.path.join(OUTDIR, f"{image_file}_clean.png"))

    print(f"\nImage: {image_file}")
    print(f"Human label: {human_label}")
    print(f"Model prediction (clean): {pred_clean} ({prob_clean:.3f})")

    # =====================================================
    # FGM Attack
    # =====================================================
    x_fgm = fast_gradient_method(net, x, EPS, np.inf)
    out_fgm = net(x_fgm)
    pred_fgm, prob_fgm = parse_prediction(out_fgm, imagenet_labels)

    save_image(x_fgm, os.path.join(OUTDIR, f"{image_file}_fgm.png"))

    print(f"FGM prediction: {pred_fgm} ({prob_fgm:.3f})")

    # =====================================================
    # PGD Attack
    # =====================================================
    x_pgd = projected_gradient_descent(
        net, x, EPS, PGD_STEP_SIZE, PGD_STEPS, np.inf
    )

    out_pgd = net(x_pgd)
    pred_pgd, prob_pgd = parse_prediction(out_pgd, imagenet_labels)

    save_image(x_pgd, os.path.join(OUTDIR, f"{image_file}_pgd.png"))

    print(f"PGD prediction: {pred_pgd} ({prob_pgd:.3f})")

    # =====================================================
    # Summary for this image
    # =====================================================
    if true_idx is not None:
        print("\nCorrect label index:", true_idx)
        print("Clean correct?", imagenet_labels.index(pred_clean) == true_idx)
        print("FGM correct?", imagenet_labels.index(pred_fgm) == true_idx)
        print("PGD correct?", imagenet_labels.index(pred_pgd) == true_idx)

    print("------------------------------------------------------")