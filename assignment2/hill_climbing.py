"""
Assignment 2 – Adversarial Image Attack via Hill Climbing

You MUST implement:
    - compute_fitness
    - mutate_seed
    - select_best
    - hill_climb

DO NOT change function signatures.
"""

import json
import os
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from keras.applications import vgg16
from keras.applications.vgg16 import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from keras.utils import array_to_img, load_img, img_to_array

# Map human-readable ImageNet label -> class index (0..999)
import json
import urllib.request

IMAGENET_CLASS_INDEX_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
)

with urllib.request.urlopen(IMAGENET_CLASS_INDEX_URL) as f:
    CLASS_INDEX = json.load(f)

# Map human-readable label -> class index (0..999)
LABEL_TO_INDEX = {v[1]: int(k) for k, v in CLASS_INDEX.items()}

# ============================================================
# Globals for measurement (no signature changes)
# ============================================================
HC_QUERY_COUNT = 0      # number of model.predict calls
HC_LAST_ITERS = 0       # iterations used in last hill_climb run


# ============================================================
# Metrics helpers (image arrays are HWC float in [0..255])
# ============================================================
def hc_perturbation_metrics(clean: np.ndarray, adv: np.ndarray, thresh: float = 1.0):
    """
    clean, adv: HWC float arrays in [0..255]
    thresh: pixel-change threshold in same units (1.0 ~= 1/255 in normalized space)

    Returns:
      pixels_changed: count of (H,W) locations where any channel changed > thresh
      linf_01: max abs perturbation / 255 in [0,1]
    """
    diff = np.abs(adv.astype(np.float32) - clean.astype(np.float32))  # HWC
    linf_raw = float(diff.max())
    linf_01 = linf_raw / 255.0

    per_pixel = diff.max(axis=2)  # HW
    pixels_changed = int((per_pixel > thresh).sum())
    return pixels_changed, linf_01


def top1_label_prob_keras(model, image_hwc_255: np.ndarray):
    """
    Returns (label_str, prob_float) for top-1 prediction.
    image_hwc_255: HWC float in [0..255]
    """
    x = image_hwc_255.astype(np.float32)
    if x.ndim == 3:
        x = np.expand_dims(x, axis=0)  # 1,H,W,3

    x_in = preprocess_input(x.copy())
    probs = model.predict(x_in, verbose=0)
    top1 = decode_predictions(probs, top=1)[0][0]  # (synset, label, prob)
    return top1[1], float(top1[2])


# ============================================================
# 1. FITNESS FUNCTION
# ============================================================
def compute_fitness(
    image_array: np.ndarray,
    model,
    target_label: str
) -> float:
    """
    Fitness definition (LOWER is better):
        - If the model predicts target_label:
              fitness = probability(target_label)
        - Otherwise:
              fitness = -probability(predicted_label)

    Note: This is black-box: uses only model outputs.
    """
    global HC_QUERY_COUNT
    HC_QUERY_COUNT += 1

    x = image_array.astype(np.float32)
    if x.ndim == 3:
        x = np.expand_dims(x, axis=0)  # (1,H,W,3)

    x_in = preprocess_input(x.copy())
    probs = model.predict(x_in, verbose=0)[0]  # (1000,)

    pred_idx = int(np.argmax(probs))
    pred_prob = float(probs[pred_idx])

    # We avoid any internet dependency: compare using decoded top-1 label string
    # (same approach you used successfully earlier)
    pred_label = decode_predictions(np.expand_dims(probs, axis=0), top=1)[0][0][1]

    if pred_label == target_label:
        # If still correct, higher prob should be "worse", so positive value.
        return pred_prob
    else:
        # If misclassified, negative with magnitude = confidence in wrong class.
        return -pred_prob


# ============================================================
# 2. MUTATION FUNCTION
# ============================================================
def mutate_seed(
    seed: np.ndarray,
    epsilon: float
) -> List[np.ndarray]:
    """
    Produce mutated neighbors that satisfy:
        |neighbor[i,j,c] - seed[i,j,c]| <= 255 * epsilon
    """
    seed = seed.astype(np.float32)
    max_delta = 255.0 * float(epsilon)
    h, w, c = seed.shape

    neighbors: List[np.ndarray] = []

    # Tunable knobs
    K = 25
    patch = 32
    sigma = max_delta * 0.5

    for _ in range(K):
        nb = seed.copy()

        y0 = np.random.randint(0, max(1, h - patch))
        x0 = np.random.randint(0, max(1, w - patch))
        y1 = min(h, y0 + patch)
        x1 = min(w, x0 + patch)

        noise = np.random.normal(0.0, sigma, size=(y1 - y0, x1 - x0, c)).astype(np.float32)
        noise = np.clip(noise, -max_delta, max_delta)

        nb[y0:y1, x0:x1, :] += noise

        # Enforce L∞ relative to seed (exactly)
        nb = np.clip(nb, seed - max_delta, seed + max_delta)
        nb = np.clip(nb, 0.0, 255.0)

        neighbors.append(nb.astype(np.float32))

    return neighbors


# ============================================================
# 3. SELECT BEST CANDIDATE
# ============================================================
def select_best(
    candidates: List[np.ndarray],
    model,
    target_label: str
) -> Tuple[np.ndarray, float]:
    """
    Evaluate fitness for all candidates and return the one with
    the LOWEST fitness score.
    """
    best_img = candidates[0]
    best_fit = compute_fitness(best_img, model, target_label)

    for cand in candidates[1:]:
        fit = compute_fitness(cand, model, target_label)
        if fit < best_fit:
            best_fit = fit
            best_img = cand

    return best_img.copy(), float(best_fit)


# ============================================================
# 4. HILL-CLIMBING ALGORITHM
# ============================================================
def hill_climb(
    initial_seed: np.ndarray,
    model,
    target_label: str,
    epsilon: float = 0.30,
    iterations: int = 300
) -> Tuple[np.ndarray, float]:
    """
    Hill-climbing loop (black-box).
    """
    global HC_LAST_ITERS

    init = initial_seed.astype(np.float32)
    max_delta = 255.0 * float(epsilon)

    current = init.copy()
    current_fit = compute_fitness(current, model, target_label)

    no_improve = 0
    patience = 120

    for it in range(iterations):
        neighbors = mutate_seed(current, epsilon)

        # Enforce SAME L∞ bound relative to ORIGINAL seed (required)
        clipped = []
        for nb in neighbors:
            nb = np.clip(nb, init - max_delta, init + max_delta)
            nb = np.clip(nb, 0.0, 255.0)
            clipped.append(nb.astype(np.float32))

        candidates = [current] + clipped
        best_img, best_fit = select_best(candidates, model, target_label)

        if best_fit < current_fit:
            current, current_fit = best_img, best_fit
            no_improve = 0
        else:
            no_improve += 1

        # stop if confidently broken: wrong top-1 prob > 0.9 => fitness < -0.9
        if current_fit < -0.90:
            HC_LAST_ITERS = it + 1
            return current, float(current_fit)

        if no_improve >= patience:
            HC_LAST_ITERS = it + 1
            return current, float(current_fit)

    HC_LAST_ITERS = iterations
    return current, float(current_fit)


# ============================================================
# 5. PROGRAM ENTRY POINT: run HC over entire dataset + write CSV
# ============================================================
if __name__ == "__main__":
    EPS = 0.30
    ITERATIONS = 300

    JSON_FILE = "data/image_labels.json"
    IMAGE_DIR = "images"
    OUTDIR = "hc_results"
    os.makedirs(OUTDIR, exist_ok=True)
    METRICS_CSV = os.path.join(OUTDIR, "metrics_hc.csv")

    model = vgg16.VGG16(weights="imagenet")

    with open(JSON_FILE) as f:
        items = json.load(f)

    rows = []
    for entry in items:
        image_file = entry["image"]
        target_label = entry["label"]
        img_path = os.path.join(IMAGE_DIR, image_file)

        # Load seed in correct size
        img = load_img(img_path, target_size=(224, 224))
        seed = img_to_array(img).astype(np.float32)  # HWC [0..255]

        # Clean prediction
        clean_label, clean_prob = top1_label_prob_keras(model, seed)

        # Save clean image
        array_to_img(seed).save(os.path.join(OUTDIR, f"{image_file}_clean.png"))

        # HC attack
        HC_QUERY_COUNT = 0
        t0 = time.perf_counter()
        adv, final_fit = hill_climb(seed, model, target_label, epsilon=EPS, iterations=ITERATIONS)
        dt = time.perf_counter() - t0

        adv_label, adv_prob = top1_label_prob_keras(model, adv)
        array_to_img(adv).save(os.path.join(OUTDIR, f"{image_file}_hc.png"))

        # success = top-1 differs from ground-truth label (assignment definition)
        hc_success = int(adv_label != target_label)

        # perturbation metrics
        pixels_changed, linf_01 = hc_perturbation_metrics(seed, adv, thresh=1.0)

        rows.append({
            "image": image_file,
            "label": target_label,

            "clean_top1": clean_label,
            "clean_prob": clean_prob,
            "clean_correct": int(clean_label == target_label),

            "hc_top1": adv_label,
            "hc_prob": adv_prob,
            "hc_success": hc_success,
            "hc_pixels_changed": pixels_changed,
            "hc_linf": linf_01,
            "hc_time_s": dt,
            "hc_iterations_used": HC_LAST_ITERS,
            "hc_model_queries": HC_QUERY_COUNT,
            "final_fitness": final_fit,
        })

        print(f"\n[{image_file}] label={target_label} | clean={clean_label} ({clean_prob:.3f})"
              f" | hc={adv_label} ({adv_prob:.3f}) | success={hc_success}"
              f" | linf={linf_01:.3f} | pixels={pixels_changed} | time={dt:.2f}s"
              f" | iters={HC_LAST_ITERS} | queries={HC_QUERY_COUNT}")

    # Write CSV
    with open(METRICS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Print aggregates
    def avg(xs): return float(sum(xs) / max(1, len(xs)))
    success_rate = avg([r["hc_success"] for r in rows])
    px_avg = avg([r["hc_pixels_changed"] for r in rows])
    linf_max = max(r["hc_linf"] for r in rows)
    t_avg = avg([r["hc_time_s"] for r in rows])
    it_avg = avg([r["hc_iterations_used"] for r in rows])
    q_avg = avg([r["hc_model_queries"] for r in rows])

    print("\n=== HC Aggregate metrics (over dataset) ===")
    print(f"HC success rate: {success_rate:.2f}")
    print(f"HC avg pixels changed: {px_avg:.1f}")
    print(f"HC max Linf: {linf_max:.4f} (expected <= {EPS:.2f})")
    print(f"HC avg time/image: {t_avg:.4f}s")
    print(f"HC avg iterations used: {it_avg:.1f}")
    print(f"HC avg model queries: {q_avg:.1f}")
    print(f"\nSaved HC metrics to: {METRICS_CSV}")
