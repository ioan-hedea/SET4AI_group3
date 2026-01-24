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

# ============================================================
# Label mapping: use the provided canonical ImageNet label list
# (same file used by baselines)
# ============================================================
IMAGENET_LABELS = None
LABEL_TO_INDEX = None


def load_imagenet_labels(labels_file: str = "data/imagenet_classes.txt"):
    global IMAGENET_LABELS, LABEL_TO_INDEX
    if IMAGENET_LABELS is not None:
        return
    with open(labels_file, "r") as f:
        IMAGENET_LABELS = [line.strip() for line in f.readlines()]
    LABEL_TO_INDEX = {lab: i for i, lab in enumerate(IMAGENET_LABELS)}


def get_true_index_or_none(human_label: str):
    """
    Returns the class index (0..999) or None if not found.
    We normalize to match baseline conventions.
    """
    if LABEL_TO_INDEX is None:
        load_imagenet_labels()

    candidates = [
        human_label,
        human_label.replace("_", " "),
        human_label.replace(" ", "_"),
    ]
    for c in candidates:
        if c in LABEL_TO_INDEX:
            return LABEL_TO_INDEX[c]
    return None


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
    Returns (top1_idx, label_str, prob_float) for top-1 prediction.
    image_hwc_255: HWC float in [0..255]
    """
    x = image_hwc_255.astype(np.float32)
    if x.ndim == 3:
        x = np.expand_dims(x, axis=0)

    x_in = preprocess_input(x.copy())
    probs = model.predict(x_in, verbose=0)[0]  # (1000,)

    top1_idx = int(np.argmax(probs))
    top1_prob = float(probs[top1_idx])

    # label string is only for display (do NOT use for success checks)
    top1_label = decode_predictions(np.expand_dims(probs, axis=0), top=1)[0][0][1]
    return top1_idx, top1_label, top1_prob


# ============================================================
# 1. FITNESS FUNCTION
# ============================================================
def compute_fitness(
    image_array: np.ndarray,
    model,
    target_label: str
) -> float:
    """
    Untargeted black-box fitness (LOWER is better):

    - If model is still correct (top-1 == true):   fitness = +p(true)
    - If model is wrong (top-1 != true):          fitness = -(1 - p(true))

    This guarantees:
      * Fitness is positive when not yet adversarial
      * Fitness becomes negative once misclassified
      * More confident misclassification => more negative => better
    """
    global HC_QUERY_COUNT
    HC_QUERY_COUNT += 1

    true_idx = get_true_index_or_none(target_label)
    if true_idx is None:
        return 1.0  # can't evaluate properly

    x = image_array.astype(np.float32)
    if x.ndim == 3:
        x = np.expand_dims(x, axis=0)

    x_in = preprocess_input(x.copy())
    probs = model.predict(x_in, verbose=0)[0]  # (1000,)

    p_true = float(probs[true_idx])
    top1_idx = int(np.argmax(probs))

    if top1_idx == int(true_idx):
        return p_true                 # still correct => positive
    else:
        return -(1.0 - p_true)        # misclassified => negative


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

    # --- knobs ---
    K_patch = 18
    K_sparse = 18

    # A) Patch noise
    patch = 32
    sigma = max_delta * 0.5

    for _ in range(K_patch):
        nb = seed.copy()

        y0 = np.random.randint(0, max(1, h - patch))
        x0 = np.random.randint(0, max(1, w - patch))
        y1 = min(h, y0 + patch)
        x1 = min(w, x0 + patch)

        noise = np.random.normal(0.0, sigma, size=(y1 - y0, x1 - x0, c)).astype(np.float32)
        noise = np.clip(noise, -max_delta, max_delta)

        nb[y0:y1, x0:x1, :] += noise

        # enforce L∞ around current seed (mutation-local)
        nb = np.clip(nb, seed - max_delta, seed + max_delta)
        nb = np.clip(nb, 0.0, 255.0)
        neighbors.append(nb)

    # B) Sparse pixel noise (random locations)
    frac = 0.005
    n_pix = max(1, int(h * w * frac))

    for _ in range(K_sparse):
        nb = seed.copy()
        ys = np.random.randint(0, h, size=n_pix)
        xs = np.random.randint(0, w, size=n_pix)

        noise = np.random.uniform(-max_delta, max_delta, size=(n_pix, c)).astype(np.float32)

        nb[ys, xs, :] += noise
        nb = np.clip(nb, seed - max_delta, seed + max_delta)
        nb = np.clip(nb, 0.0, 255.0)
        neighbors.append(nb)

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
    - Always enforces L∞ bound relative to ORIGINAL seed.
    - Success: top-1 != true class index (assignment definition).
    """
    global HC_LAST_ITERS, HC_QUERY_COUNT

    init = initial_seed.astype(np.float32)
    max_delta = 255.0 * float(epsilon)

    current = init.copy()
    current_fit = compute_fitness(current, model, target_label)

    no_improve = 0
    patience = 120

    true_idx = get_true_index_or_none(target_label)

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
        if current_fit < 0.0:
            HC_LAST_ITERS = it + 1
            return current, float(current_fit)
        # Stop when adversarial found (top-1 != true class)
        # if true_idx is not None:
        #     x_in = preprocess_input(np.expand_dims(current.astype(np.float32), axis=0).copy())
        #     probs = model.predict(x_in, verbose=0)[0]
        #     HC_QUERY_COUNT += 1
        #     if int(np.argmax(probs)) != int(true_idx):
        #         HC_LAST_ITERS = it + 1
        #         return current, float(current_fit)

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

    # Ensure label mapping is loaded once
    load_imagenet_labels("data/imagenet_classes.txt")

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

        true_idx = get_true_index_or_none(target_label)

        # Clean prediction
        clean_idx, clean_label, clean_prob = top1_label_prob_keras(model, seed)

        # Save clean image
        array_to_img(seed).save(os.path.join(OUTDIR, f"{image_file}_clean.png"))

        # HC attack
        HC_QUERY_COUNT = 0
        t0 = time.perf_counter()
        adv, final_fit = hill_climb(seed, model, target_label, epsilon=EPS, iterations=ITERATIONS)
        dt = time.perf_counter() - t0

        adv_idx, adv_label, adv_prob = top1_label_prob_keras(model, adv)
        array_to_img(adv).save(os.path.join(OUTDIR, f"{image_file}_hc.png"))

        # success = top-1 differs from ground-truth class index (assignment definition)
        clean_correct = int(true_idx is not None and clean_idx == true_idx)
        hc_success = int(true_idx is not None and adv_idx != true_idx)

        # perturbation metrics
        pixels_changed, linf_01 = hc_perturbation_metrics(seed, adv, thresh=1.0)

        rows.append({
            "image": image_file,
            "label": target_label,

            "true_idx": true_idx if true_idx is not None else -1,

            "clean_top1": clean_label,
            "clean_top1_idx": clean_idx,
            "clean_prob": clean_prob,
            "clean_correct": clean_correct,

            "hc_top1": adv_label,
            "hc_top1_idx": adv_idx,
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

    # ============================================================
    # Epsilon sweep (scalability vs epsilon)
    # ============================================================
    EPS_SWEEP = [0.05, 0.10, 0.20, 0.30]
    SWEEP_CSV = os.path.join(OUTDIR, "metrics_hc_eps_sweep.csv")

    sweep_rows = []
    for eps in EPS_SWEEP:
        for entry in items:
            image_file = entry["image"]
            target_label = entry["label"]
            img_path = os.path.join(IMAGE_DIR, image_file)

            img = load_img(img_path, target_size=(224, 224))
            seed = img_to_array(img).astype(np.float32)

            true_idx = get_true_index_or_none(target_label)

            HC_QUERY_COUNT = 0
            t0 = time.perf_counter()
            adv, final_fit = hill_climb(seed, model, target_label, epsilon=eps, iterations=ITERATIONS)
            dt = time.perf_counter() - t0

            adv_idx, adv_label, adv_prob = top1_label_prob_keras(model, adv)

            success = int(true_idx is not None and adv_idx != true_idx)
            pixels_changed, linf_01 = hc_perturbation_metrics(seed, adv, thresh=1.0)

            sweep_rows.append({
                "epsilon": eps,
                "image": image_file,
                "label": target_label,
                "true_idx": true_idx if true_idx is not None else -1,
                "hc_success": success,
                "hc_time_s": dt,
                "hc_iterations_used": HC_LAST_ITERS,
                "hc_model_queries": HC_QUERY_COUNT,
                "hc_pixels_changed": pixels_changed,
                "hc_linf": linf_01,
                "final_fitness": final_fit,
                "adv_top1": adv_label,
                "adv_top1_idx": adv_idx,
                "adv_prob": adv_prob,
            })

    # write sweep csv
    with open(SWEEP_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_rows)

    print(f"\nSaved epsilon sweep metrics to: {SWEEP_CSV}")