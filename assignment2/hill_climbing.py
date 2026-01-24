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
# 1. FITNESS FUNCTION (margin-based, smoother signal)
# ============================================================
def compute_fitness(image_array: np.ndarray, model, target_label: str) -> float:
    global HC_QUERY_COUNT
    HC_QUERY_COUNT += 1

    true_idx = get_true_index_or_none(target_label)
    if true_idx is None:
        return 1.0

    x = image_array.astype(np.float32)
    if x.ndim == 3:
        x = np.expand_dims(x, axis=0)

    probs = model.predict(preprocess_input(x.copy()), verbose=0)[0]

    p_true = float(probs[true_idx])
    top1 = int(np.argmax(probs))
    p_top1 = float(probs[top1])

    if top1 != true_idx:
        # best other is already the top1
        p_best_other = p_top1
    else:
        # need second best; do a copy only in this case
        probs2 = probs.copy()
        probs2[true_idx] = -1.0
        p_best_other = float(np.max(probs2))

    return p_true - p_best_other


# ============================================================
# 2. MUTATION FUNCTION (stronger neighbor generation)
# ============================================================
def mutate_seed(
    seed: np.ndarray,
    epsilon: float
) -> List[np.ndarray]:
    """
    Produce mutated neighbors that satisfy:
        |neighbor[i,j,c] - seed[i,j,c]| <= 255 * epsilon

    Mix of:
      A) Sign patch (strong boundary push)
      B) Gaussian patch (fine-grained)
      C) Low-frequency block noise (very effective on CNNs)
      D) Sparse pixel noise (diversity)
    """
    seed = seed.astype(np.float32)
    max_delta = 255.0 * float(epsilon)
    h, w, c = seed.shape

    neighbors: List[np.ndarray] = []

    # --- knobs (reasonable defaults) ---
    K_sign_patch = 16
    K_gauss_patch = 12
    K_lowfreq = 4
    K_sparse = 12

    # Patch sizes
    patch_big = 48
    patch_med = 32

    # A) Sign patch noise (± amplitude in a patch)
    # Use a fraction of max_delta to avoid saturating too fast.
    amp = 0.65 * max_delta
    for _ in range(K_sign_patch):
        nb = seed.copy()

        patch = patch_big if np.random.rand() < 0.5 else patch_med
        y0 = np.random.randint(0, max(1, h - patch))
        x0 = np.random.randint(0, max(1, w - patch))
        y1 = min(h, y0 + patch)
        x1 = min(w, x0 + patch)

        noise = np.random.choice([-1.0, 1.0], size=(y1 - y0, x1 - x0, c)).astype(np.float32)
        nb[y0:y1, x0:x1, :] += noise * amp

        nb = np.clip(nb, seed - max_delta, seed + max_delta)
        nb = np.clip(nb, 0.0, 255.0)
        neighbors.append(nb)

    # B) Gaussian patch noise (smaller variance for refinement)
    sigma = 0.35 * max_delta
    for _ in range(K_gauss_patch):
        nb = seed.copy()

        patch = patch_med
        y0 = np.random.randint(0, max(1, h - patch))
        x0 = np.random.randint(0, max(1, w - patch))
        y1 = min(h, y0 + patch)
        x1 = min(w, x0 + patch)

        noise = np.random.normal(0.0, sigma, size=(y1 - y0, x1 - x0, c)).astype(np.float32)
        noise = np.clip(noise, -max_delta, max_delta)

        nb[y0:y1, x0:x1, :] += noise

        nb = np.clip(nb, seed - max_delta, seed + max_delta)
        nb = np.clip(nb, 0.0, 255.0)
        neighbors.append(nb)

    # C) Low-frequency / block noise
    # Create noise on a coarse grid and upscale via repeat (nearest neighbor).
    # This tends to move CNN features coherently and improves success.
    # Grid sizes that divide 224 nicely: 7, 8, 14, 16, 28
    for _ in range(K_lowfreq):
        nb = seed.copy()

        g = np.random.choice([7, 8, 14, 16, 28])
        gh, gw = g, g
        # coarse noise in [-max_delta, max_delta]
        coarse = np.random.uniform(-1.0, 1.0, size=(gh, gw, c)).astype(np.float32)

        # scale amplitude (don’t always max out)
        amp = np.random.uniform(0.10, 0.35) * max_delta
        coarse *= amp
        # upscale by repeating blocks
        rep_h = int(np.ceil(h / gh))
        rep_w = int(np.ceil(w / gw))
        up = np.repeat(np.repeat(coarse, rep_h, axis=0), rep_w, axis=1)[:h, :w, :]

        nb += up
        nb = np.clip(nb, seed - max_delta, seed + max_delta)
        nb = np.clip(nb, 0.0, 255.0)
        neighbors.append(nb)

    # D) Sparse pixel noise (diversity + small edits)
    frac = 0.004  # slightly lower than before
    n_pix = max(1, int(h * w * frac))
    for _ in range(K_sparse):
        nb = seed.copy()
        ys = np.random.randint(0, h, size=n_pix)
        xs = np.random.randint(0, w, size=n_pix)

        noise = np.random.uniform(-max_delta, max_delta, size=(n_pix, c)).astype(np.float32)
        # dampen a bit so it doesn't look like pure salt-pepper
        noise *= 0.55

        nb[ys, xs, :] += noise
        nb = np.clip(nb, seed - max_delta, seed + max_delta)
        nb = np.clip(nb, 0.0, 255.0)
        neighbors.append(nb)

    return neighbors


# ============================================================
# 3. SELECT BEST CANDIDATE (keep, minor robustness)
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
    if len(candidates) == 0:
        raise ValueError("select_best: empty candidates list")

    best_img = candidates[0]
    best_fit = compute_fitness(best_img, model, target_label)

    for cand in candidates[1:]:
        fit = compute_fitness(cand, model, target_label)
        if fit < best_fit:
            best_fit = fit
            best_img = cand

    return best_img.copy(), float(best_fit)


# ============================================================
# 4. HILL-CLIMBING ALGORITHM (random restarts + mild SA)
# ============================================================
def hill_climb(
    initial_seed: np.ndarray,
    model,
    target_label: str,
    epsilon: float = 0.30,
    iterations: int = 300
) -> Tuple[np.ndarray, float]:
    """
    Hill-climbing loop (black-box) with:
      - L∞ bound enforced relative to ORIGINAL seed (required)
      - Random restarts (improves success a lot)
      - Mild simulated annealing acceptance to escape plateaus

    Success: fitness < 0 (equivalent to adv_top1 != true for margin fitness).
    """
    global HC_LAST_ITERS, HC_QUERY_COUNT

    init = initial_seed.astype(np.float32)
    max_delta = 255.0 * float(epsilon)

    # If already adversarial w.r.t. ground truth, return immediately
    best_overall = init.copy()
    best_overall_fit = compute_fitness(best_overall, model, target_label)
    if best_overall_fit < 0.0:
        HC_LAST_ITERS = 0
        return best_overall, float(best_overall_fit)

    # Restart budget: tradeoff runtime vs success
    # 5 restarts is a good baseline; still within typical assignment budgets.
    R = 5

    # Split iterations across restarts (more early exploration)
    # Ensure at least ~40 iters per restart.
    iters_per = max(40, iterations // R)

    total_iters_used = 0
    MAX_QUERIES = 2000

    for r in range(R):
        # Random start inside ε-ball around init
        start = init.copy()
        restart_scale = 0.10  # 10% of epsilon at restart
        start += np.random.uniform(-restart_scale * max_delta, restart_scale * max_delta, size=start.shape)
        start = np.clip(start, init - max_delta, init + max_delta)
        start = np.clip(start, 0.0, 255.0)

        current = start
        current_fit = compute_fitness(current, model, target_label)

        # SA temperature schedule
        T0 = 0.20  # higher => more exploration
        Tend = 0.02

        no_improve = 0
        patience = max(20, iters_per // 3)
        for it in range(iters_per):
            if HC_QUERY_COUNT >= MAX_QUERIES:
                HC_LAST_ITERS = total_iters_used
                return best_overall, float(best_overall_fit)
            total_iters_used += 1

            # Generate neighbors around current
            neighbors = mutate_seed(current, epsilon)

            # Enforce global L∞ around ORIGINAL init (required by assignment)
            clipped = []
            for nb in neighbors:
                nb = np.clip(nb, init - max_delta, init + max_delta)
                nb = np.clip(nb, 0.0, 255.0)
                clipped.append(nb.astype(np.float32))

            # Always include current
            candidates = [current] + clipped
            best_img, best_fit = select_best(candidates, model, target_label)

            # Accept rule: greedy or SA escape
            if best_fit < current_fit:
                accept = True
            else:
                # temperature decays linearly within this restart
                t = it / max(1, iters_per - 1)
                T = (1 - t) * T0 + t * Tend
                # probability of accepting a worse move
                delta = float(best_fit - current_fit)
                accept = (np.random.rand() < np.exp(-delta / max(1e-6, T)))

            if accept:
                current, current_fit = best_img, best_fit

            # Track overall best
            if current_fit < best_overall_fit:
                best_overall, best_overall_fit = current.copy(), float(current_fit)
                no_improve = 0
            else:
                no_improve += 1

            # Success condition (margin < 0 means some other class beats true class)
            if best_overall_fit < 0.0:
                HC_LAST_ITERS = total_iters_used
                return best_overall, float(best_overall_fit)

            # Plateau stop for this restart
            if no_improve >= patience:
                break

    HC_LAST_ITERS = total_iters_used
    return best_overall, float(best_overall_fit)
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