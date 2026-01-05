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
    """
    x = image_array.astype(np.float32)
    if x.ndim == 3:
        x = np.expand_dims(x, axis=0)  # (1,H,W,3)

    # VGG16 expects preprocess_input on RGB values in [0..255]
    x_in = preprocess_input(x.copy())
    probs = model.predict(x_in, verbose=0)[0]  # (1000,)

    pred_idx = int(np.argmax(probs))
    pred_prob = float(probs[pred_idx])

    t_idx = LABEL_TO_INDEX.get(target_label, None)
    if t_idx is not None and pred_idx == t_idx:
        return float(probs[t_idx])
    else:
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
    K = 25          # neighbors per iteration
    patch = 32      # patch size (try 48 if too weak)
    sigma = max_delta * 0.5  # noise scale inside patch

    for _ in range(K):
        nb = seed.copy()

        # random patch location
        y0 = np.random.randint(0, max(1, h - patch))
        x0 = np.random.randint(0, max(1, w - patch))
        y1 = min(h, y0 + patch)
        x1 = min(w, x0 + patch)

        # gaussian patch noise, clipped to L∞ budget
        noise = np.random.normal(0.0, sigma, size=(y1 - y0, x1 - x0, c)).astype(np.float32)
        noise = np.clip(noise, -max_delta, max_delta)

        nb[y0:y1, x0:x1, :] += noise

        # Enforce L∞ relative to seed (exactly)
        nb = np.clip(nb, seed - max_delta, seed + max_delta)

        # Clamp to valid pixel range
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
    Hill-climbing loop.

    Requirements:
        ✓ Start from initial_seed
        ✓ EACH iteration:
              - Generate neighbors using mutate_seed()
              - Enforce SAME L∞ bound relative to initial_seed
              - Add current image to candidates (elitism)
              - Use select_best() to pick the winner
        ✓ Accept new candidate only if fitness improves (decreases)
    """
    init = initial_seed.astype(np.float32)
    max_delta = 255.0 * float(epsilon)

    current = init.copy()
    current_fit = compute_fitness(current, model, target_label)

    no_improve = 0
    patience = 120

    print(f"Initial fitness: {current_fit:.5f}")

    for it in range(iterations):
        neighbors = mutate_seed(current, epsilon)

        # Enforce SAME L∞ bound relative to ORIGINAL seed
        clipped = []
        for nb in neighbors:
            nb = np.clip(nb, init - max_delta, init + max_delta)
            nb = np.clip(nb, 0.0, 255.0)
            clipped.append(nb.astype(np.float32))

        # Elitism
        candidates = [current] + clipped

        best_img, best_fit = select_best(candidates, model, target_label)

        if best_fit < current_fit:
            current, current_fit = best_img, best_fit
            no_improve = 0
        else:
            no_improve += 1

        if it % 5 == 0:
            print(f"Iter {it:3d} | fitness={current_fit:.5f}")

        # Negative fitness means top-1 != target_label (broken).
        # "Confidently broken" if wrong top-1 prob > 0.9 => fitness < -0.9
        if current_fit < -0.90:
            print(f"Confidently broken at iter {it}: fitness={current_fit:.5f}")
            break

        if no_improve >= patience:
            print(f"No improvement for {patience} iters. Stopping.")
            break

    return current, float(current_fit)


# ============================================================
# 5. PROGRAM ENTRY POINT FOR RUNNING A SINGLE ATTACK
# ============================================================
if __name__ == "__main__":
    model = vgg16.VGG16(weights="imagenet")

    with open("data/image_labels.json") as f:
        image_list = json.load(f)

    item = image_list[0]
    image_path = "images/" + item["image"]
    target_label = item["label"]

    print(f"Loaded image: {image_path}")
    print(f"Target label: {target_label}")

    img = load_img(image_path, target_size=(224, 224))
    plt.imshow(img)
    plt.title("Original image")
    plt.show()

    seed = img_to_array(img).astype(np.float32)

    # Baseline predictions (top-5) with correct preprocessing
    print("\nBaseline predictions (top-5):")
    preds = model.predict(preprocess_input(np.expand_dims(seed, axis=0)), verbose=0)
    for cl in decode_predictions(preds, top=5)[0]:
        print(f"{cl[1]:20s}  prob={cl[2]:.5f}")

    final_img, final_fitness = hill_climb(
        initial_seed=seed,
        model=model,
        target_label=target_label,
        epsilon=0.30,
        iterations=300
    )

    print("\nFinal fitness:", final_fitness)

    plt.imshow(array_to_img(final_img))
    plt.title(f"Adversarial Result — fitness={final_fitness:.4f}")
    plt.show()

    final_preds = model.predict(preprocess_input(np.expand_dims(final_img, axis=0)), verbose=0)
    print("\nFinal predictions:")
    for cl in decode_predictions(final_preds, top=5)[0]:
        print(cl)