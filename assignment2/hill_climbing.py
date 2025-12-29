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
from keras.applications.imagenet_utils import decode_predictions
from keras.utils import array_to_img, load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
import numpy as np
import torch
from typing import List, Tuple
# ImageNet normalization for torchvision models
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ============================================================
# 1. FITNESS FUNCTION (KERAS)
# ============================================================
def compute_fitness(
    image_array: np.ndarray,
    model,
    target_label: str
) -> float:
    """
    Fitness (LOWER is better):
      - If top-1 == target_label: fitness = P(target_label)
      - Else:                     fitness = -P(top-1)
    """
    x = image_array.astype(np.float32)
    if x.ndim == 3:
        x = np.expand_dims(x, axis=0)  # (1,H,W,3)

    # VGG16 expects preprocess_input on RGB [0..255]
    x_in = preprocess_input(x.copy())
    preds = model.predict(x_in, verbose=0)  # (1,1000)

    top1 = decode_predictions(preds, top=1)[0][0]  # (synset, label, prob)
    pred_label = top1[1]
    pred_prob = float(top1[2])

    if pred_label == target_label:
        return pred_prob
    else:
        return -pred_prob


# ============================================================
# 2. MUTATION FUNCTION (KERAS)
# ============================================================
def mutate_seed(
    seed: np.ndarray,
    epsilon: float
) -> List[np.ndarray]:
    """
    Create neighbors s.t. ||neighbor - seed||_inf <= 255*epsilon.
    """
    seed = seed.astype(np.float32)
    max_pert = 255.0 * float(epsilon)
    h, w, c = seed.shape

    neighbors: List[np.ndarray] = []

    K = 30                       # how many neighbors
    M = max(1, (h * w) // 150)   # how many pixels per neighbor (~0.67%)

    for _ in range(K):
        nb = seed.copy()

        ys = np.random.randint(0, h, size=M)
        xs = np.random.randint(0, w, size=M)

        delta = np.random.uniform(-max_pert, max_pert, size=(M, c)).astype(np.float32)
        nb[ys, xs, :] = nb[ys, xs, :] + delta

        # enforce L∞ relative to seed
        nb = np.clip(nb, seed - max_pert, seed + max_pert)

        # keep valid pixel range
        nb = np.clip(nb, 0.0, 255.0)

        neighbors.append(nb)

    return neighbors


# ============================================================
# 3. SELECT BEST (KERAS)
# ============================================================
def select_best(
    candidates: List[np.ndarray],
    model,
    target_label: str
) -> Tuple[np.ndarray, float]:
    """
    Return candidate with LOWEST fitness.
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
# 4. HILL CLIMB (KERAS)
# ============================================================
def hill_climb(
    initial_seed: np.ndarray,
    model,
    target_label: str,
    epsilon: float = 0.30,
    iterations: int = 300
) -> Tuple[np.ndarray, float]:
    """
    Hill-climbing with global L∞ bound relative to initial_seed.
    Accept only if fitness decreases.
    """
    init = initial_seed.astype(np.float32)
    max_pert = 255.0 * float(epsilon)

    current = init.copy()
    current_fit = compute_fitness(current, model, target_label)

    no_improve = 0
    patience = 60

    print(f"Initial fitness: {current_fit:.5f}")

    for it in range(iterations):
        neighbors = mutate_seed(current, epsilon)

        # Enforce SAME L∞ bound relative to ORIGINAL seed (required)
        clipped = []
        for nb in neighbors:
            nb = np.clip(nb, init - max_pert, init + max_pert)
            nb = np.clip(nb, 0.0, 255.0)
            clipped.append(nb.astype(np.float32))

        # elitism
        candidates = [current] + clipped

        best_img, best_fit = select_best(candidates, model, target_label)

        if best_fit < current_fit:
            current, current_fit = best_img, best_fit
            no_improve = 0
        else:
            no_improve += 1

        if it % 10 == 0:
            print(f"Iter {it:3d} | fitness={current_fit:.5f}")

        # Attack success condition for this fitness:
        # negative fitness => top-1 != target label (broken)
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
    # Load classifier
    model = vgg16.VGG16(weights="imagenet")

    # Load JSON describing dataset
    with open("data/image_labels.json") as f:
        image_list = json.load(f)

    # Pick first entry
    item = image_list[0]
    image_path = "images/" + item["image"]
    target_label = item["label"]

    print(f"Loaded image: {image_path}")
    print(f"Target label: {target_label}")

    img = load_img(image_path, target_size=(224, 224))
    plt.imshow(img)
    plt.title("Original image")
    plt.show()

    img_array = img_to_array(img)
    seed = img_array.copy()

    # Print baseline top-5 predictions
    print("\nBaseline predictions (top-5):")
    preds = model.predict(preprocess_input(np.expand_dims(seed.astype(np.float32), axis=0)))
    for cl in decode_predictions(preds, top=5)[0]:
        print(f"{cl[1]:20s}  prob={cl[2]:.5f}")

    # Run hill climbing attack
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

    # Print final predictions
    final_preds = model.predict(np.expand_dims(final_img, axis=0))
    print("\nFinal predictions:")
    for cl in decode_predictions(final_preds, top=5)[0]:
        print(cl)