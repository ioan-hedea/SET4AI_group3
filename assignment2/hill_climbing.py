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


# ============================================================
# 1. FITNESS FUNCTION
# ============================================================

def compute_fitness(
    image_array: np.ndarray,
    model,
    target_label: str
) -> float:
    """
    Compute fitness of an image for hill climbing.

    Fitness definition (LOWER is better):
        - If the model predicts target_label:
              fitness = probability(target_label)
        - Otherwise:
              fitness = -probability(predicted_label)
    """

    # TODO (student)
    
    predictions = model.predict(image_array)
    predicted_label = model.get_label(predictions)
    target_prob = model.get_probability(predictions, target_label)
    
    if predicted_label == target_label:
        # Success! As target_prob goes up to 1.0, fitness goes down to 0.0
        fitness = 1.0 - target_prob
    else:
        # Not there yet. Penalize based on how far target_prob is from 1.0
        fitness = 1.0 + (1.0 - target_prob)
        
    return fitness


# ============================================================
# 2. MUTATION FUNCTION
# ============================================================

def mutate_seed(
    seed: np.ndarray,
    epsilon: float
) -> List[np.ndarray]:
    """
    Produce ANY NUMBER of mutated neighbors.

    Students may implement ANY mutation strategy:
        - modify 1 pixel
        - modify multiple pixels
        - patch-based mutation
        - channel-based mutation
        - gaussian noise (clipped)
        - etc.

    BUT EVERY neighbor must satisfy the L∞ constraint:

        For all pixels i,j,c:
            |neighbor[i,j,c] - seed[i,j,c]| <= 255 * epsilon

    Requirements:
        ✓ Return a list of neighbors: [neighbor1, neighbor2, ..., neighborK]
        ✓ K can be ANY size ≥ 1
        ✓ Neighbors must be deep copies of seed
        ✓ Pixel values must remain in [0, 255]
        ✓ Must obey the L∞ bound exactly

    Args:
        seed (np.ndarray): input image
        epsilon (float): allowed perturbation budget

    Returns:
        List[np.ndarray]: mutated neighbors
    """

    # TODO (student)
    neighbors = []
    # The absolute max/min bounds allowed by the epsilon budget
    max_delta = 255 * epsilon
    
    # Calculate bounds relative to the ORIGINAL image to prevent drift
    lower_bound = np.clip(original_image - max_delta, 0, 255)
    upper_bound = np.clip(original_image + max_delta, 0, 255)

    for _ in range(K):
        # 1. Start with a deep copy
        neighbor = seed.copy().astype(np.float32)
        
        # 2. Strategy: Mix different types of noise
        # Some neighbors get small global noise, some get large local noise
        mutation_type = np.random.choice(['global', 'sparse', 'patch'])
        
        if mutation_type == 'global':
            # Add uniform noise to the whole image
            noise = np.random.uniform(-max_delta, max_delta, seed.shape)
            neighbor += noise
            
        elif mutation_type == 'sparse':
            # Change only 10% of pixels, but change them significantly
            mask = np.random.random(seed.shape) < 0.10
            noise = np.random.uniform(-max_delta, max_delta, seed.shape)
            neighbor[mask] += noise[mask]
            
        elif mutation_type == 'patch':
            # Change a random 8x8 square patch
            h, w = seed.shape[:2]
            ph, pw = 8, 8
            y = np.random.randint(0, h - ph)
            x = np.random.randint(0, w - pw)
            neighbor[y:y+ph, x:x+pw] += np.random.uniform(-max_delta, max_delta, (ph, pw, seed.shape[2]))

        # 3. CRITICAL: The L-infinity Constraint
        # This ensures we never move further than epsilon from the ORIGINAL image
        neighbor = np.clip(neighbor, lower_bound, upper_bound)
        
        # 4. Final cast back to original data type with proper rounding
        neighbors.append(np.round(neighbor).astype(seed.dtype))
    
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

    Args:
        candidates (List[np.ndarray])
        model: classifier
        target_label (str)

    Returns:
        (best_image, best_fitness)
    """

    # TODO (student)
    best_image = None
    best_fitness = float('inf')  # Start with worst possible fitness
    
    for candidate in candidates:
        fitness = compute_fitness(candidate, model, target_label)
        
        # Lower fitness is better
        if fitness < best_fitness:
            best_fitness = fitness
            best_image = candidate.copy()
    
    return best_image, best_fitness


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
    Main hill-climbing loop.

    Requirements:
        ✓ Start from initial_seed
        ✓ EACH iteration:
              - Generate ANY number of neighbors using mutate_seed()
              - Enforce the SAME L∞ bound relative to initial_seed
              - Add current image to candidates (elitism)
              - Use select_best() to pick the winner
        ✓ Accept new candidate only if fitness improves
        ✓ Stop if:
              - target class is broken confidently, OR
              - no improvement for multiple steps (optional)

    Returns:
        (final_image, final_fitness)
    """

    # TODO (team work)
    current_image = initial_seed.copy()
    current_fitness = compute_fitness(current_image, model, target_label)
    
    no_improvement_count = 0
    max_no_improvement = 50  # Stop if no improvement for 50 iterations
    
    print(f"Initial fitness: {current_fitness:.4f}")
    
    for iteration in range(iterations):
        # Generate neighbors from current image, constrained by initial_seed
        neighbors = mutate_seed(current_image, epsilon, initial_seed, K)
        
        # Add current image to candidates (elitism)
        candidates = [current_image] + neighbors
        
        # Select best candidate
        best_image, best_fitness = select_best(candidates, model, target_label)
        
        # Check if fitness improved
        if best_fitness < current_fitness:
            current_image = best_image.copy()
            current_fitness = best_fitness
            no_improvement_count = 0
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: fitness = {current_fitness:.4f}")
            
            # Check if target is confidently reached
            if current_fitness < 0.1:  # Very close to 0 = High confidence success
                print(f"Target reached with high confidence at iteration {iteration}!")
                break
        else:
            no_improvement_count += 1
        
        # Stop if no improvement for too long
        if no_improvement_count >= max_no_improvement:
            print(f"No improvement for {max_no_improvement} iterations. Stopping.")
            break
    
    print(f"Final fitness: {current_fitness:.4f}")
    return current_image, current_fitness
    


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

    img = load_img(image_path)
    plt.imshow(img)
    plt.title("Original image")
    plt.show()

    img_array = img_to_array(img)
    seed = img_array.copy()

    # Print baseline top-5 predictions
    print("\nBaseline predictions (top-5):")
    preds = model.predict(np.expand_dims(seed, axis=0))
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