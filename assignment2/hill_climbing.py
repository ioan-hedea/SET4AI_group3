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
    # Get model predictions
    # Assuming model returns probabilities for each class
    # print(image_array.shape)
    # exit()
    predictions = model.predict(image_array)
    
    # Get the predicted label (highest probability)
    predicted_label = model.get_label(predictions)
    
    # Get probability of target label
    target_prob = model.get_probability(predictions, target_label)
    
    # If model predicts target label, fitness is the probability (we want to maximize it)
    # Otherwise, fitness is negative of the predicted label's probability
    if predicted_label == target_label:
        fitness = target_prob
    else:
        predicted_prob = model.get_probability(predictions, predicted_label)
        fitness = -predicted_prob
    
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
    max_perturbation = 255 * epsilon
    for channel in range(seed.shape[2]):
        neighbor = seed.copy()
        # Perturb entire channel
        perturbation = np.random.uniform(-max_perturbation, max_perturbation, 
                                        (seed.shape[0], seed.shape[1]))
        neighbor[:, :, channel] = seed[:, :, channel] + perturbation
        neighbor[:, :, channel] = np.clip(neighbor[:, :, channel], 0, 255)
        neighbors.append(neighbor)
    
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
    
    max_perturbation = 255 * epsilon
    no_improvement_count = 0
    max_no_improvement = 50  # Stop if no improvement for 50 iterations
    
    print(f"Initial fitness: {current_fitness:.4f}")
    
    for iteration in range(iterations):
        # Generate neighbors from current image
        neighbors = mutate_seed(current_image, epsilon)
        
        # Enforce L∞ constraint relative to ORIGINAL seed
        valid_neighbors = []
        for neighbor in neighbors:
            # Check if neighbor satisfies L∞ constraint relative to initial_seed
            diff = np.abs(neighbor - initial_seed)
            if np.max(diff) <= max_perturbation:
                valid_neighbors.append(neighbor)
            else:
                # Clip to satisfy constraint
                neighbor_clipped = np.clip(
                    neighbor,
                    initial_seed - max_perturbation,
                    initial_seed + max_perturbation
                )
                neighbor_clipped = np.clip(neighbor_clipped, 0, 255)
                valid_neighbors.append(neighbor_clipped)
        
        # Add current image to candidates (elitism)
        candidates = [current_image] + valid_neighbors
        
        # Select best candidate
        best_image, best_fitness = select_best(candidates, model, target_label)
        
        # Check if fitness improved
        if best_fitness < current_fitness:
            current_image = best_image.copy()
            current_fitness = best_fitness
            no_improvement_count = 0
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: fitness = {current_fitness:.4f}")
            
            # Check if target is confidently reached (fitness > 0 means target is predicted)
            if current_fitness > 0.9:  # High confidence in target class
                print(f"Target reached with confidence {current_fitness:.4f} at iteration {iteration}")
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