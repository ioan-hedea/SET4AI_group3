"""
Assignment 3 — Scenario-Based Testing of an RL Agent (Hill Climbing)

You MUST implement:
    - compute_objectives_from_time_series
    - compute_fitness
    - mutate_config
    - hill_climb

DO NOT change function signatures.
You MAY add helper functions.

Goal
----
Find a scenario (environment configuration) that triggers a collision.
If you cannot trigger a collision, minimize the minimum distance between the ego
vehicle and any other vehicle across the episode.

Black-box requirement
---------------------
Your evaluation must rely only on observable behavior during execution:
- crashed flag from the environment
- time-series data returned by run_episode (positions, lane_id, etc.)
No internal policy/model details beyond calling policy(obs, info).
"""

import copy
import math
import time
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from tqdm import trange

# Project imports
from envs.highway_env_utils import run_episode, record_video_episode
from search.base_search import ScenarioSearch


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _is_safe_adjacency(ego_pos: np.ndarray, other_pos: np.ndarray, ego_length: float) -> bool:
    """
    Determines if a vehicle is directly adjacent (in a neighboring lane) 
    and within the longitudinal safety shadow of the ego vehicle.
    Returns True if the 'other' vehicle is strictly adjacent (ignored for distance calc).
    """
    # Check lateral distance (approx 1 lane width)
    is_neighbor_lane = abs(other_pos[1] - ego_pos[1]) == 1
    # Check longitudinal overlap
    is_longitudinal_overlap = (ego_pos[0] - ego_length) < other_pos[0] < (ego_pos[0] + ego_length)
    
    return is_neighbor_lane and is_longitudinal_overlap


def _evaluate_scenario(
    env_id: str, 
    cfg: Dict[str, Any], 
    policy, 
    defaults: Dict[str, Any], 
    seed: int
) -> Tuple[float, Dict[str, Any], bool, List[Dict[str, Any]]]:
    """
    Helper to run an episode and calculate fitness/objectives.
    Reduces code duplication in the search loop.
    """
    crashed, time_series = run_episode(env_id, cfg, policy, defaults, seed)
    objectives = compute_objectives_from_time_series(time_series)
    
    # Calculate raw fitness
    fitness = compute_fitness(objectives)
    
    # Cap fitness at 100.0 for stability/plotting, consistent with legacy logic
    # If crashed, we treat it as max fitness (100.0)
    if crashed:
        fitness = 100.0
    else:
        fitness = min(fitness, 100.0)
        
    return fitness, objectives, crashed, time_series


# ============================================================
# 1) OBJECTIVES FROM TIME SERIES
# ============================================================

def compute_objectives_from_time_series(time_series: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute your objective values from the recorded time-series.
    """
    crash_count = 0
    min_distance = float('inf')
    
    lane_switches = 0
    total_speed_change = 0.0
    
    # State trackers
    prev_lane_id = None
    prev_speed = None

    for frame in time_series:
        # 1. Check Crash
        if frame.get("crashed", False):
            crash_count = 1
            min_distance = 0.0
            break # Stop processing if crashed

        ego = frame.get("ego")
        others = frame.get("others", [])

        if not ego or ego.get("pos") is None:
            continue

        ego_pos = np.array(ego["pos"])
        ego_length = ego.get("length", 5.0)
        ego_heading = ego.get("heading", 0.0)

        # 2. Update Min Distance
        for other in others:
            other_pos = np.array(other["pos"])
            
            # We ignore distance calculations if the ego is driving straight 
            # and the other car is directly alongside (passing/being passed).
            should_ignore = (ego_heading == 0) and _is_safe_adjacency(ego_pos, other_pos, ego_length)
            
            if not should_ignore:
                dist = np.linalg.norm(ego_pos - other_pos)
                if dist < min_distance:
                    min_distance = dist

        # 3. Track Lane Switches
        current_lane = ego.get("lane_id")
        if prev_lane_id is not None and current_lane != prev_lane_id:
            lane_switches += 1
        prev_lane_id = current_lane

        # 4. Track Speed Changes (erratic driving)
        current_speed = ego.get("speed", 0.0)
        if prev_speed is not None:
            total_speed_change += abs(current_speed - prev_speed)
        prev_speed = current_speed

    # Fallback if no valid distance was found (e.g. empty road)
    if min_distance == float('inf'):
        min_distance = 1000.0

    return {
        "crash_count": crash_count,
        "min_distance": float(min_distance),
        "lane_switches": lane_switches,
        "speed_change": total_speed_change,
    }


def compute_fitness(
        objectives: Dict[str, Any],
        w_distance: float = 50,
        w_lane_switch: float = 0.1,
        w_speed_change: float = 0.1
) -> float:
    """
    Convert objectives into ONE scalar fitness value to MAXIMIZE.
    """
    crash_count = objectives.get("crash_count", 0)
    min_dist = objectives.get("min_distance", 1e-6) # Avoid div by zero
    lane_switches = objectives.get("lane_switches", 0)
    speed_change = objectives.get("speed_change", 0)

    # Primary Goal: Crash
    if crash_count > 0:
        return math.inf

    # Secondary Goal: Minimize distance (Maximize 1/distance)
    # Penalize erratic behavior (lane switches, jerky speed) slightly
    fitness = (w_distance / min_dist) + \
              (w_lane_switch * lane_switches) + \
              (w_speed_change * speed_change)
              
    return fitness


# ============================================================
# 2) MUTATION / NEIGHBOR GENERATION
# ============================================================

def mutate_config(
    cfg: Dict[str, Any],
    param_spec: Dict[str, Any],
    rng: np.random.Generator
) -> Dict[str, Any]:
    """
    Generate ONE neighbor configuration by mutating the current scenario.
    """
    new_cfg = copy.deepcopy(cfg)
    
    # Identify which parameters exist in the current config
    valid_params = [k for k in param_spec.keys() if k in new_cfg]
    if not valid_params:
        return new_cfg

    # Determine how many parameters to mutate (Heuristic: half available, at least 1)
    num_to_mutate = max(1, len(valid_params) // 2)
    params_to_mutate = rng.choice(valid_params, size=num_to_mutate, replace=False)

    for param_key in params_to_mutate:
        spec = param_spec[param_key]
        p_type = spec.get("type", "float")
        p_min = spec.get("min", 0)
        p_max = spec.get("max", 1)
        
        current_val = new_cfg[param_key]
        
        # Mutation Logic: Add Gaussian noise
        # Sigma is set to cover half the range, allowing for large jumps
        sigma = (p_max - p_min) / 2.0
        noise = rng.normal(0, sigma)
        
        mutated_val = current_val + noise
        
        # Clip to bounds
        mutated_val = np.clip(mutated_val, p_min, p_max)

        # Cast to correct type
        if p_type == "int":
            new_cfg[param_key] = int(round(mutated_val))
        else:
            new_cfg[param_key] = float(mutated_val)

    # Constraint Maintenance: 
    # initial_lane_id must be valid within lanes_count
    if "lanes_count" in new_cfg and "initial_lane_id" in new_cfg:
        max_lane_index = new_cfg["lanes_count"] - 1
        new_cfg["initial_lane_id"] = int(np.clip(new_cfg["initial_lane_id"], 0, max_lane_index))

    return new_cfg


# ============================================================
# 3) HILL CLIMBING SEARCH
# ============================================================

def hill_climb(
    env_id: str,
    base_cfg: Dict[str, Any],
    param_spec: Dict[str, Any],
    policy,
    defaults: Dict[str, Any],
    seed: int = 0,
    iterations: int = 100,
    neighbors_per_iter: int = 10,
    patience: int = 3
) -> Dict[str, Any]:
    """
    Hill climbing loop.
    """
    rng = np.random.default_rng(seed)

    # 1. Initialization
    current_cfg = copy.deepcopy(base_cfg)
    episode_seed = int(rng.integers(0, 2**31 - 1))
    
    # Initial Evaluation
    cur_fit, cur_obj, _, cur_ts = _evaluate_scenario(env_id, current_cfg, policy, defaults, episode_seed)

    # Track Global Best
    best_cfg = copy.deepcopy(current_cfg)
    best_obj = cur_obj
    best_fit = cur_fit
    best_ts = cur_ts
    
    history = [best_fit]
    eval_count = 1
    no_improvement_counter = 0

    # 2. Search Loop
    pbar = trange(iterations, desc="Hill Climbing")
    for _ in pbar:
        # Stop immediately if we found a crash (fitness >= 100)
        if best_fit >= 100.0:
            pbar.set_description("Crash Found!")
            break

        # Generate and Evaluate Neighbors
        iter_best_fit = -math.inf
        iter_best_cfg = None
        iter_best_obj = None
        iter_best_ts = None

        for _ in range(neighbors_per_iter):
            # Create Neighbor
            neighbor_cfg = mutate_config(current_cfg, param_spec, rng)
            
            # Evaluate Neighbor
            fit, obj, crashed, ts = _evaluate_scenario(env_id, neighbor_cfg, policy, defaults, episode_seed)
            eval_count += 1

            # Keep track of the best neighbor in this batch
            if fit > iter_best_fit:
                iter_best_fit = fit
                iter_best_cfg = neighbor_cfg
                iter_best_obj = obj
                iter_best_ts = ts
                
            # Micro-optimization: if we found a crash in neighbors, stop generating neighbors
            if crashed:
                break

        # 3. Selection & Updates
        if iter_best_fit > cur_fit:
            # Improvement found: Move current state
            current_cfg = iter_best_cfg
            cur_fit = iter_best_fit
            cur_obj = iter_best_obj
            # Note: We don't update episode_seed
            
            no_improvement_counter = 0
            
            # Update Global Best
            if cur_fit > best_fit:
                best_fit = cur_fit
                best_cfg = copy.deepcopy(current_cfg)
                best_obj = cur_obj
                best_ts = iter_best_ts

        else:
            # No improvement
            no_improvement_counter += 1
            
            # Restart Mechanism
            if no_improvement_counter >= patience:
                # Restart from a random configuration
                temp_search = ScenarioSearch(env_id, base_cfg, param_spec, policy, defaults)
                current_cfg = temp_search.sample_random_config(rng)
                
                # Re-evaluate new start point
                cur_fit, cur_obj, crashed, cur_ts = _evaluate_scenario(env_id, current_cfg, policy, defaults, episode_seed)
                eval_count += 1
                
                no_improvement_counter = 0
                
                # If random restart is lucky and better than global best
                if cur_fit > best_fit:
                    best_fit = cur_fit
                    best_cfg = copy.deepcopy(current_cfg)
                    best_obj = cur_obj
                    best_ts = cur_ts

        history.append(best_fit)
        pbar.set_postfix({"Best Fit": f"{best_fit:.2f}", "Curr Fit": f"{cur_fit:.2f}"})

    return {
        "best_cfg": best_cfg,
        "best_objectives": best_obj,
        "best_fitness": best_fit,
        "best_seed_base": episode_seed,
        "best_time_series": best_ts,
        "history": history,
        "evaluations": eval_count,
    }


# ============================================================
# 4) HILLCLIMBING CLASS (Integration)
# ============================================================

class HillClimbing(ScenarioSearch):
    """Hill Climbing search class that integrates with the project framework."""

    def run_search(self, n_scenarios=50, n_eval=1, seed=42) -> Dict[str, Any]:
        """
        Run hill climbing search.
        
        Returns:
            Dictionary with crash logs, metrics, and timing info.
        """
        start_time = time.time()
        print(f"Running Hill Climbing Search (Iter={n_scenarios}, Neighbors={n_eval})...")

        rng = np.random.default_rng(seed)
        
        # Ensure base_cfg is populated
        if not any(k in self.base_cfg for k in self.param_spec):
             self.base_cfg.update(self.sample_random_config(rng))
        
        initial_cfg = copy.deepcopy(self.base_cfg)
        hc_seed = int(rng.integers(0, 2**31 - 1))

        # Execute Search
        result = hill_climb(
            env_id=self.env_id,
            base_cfg=self.base_cfg,
            param_spec=self.param_spec,
            policy=self.policy,
            defaults=self.defaults,
            seed=hc_seed,
            iterations=n_scenarios,
            neighbors_per_iter=n_eval,
        )

        # Process Results
        crash_log = []
        first_crash_evals = None
        
        if result["best_fitness"] >= 100.0:
            print(f"Collision found! Fitness: {result['best_fitness']}")
            
            crash_entry = {
                "cfg": result["best_cfg"], 
                "seed": result["best_seed_base"]
            }
            crash_log.append(crash_entry)
            
            # Since we stop exactly when we find a crash, total evaluations = first_crash_evals
            first_crash_evals = result["evaluations"]

            # Save Replay
            record_video_episode(
                self.env_id, 
                result["best_cfg"], 
                self.policy, 
                self.defaults, 
                result["best_seed_base"], 
                out_dir="videos"
            )
        else:
            print(f"No crash found. Best Fitness: {result['best_fitness']:.4f}")

        return {
            "crash_log": crash_log,
            "evaluations": result["evaluations"],
            "total_time": time.time() - start_time,
            "initial_cfg": initial_cfg,
            "first_crash_evals": first_crash_evals
        }