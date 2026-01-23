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
from typing import Dict, Any, List

import numpy as np
from envs.highway_env_utils import run_episode, record_video_episode
from search.base_search import ScenarioSearch


# ============================================================
# 1) OBJECTIVES FROM TIME SERIES
# ============================================================

def is_not_directly_adjacent(ego_pos, other_pos, ego_length):
    # directly adjacent = other is on the neighboring lane
    # and its center is within the ego vehicle's length
    adjacent = abs(other_pos[1] - ego_pos[1]) == 1 and ego_pos[0] - ego_length < other_pos[0] < ego_pos[0] + ego_length
    return not adjacent


def compute_objectives_from_time_series(time_series: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute your objective values from the recorded time-series.

    The time_series is a list of frames. Each frame typically contains:
      - frame["crashed"]: bool
      - frame["ego"]: dict or None, e.g. {"pos":[x,y], "lane_id":..., "length":..., "width":...}
      - frame["others"]: list of dicts with positions, lane_id, etc.

    Minimum requirements (suggested):
      - crash_count: 1 if any collision happened, else 0
      - min_distance: minimum distance between ego and any other vehicle over time (float)

    Return a dictionary, e.g.:
        {
          "crash_count": 0 or 1,
          "min_distance": float
        }

    NOTE: If you want, you can add more objectives (lane-specific distances, time-to-crash, etc.)
    but keep the keys above at least.
    """
    # TODO: improve fitness function
    # extra score for more close encounters
    # want to make function smoother, so give more points for erratic-like behaviour
    # lane switching
    # average change in speed
    # avg distance to closest vehicles (< 2 car lengths)
    # number of cars it passes, get passed by

    crash_count = 0
    min_distance = float('inf')

    prev_lane_id = None
    lane_switches = 0

    prev_speed = None
    speed_changes = []

    prev_cars_behind = 0
    cars_passed_count = 0

    for frame in time_series:
        if frame.get("crashed", False):
            crash_count = 1
            min_distance = 0.0
            break

        ego = frame.get("ego")
        others = frame.get("others", [])

        if ego is not None and ego["pos"] is not None:
            ego_pos = np.array(ego["pos"])
            ego_length = ego.get("length", 4.5)
            ego_width = ego.get("width", 1.8)

            for other in others:
                other_pos = np.array(other["pos"])
                other_length = other.get("length", 4.5)
                other_width = other.get("width", 1.8)

                # Compute Euclidean distance between centers
                # ignore others that are directly adjacent to ego vehicle
                # while the ego vehicle is moving in a straight line
                if ego['heading'] != 0 or is_not_directly_adjacent(ego_pos, other_pos, ego_length):
                    center_distance = float(np.linalg.norm(ego_pos - other_pos))
                else:
                    center_distance = float('inf')

                min_distance = min(min_distance, center_distance)

            # lane switching
            if prev_lane_id is not None and prev_lane_id != ego["lane_id"]:
                lane_switches += 1
            prev_lane_id = ego["lane_id"]

            # speed changes
            if prev_speed is not None:
                delta_speed = abs(ego["speed"] - prev_speed)
                speed_changes.append(delta_speed)
            prev_speed = ego["speed"]

            # TODO: overtaking

    if min_distance == float('inf'):
        min_distance = 1000.0  # Default large value if no others

    # erratic speed
    # higher total speed change => more erratic, braking, accelerations
    total_speed_change = sum(speed_changes)

    return {
        "crash_count": crash_count,
        "min_distance": min_distance,
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

    Requirement:
    - Any crashing scenario must be strictly better than any non-crashing scenario.

    Examples:
    - If crash_count==1: fitness = -1 (best)
    - Else: fitness = min_distance (smaller is better)

    You can design a more refined scalarization if desired.
    """
    crash_count = objectives.get("crash_count", 0)
    min_distance = objectives.get("min_distance", float('inf'))
    lane_switches = objectives.get("lane_switches", 0)
    speed_change = objectives.get("speed_change", 0)

    if crash_count > 0:
        # Crashing scenario is best (we want to maximize fitness)
        return math.inf
    else:
        return w_distance / min_distance + w_lane_switch * lane_switches + w_speed_change * speed_change


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

    Inputs:
      - cfg: current scenario dict (e.g., vehicles_count, initial_spacing, ego_spacing, initial_lane_id)
      - param_spec: search space bounds, types (int/float), min/max
      - rng: random generator

    Requirements:
      - Do NOT modify cfg in-place (return a copy).
      - Keep mutated values within [min, max] from param_spec.
      - If you mutate lanes_count, keep initial_lane_id valid (0..lanes_count-1).

    Students can implement:
      - single-parameter mutation (recommended baseline)
      - multiple-parameter mutation
      - adaptive step sizes, etc.
    """
    # Create a copy to avoid modifying the original
    new_cfg = copy.deepcopy(cfg)

    # Select a random parameter to mutate
    mutable_params = [k for k in param_spec.keys() if k in new_cfg]

    if not mutable_params:
        return new_cfg

    param_to_mutate = rng.choice(mutable_params)
    spec = param_spec[param_to_mutate]
    param_type = spec.get("type", "float")
    min_val = spec.get("min", 0)
    max_val = spec.get("max", 1)

    if param_type == "int":
        # Mutate by adding/subtracting a small integer step
        step = max(1, (max_val - min_val) // 10)
        if rng.random() < 0.5:
            new_val = new_cfg[param_to_mutate] + step
        else:
            new_val = new_cfg[param_to_mutate] - step
        new_val = int(np.clip(new_val, min_val, max_val))
        new_cfg[param_to_mutate] = new_val

    elif param_type == "float":
        # Mutate by adding/subtracting a small float step
        step = (max_val - min_val) * 0.1
        if rng.random() < 0.5:
            new_val = new_cfg[param_to_mutate] + step
        else:
            new_val = new_cfg[param_to_mutate] - step
        new_val = float(np.clip(new_val, min_val, max_val))
        new_cfg[param_to_mutate] = new_val

    # Ensure initial_lane_id is valid if lanes_count was mutated
    if "lanes_count" in new_cfg and "initial_lane_id" in new_cfg:
        lanes = new_cfg["lanes_count"]
        new_cfg["initial_lane_id"] = int(np.clip(new_cfg["initial_lane_id"], 0, lanes - 1))

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
) -> Dict[str, Any]:
    """
    Hill climbing loop.

    You should:
      1) Start from an initial scenario (base_cfg or random sample).
      2) Evaluate it by running:
            crashed, ts = run_episode(env_id, cfg, policy, defaults, seed_base)
         Then compute objectives + fitness.
      3) For each iteration:
            - Generate neighbors_per_iter neighbors using mutate_config
            - Evaluate each neighbor
            - Select the best neighbor
            - Accept it if it improves fitness (or implement another acceptance rule)
            - Optionally stop early if a crash is found
      4) Return the best scenario found and enough info to reproduce.

    Return dict MUST contain at least:
        {
          "best_cfg": Dict[str, Any],
          "best_objectives": Dict[str, Any],
          "best_fitness": float,
          "best_seed_base": int,
          "history": List[float]
        }

    Optional but useful:
        - "best_time_series": ts
        - "evaluations": int
    """
    rng = np.random.default_rng(seed)

    # Initialize from base_cfg
    current_cfg = dict(base_cfg)

    # Evaluate initial solution
    seed_base = int(rng.integers(1e9))
    crashed, ts = run_episode(env_id, current_cfg, policy, defaults, seed_base)
    obj = compute_objectives_from_time_series(ts)
    cur_fit = compute_fitness(obj)

    best_cfg = copy.deepcopy(current_cfg)
    best_obj = dict(obj)
    best_fit = float(cur_fit)
    best_seed_base = seed_base
    best_time_series = ts

    history = [best_fit]
    evaluations = 1

    # Hill climbing main loop
    for iteration in range(iterations):
        # Early stopping if we found a crash
        if best_fit is math.inf:  # Crash found
            print(f" Crash found at iteration {iteration}!")
            break

        # Generate and evaluate neighbors
        best_neighbor_cfg = None
        best_neighbor_fit = float('inf')
        best_neighbor_obj = None
        best_neighbor_seed = None
        best_neighbor_ts = None

        for _ in range(neighbors_per_iter):
            # Generate neighbor
            neighbor_cfg = mutate_config(current_cfg, param_spec, rng)
            seed_base_neighbor = int(rng.integers(1e9))

            # Evaluate neighbor
            crashed_neighbor, ts_neighbor = run_episode(env_id, neighbor_cfg, policy, defaults, seed_base_neighbor)
            obj_neighbor = compute_objectives_from_time_series(ts_neighbor)
            fit_neighbor = compute_fitness(obj_neighbor)

            evaluations += 1

            # Track best neighbor
            if fit_neighbor > best_neighbor_fit:
                best_neighbor_fit = fit_neighbor
                best_neighbor_cfg = copy.deepcopy(neighbor_cfg)
                best_neighbor_obj = dict(obj_neighbor)
                best_neighbor_seed = seed_base_neighbor
                best_neighbor_ts = ts_neighbor

        # Accept best neighbor if it improves current solution
        if best_neighbor_fit > cur_fit:
            current_cfg = best_neighbor_cfg
            cur_fit = best_neighbor_fit
            obj = best_neighbor_obj
            print(f"Iteration {iteration}: Improved fitness from {history[-1]:.4f} to {cur_fit:.4f}")

            # Update global best if neighbor is better than global best
            if best_neighbor_fit > best_fit:
                best_cfg = copy.deepcopy(best_neighbor_cfg)
                best_fit = best_neighbor_fit
                best_obj = dict(best_neighbor_obj)
                best_seed_base = best_neighbor_seed
                best_time_series = best_neighbor_ts

        else:
            print(f"Iteration {iteration}: No improvement (best fitness: {best_fit:.4f})")

        history.append(best_fit)

    return {
        "best_cfg": best_cfg,
        "best_objectives": best_obj,
        "best_fitness": best_fit,
        "best_seed_base": best_seed_base,
        "best_time_series": best_time_series,
        "history": history,
        "evaluations": evaluations,
    }


# ============================================================
# 4) HILLCLIMBING CLASS (for integration with framework)
# ============================================================

class HillClimbing(ScenarioSearch):
    """Hill Climbing search class that integrates with the project framework."""

    def run_search(self, n_scenarios=50, n_eval=1, seed=42, iterations=100, neighbors_per_iter=10):
        """
        Run hill climbing search.

        Parameters:
            n_scenarios: number of independent HC runs (not used in single HC)
            n_eval: number of evaluations per scenario (not used in HC loop)
            seed: random seed
            iterations: HC iterations
            neighbors_per_iter: number of neighbors to evaluate per iteration

        Returns:
            crash_log: list of crashes found
        """
        print(f"Running Hill Climbing Search...")

        # Run a single hill climbing session
        result = hill_climb(
            self.env_id,
            self.base_cfg,
            self.param_spec,
            self.policy,
            self.defaults,
            seed=seed,
            iterations=iterations,
            neighbors_per_iter=neighbors_per_iter,
        )

        crash_log = []

        # Check if we found a crash
        if result["best_fitness"] < 0:  # Crash found
            cfg = result["best_cfg"]
            seed_found = result["best_seed_base"]
            print(f" Collision found! Config: {cfg}, Seed: {seed_found}")
            crash_log.append({"cfg": copy.deepcopy(cfg), "seed": seed_found})

            # Record video of the crash
            record_video_episode(self.env_id, cfg, self.policy, self.defaults, seed_found, out_dir="videos")
        else:
            print(f"  No crash found. Best fitness: {result['best_fitness']:.4f}")

        return crash_log