# envs/highway_env_utils.py
import copy
import os
from typing import Tuple, Dict, Any

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np


# ============================================================================
#  TRAINING CONFIG USED TO NORMALIZE ENVIRONMENT FOR BOTH TRAINING & TESTING
# ============================================================================
TRAINING_ENV_CONFIG: Dict[str, Any] = {
    "duration": 40,
    "lanes_count": 4,
    "collision_reward": -1.0,
    "high_speed_reward": 0.6,
    "right_lane_reward": 0.0,
    "headway_reward": 0.0,
    "reward_speed_range": [20, 40],
    # other parameters remain the default from env.default_config()
}


# ============================================================================
#  ENV CREATION
# ============================================================================

def make_env(env_id: str = "highway-fast-v0",
             render_mode: str | None = None) -> Tuple[gym.Env, Dict[str, Any]]:
    """
    Create a highway-fast-v0 environment whose config is aligned with PPO training.
    Returns:
        env        : new environment instance
        defaults   : copy of env.unwrapped.config after applying TRAINING_ENV_CONFIG
    """
    env = gym.make(env_id, render_mode=render_mode)

    # Apply PPO training overrides
    env.unwrapped.config.update(TRAINING_ENV_CONFIG)

    # Copy defaults (base for scenario configs)
    defaults = dict(env.unwrapped.config)
    return env, defaults


# ============================================================================
#  CONFIG CANONICALIZATION
# ============================================================================

def _canonical_cfg(env: gym.Env,
                   scenario_cfg: Dict[str, Any],
                   defaults: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge scenario_cfg with training defaults and clamp values.
    """
    cfg = dict(defaults)
    user = dict(scenario_cfg)

    # Clamp search parameters
    user["vehicles_count"] = int(np.clip(user.get("vehicles_count", 20), 5, 120))
    user["lanes_count"] = int(np.clip(user.get("lanes_count", 3), 2, 10))
    user["initial_spacing"] = float(np.clip(user.get("initial_spacing", 2.0), 0.5, 5.0))
    user["ego_spacing"] = float(np.clip(user.get("ego_spacing", 1.0), 1.0, 4.0))

    # Ensure lane_id is valid
    lanes = user.get("lanes_count", cfg.get("lanes_count"))
    user["initial_lane_id"] = int(np.clip(user.get("initial_lane_id", 1), 0, lanes - 1))

    # Merge into config
    cfg.update(user)
    return cfg


def apply_scenario_config(env: gym.Env,
                          cfg: Dict[str, Any],
                          defaults: Dict[str, Any],
                          seed: int | None = None):
    """
    Reset env with a scenario config that starts from training defaults.
    """
    cfg2 = _canonical_cfg(env, cfg, defaults)
    try:
        return env.reset(options={"config": cfg2}, seed=seed)
    except TypeError:
        env.unwrapped.config.update(cfg2)
        return env.reset(seed=seed)


# ============================================================================
#  VEHICLE EXTRACTION
# ============================================================================

def _ego_and_others(env):
    """
    Return:
        ego    : the ego vehicle (ControlledVehicle / MDPVehicle)
        others : list of other vehicles

    Why needed?
    -----------
    Different vehicle classes expose different attributes:
        - MDPVehicle has no .length but has LENGTH
        - Some vehicle types do not expose lane_index until spawned
    """
    ego = getattr(env.unwrapped, "vehicle", None)

    others = []
    road = getattr(env.unwrapped, "road", None)
    if road is not None and hasattr(road, "vehicles"):
        for v in road.vehicles:
            if v is not ego:
                others.append(v)

    return ego, others


# ============================================================================
#  EPISODE EXECUTION + TIME SERIES LOGGING
# ============================================================================

def run_episode(env_id: str,
                cfg: Dict[str, Any],
                policy,
                defaults: Dict[str, Any],
                seed: int,
                render: bool = False):
    """
    Run ONE driving episode for a given scenario configuration and log a time series.

    Returns
    -------
    crashed : bool
        True iff the environment reports a crash at any step.
    time_series : List[Dict[str, Any]]
        A list of frames, one per time step. Each frame has:

        frame = {
            "t": int,                 # time step index starting at 0
            "crashed": bool,          # whether a crash was reported at that step
            "ego": {
                "pos": [x, y],        # position in world coordinates
                "speed": float,
                "heading": float,     # radians
                "length": float,      # meters (or env units)
                "width": float,
                "lane_id": int | None # lane index (0..lanes_count-1) if available
            } | None,
            "others": [
                {
                    "pos": [x, y],
                    "length": float,
                    "width": float,
                    "lane_id": int | None
                },
                ...
            ]
        }

    Notes for objective/fitness design (Assignment 3)
    -------------------------------------------------
    Typical black-box objectives computed from `time_series` include:
      1) Crash indicator:
         - crash_count = 1 if any frame["crashed"] is True else 0
      2) Minimum ego–other distance over time (smaller is "more dangerous"):
         - For each frame, compute distance between ego and every other vehicle.
           Then take the minimum across all frames.
         - You can use simple Euclidean center distance:
             d = sqrt((x2-x1)^2 + (y2-y1)^2)
           or a more realistic rectangle-clearance distance using length/width.
      3) Lane-specific proximity:
         - Track min distance to vehicles in same lane / left lane / right lane.
         - Use ego["lane_id"] and v["lane_id"] when available.

    Important: objectives must rely ONLY on observable behaviour during execution
    (time series states/outcomes), not on internal agent/model information.
    """
    env, _ = make_env(env_id, render_mode="rgb_array" if render else None)

    obs, info = apply_scenario_config(env, cfg, defaults, seed=seed)

    crashed = False
    time_series = []
    t = 0

    while True:
        ego, others = _ego_and_others(env)

        frame = {"t": t}

        # -----------------------------
        #  EGO VEHICLE INFO
        # -----------------------------
        if ego is not None:
            # Robust length/width extraction
            ego_length = float(getattr(ego, "length", getattr(ego, "LENGTH", 4.5)))
            ego_width = float(getattr(ego, "width", getattr(ego, "WIDTH", 1.8)))

            frame["ego"] = {
                "pos": list(map(float, ego.position)),
                "speed": float(getattr(ego, "speed", 0.0)),
                "heading": float(getattr(ego, "heading", 0.0)),
                "length": ego_length,
                "width": ego_width,
                "lane_id": getattr(ego, "lane_index", [None, None, None])[2],
            }
        else:
            frame["ego"] = None

        # -----------------------------
        #  OTHER VEHICLES INFO
        # -----------------------------
        frame["others"] = []
        for v in others:
            v_length = float(getattr(v, "length", getattr(v, "LENGTH", 4.5)))
            v_width = float(getattr(v, "width", getattr(v, "WIDTH", 1.8)))

            frame["others"].append({
                "pos": list(map(float, v.position)),
                "length": v_length,
                "width": v_width,
                "lane_id": getattr(v, "lane_index", [None, None, None])[2],
            })

        frame["crashed"] = info.get("crashed", False)
        time_series.append(frame)

        # -----------------------------
        #  STEP THE ENVIRONMENT
        # -----------------------------
        action = policy(obs, info)
        obs, reward, terminated, truncated, info = env.step(int(action))

        if info.get("crashed", False):
            crashed = True

        if terminated or truncated:
            break

        t += 1

    env.close()
    return crashed, time_series


# ============================================================================
#  VIDEO RECORDING
# ============================================================================

def record_video_episode(env_id: str,
                         cfg: Dict[str, Any],
                         agent_policy,
                         defaults: Dict[str, Any],
                         seed: int,
                         out_dir: str = "videos"):
    """
    Replay a scenario and save a .mp4 video inside a dedicated folder.
    """
    os.makedirs(out_dir, exist_ok=True)
    video_folder = os.path.join(out_dir, f"scenario_seed_{seed}")
    os.makedirs(video_folder, exist_ok=True)

    env_v, _ = make_env(env_id, render_mode="rgb_array")

    env_v = gym.wrappers.RecordVideo(
        env_v,
        video_folder=video_folder,
        episode_trigger=lambda e: True,
        name_prefix=f"seed_{seed}",
    )

    obs, info = apply_scenario_config(env_v, cfg, defaults, seed=seed)

    crashed = False
    while True:
        action = agent_policy(obs, info)
        obs, reward, terminated, truncated, info = env_v.step(int(action))

        if info.get("crashed", False):
            crashed = True
            break
        if terminated or truncated:
            break

    env_v.close()
    return crashed, video_folder