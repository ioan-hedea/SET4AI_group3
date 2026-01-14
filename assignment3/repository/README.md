# Testing Reinforcement Learning Agents in HighwayEnv

This project provides a modular framework for **testing pre-trained reinforcement learning (RL) agents** on the [Highway-Env](https://github.com/Farama-Foundation/HighwayEnv) simulation environment.  
The main goal is to **discover failure-inducing driving scenarios** (e.g., collisions) using different search algorithms.

It serves as the foundation for lab assignments where students will:
1. Understand and modify a reinforcement-learning test environment.
2. Implement and compare different **search-based scenario generation algorithms** (e.g., Random Search, Hill Climber).
3. Analyze the agent’s robustness by reproducing and recording crash scenarios.

---

## Project Overview

A pre-trained RL agent (trained using **PPO**) drives autonomously in a simulated highway.  
The task is to **automatically generate test scenarios** (traffic density, spacing, initial lane, etc.) that reveal weaknesses in the policy.

### Objective example
> *Find scenario parameters that make the ego vehicle crash.*  
If no crash is found, the goal becomes finding scenarios that **minimize the distance** between the ego vehicle and other vehicles.

Scenario search changes parameters such as:
- `vehicles_count`: number of cars on the road  
- `lanes_count`: number of lanes  
- `initial_spacing`: distance between vehicles  
- `ego_spacing`: spacing between ego and surrounding traffic  
- `initial_lane_id`: initial lane position of the ego vehicle  

---

## Project Structure

```
testing-rl/
│
├── main.py                          # Entry point – runs a chosen search algorithm
│
├── config/
│   ├── __init__.py
│   └── search_space.py              # Defines search-space & base configuration
│
├── envs/
│   ├── __init__.py
│   └── highway_env_utils.py         # Environment setup, resets, and episode evaluation
│
├── policies/
│   ├── __init__.py
│   └── pretrained_policy.py         # Loads a trained model and defines policy()
│
├── search/
│   ├── __init__.py
│   ├── base_search.py               # Abstract class ScenarioSearch
│   ├── random_search.py             # Implements RandomSearch
│   └── hill_climbing.py             # YOUR TASK (template)
│
├── agents/                          # Folder containing trained RL policies
│
├── videos/                          # Recorded crash videos
│
└── requirements.txt                 # Python dependencies
```

---

## What We Provide

1. **Pretrained RL agent**  
   A PPO agent trained on `highway-fast-v0` with domain randomization.

2. **Scenario execution utilities** (`envs/highway_env_utils.py`)  
   - Scenario application / reset
   - Episode execution
   - Collision detection
   - Time-series logging (ego + other vehicles over time)
   - Video recording utilities

3. **Random Search baseline** (`search/random_search.py`)  
   A simple baseline that samples scenarios uniformly at random.

4. **Hill Climber template** (`search/hill_climbing.py`)  
   A template with TODO sections for students.


**Do not change public function signatures**, as they are used for grading.

### What you must implement (for grading)
- You are expected to implement your solution **only by editing**: `search/hill_climbing.py`.
- You may add helper code **only inside** `search/` (e.g., `search/utils_hc.py`) and import it from `search/hill_climbing.py`.
- Do **not** change the interface of any provided classes/functions (especially `ScenarioSearch` and `run_search()`), otherwise our grading scripts may fail.

## Installation

### 1. Clone and create a virtual environment
```bash
git clone https://github.com/<your-repo>/testing-rl.git
cd testing-rl

python -m venv .venv
source .venv/bin/activate    # (Mac/Linux)
# OR
.venv\\Scripts\\activate       # (Windows)
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

The key dependencies include:
- `gymnasium>=0.29.0`
- `highway-env>=1.8.2`
- `torch>=2.2.0`
- `stable-baselines3==2.3.0`
- `tqdm`, `numpy`

---

## Pre-trained Agent

The trained policy (PPO) files are stored in:
```
agents/
└── model.zip
└── vec_normalized.pkl
```

The file `pretrained_policy.py` handles loading the pre-trained PPO agent

---

## Running the Project

Run the random search baseline from **main.py**:

```bash
python main.py
```

This will:
- Load the pre-trained PPO policy.
- Sample multiple random highway scenarios.
- Run each configuration for several episodes.
- Record any crashes to `videos/` as MP4 files.

---

## Time-series data (important for fitness/objectives)

`run_episode(...)` (defined in `envs/highway_env_utils.py`) returns:
- `crashed` (bool): whether a collision occurred at any time during the episode
- `time_series` (list[dict]): a list of frames (one per time step)

Each frame is a dictionary with (at least) the following keys:
- `frame["t"]`: time step index
- `frame["crashed"]`: whether a crash has occurred at that step
- `frame["ego"]`: ego vehicle state, or `None` if unavailable
- `frame["others"]`: list of other vehicles (possibly empty)

A typical frame looks like:
```python
{
  "t": 12,
  "crashed": False,
  "ego": {
    "pos": [x, y],
    "speed": v,
    "heading": h,
    "length": L,
    "width": W,
    "lane_id": k,
  },
  "others": [
    {"pos": [x, y], "length": L, "width": W, "lane_id": k},
    ...
  ]
}
```

### Hints for computing objectives (fitness)
The exact objective definition is intentionally left open and must be explained and justified by the student.

Hints for computing objectives (fitness)
* A natural goal in this assignment is to identify failure-inducing scenarios, such as those leading to a collision.
* When no collision is observed, the time-series data produced during execution can be used to quantify how unsafe a scenario is, based solely on observable behavior (e.g., vehicle positions and interactions over time).
* Such objective values may capture notions of proximity, risk, or progress toward failure without requiring access to internal agent information.

The exact definition of the objective (fitness) function is intentionally left open and must be clearly described and justified in the report.


### Useful utilities (recommended)
See `envs/highway_env_utils.py` for helper routines you can reuse, including:
- `make_env(...)` and `apply_scenario_config(...)` for consistent environment setup
- `run_episode(...)` to execute a scenario and collect `time_series`
- `record_video_episode(...)` to replay a scenario and save an MP4 video under `videos/`

## What students must implement (Assignment 3)

For grading, you are expected to complete **only** the Hill Climbing implementation provided in:

- `search/hill_climbing.py`

You may add helper functions/modules **inside the `search/` package** (e.g., `search/utils_hc.py`) if you want, but:
- **Do not change public function signatures** in the provided files.
- **Do not rename/move files** that are imported by `main.py`.
- Your code must remain runnable via `python main.py`.

We provide **RandomSearch** as a baseline in `search/random_search.py`. Your Hill Climber must follow the same general workflow:
1. Generate or mutate a scenario configuration (based on the parameter space in `config/search_space.py`).
2. Evaluate a scenario by executing it with the provided PPO policy (via `envs/highway_env_utils.py`).
3. Use the observed outcome (collision + time-series signals) to guide search.

---

## Baseline: Random Search (provided)

The Random Search baseline is already implemented. You can run it from `main.py` (see the code snippet below) to understand:
- how scenarios are sampled,
- how episodes are executed,
- how crashes are detected,
- and how videos are recorded.

```python
from search.random_search import RandomSearch
from config.search_space import param_spec, base_cfg
from policies.pretrained_policy import load_pretrained_policy
from envs.highway_env_utils import make_env

def main():
    env_id = "highway-fast-v0"
    policy = load_pretrained_policy("agents/model")
    env, defaults = make_env(env_id)

    search = RandomSearch(env_id, base_cfg, param_spec, policy, defaults)
    crashes = search.run_search(n_scenarios=30, n_eval=3, seed=11)

    print(f"✅ Found {len(crashes)} crashes.")

if __name__ == "__main__":
    main()
 ```

## License
This project is provided for educational use in the *Software Engineering and Testing for AI Systems* course (DSAIT4015).  
Based on [Highway-Env](https://github.com/Farama-Foundation/HighwayEnv)  
and [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).

---

**Author:** Annibale Panichella  
**Institution:** TU Delft – SERG / CISELab