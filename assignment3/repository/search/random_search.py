from tqdm import trange
import numpy as np
import copy
from envs.highway_env_utils import record_video_episode, run_episode


class RandomSearch:
    def __init__(self, env_id, base_cfg, param_spec, policy, defaults):
        self.env_id = env_id
        self.base_cfg = base_cfg
        self.param_spec = param_spec
        self.policy = policy
        self.defaults = defaults

    def run_search(self, n_scenarios=50, n_eval=1, seed=42):
        print(f"Running Random Search for {n_scenarios} scenarios...")
        rng = np.random.default_rng(seed)
        crash_log = []

        for i in trange(n_scenarios, desc="Random search"):
            cfg = self.sample_random_config(rng)
            for j in range(n_eval):
                s = int(rng.integers(1e9))
                crashed, ts = run_episode(self.env_id, cfg, self.policy, self.defaults, s)

                if crashed:
                    print(f"💥 Collision: scenario {i}, seed={s}")
                    crash_log.append({"cfg": copy.deepcopy(cfg), "seed": s})
                    record_video_episode(self.env_id, cfg, self.policy, self.defaults, s, out_dir="videos")
                    break
                else:
                    print(f"No Crash: scenario {i}, seed={s}")
                    # crash_log.append({"cfg": copy.deepcopy(cfg), "seed": s})
                    # record_video_episode(self.env_id, cfg, self.policy, self.defaults, s, out_dir="videos")
        return crash_log

    def sample_random_config(self, rng):
        from search.base_search import ScenarioSearch
        return ScenarioSearch.sample_random_config(self, rng)