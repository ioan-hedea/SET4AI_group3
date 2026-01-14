import numpy as np
import copy

class ScenarioSearch:
    """Base class for scenario search algorithms."""
    def __init__(self, env_id, base_cfg, param_spec, policy, defaults):
        self.env_id = env_id
        self.base_cfg = base_cfg
        self.param_spec = param_spec
        self.policy = policy
        self.defaults = defaults

    def sample_random_config(self, rng):
        cfg = {}
        lanes = None
        if "lanes_count" in self.param_spec:
            s = self.param_spec["lanes_count"]
            lanes = int(rng.integers(s["min"], s["max"] + 1))
            cfg["lanes_count"] = lanes

        for k, s in self.param_spec.items():
            if k == "lanes_count":
                continue
            if k == "initial_lane_id":
                lanes = lanes or 3
                cfg[k] = int(rng.integers(0, lanes))
                continue
            if s["type"] == "int":
                cfg[k] = int(rng.integers(s["min"], s["max"] + 1))
            elif s["type"] == "float":
                cfg[k] = float(rng.uniform(s["min"], s["max"]))

        for k, v in self.base_cfg.items():
            if k not in cfg:
                cfg[k] = v
        return cfg

    def run_search(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement run_search()")