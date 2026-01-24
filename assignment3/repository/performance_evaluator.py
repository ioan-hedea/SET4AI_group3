import time
import numpy as np
import copy
import json
import os
from typing import List, Dict, Any, Type

from assignment3.repository.envs.highway_env_utils import make_env
from assignment3.repository.policies.pretrained_policy import load_pretrained_policy
from search.random_search import RandomSearch
from search.hill_climbing import HillClimbing
from config.search_space import param_spec, base_cfg

class PerformanceEvaluator:
    def __init__(self, env_id, base_cfg, param_spec, policy, defaults):
        self.env_id = env_id
        self.base_cfg = base_cfg
        self.param_spec = param_spec
        self.policy = policy
        self.defaults = defaults
        self.results = {}

    def run_evaluation(self, algo_class: Type, name: str, n_runs: int = 10, n_scenarios: int = 10, n_eval: int = 10):
        print(f"\n--- Evaluating {name} over {n_runs} runs ---")
        
        runs_data = []
        all_fails = []
        
        for i in range(n_runs):
            print(f"Run {i+1}/{n_runs}...")
            # Use a different seed for each run
            seed = 420 + i
            
            algo = algo_class(
                self.env_id, 
                copy.deepcopy(self.base_cfg), 
                self.param_spec, 
                self.policy, 
                self.defaults
            )
            
            result = algo.run_search(seed=seed, n_scenarios=n_scenarios, n_eval=n_eval)
            runs_data.append(result)
            
            for crash in result.get("crash_log", []):
                all_fails.append({
                    "run": i,
                    "cfg": crash["cfg"],
                    "seed": crash["seed"],
                    "initial_cfg": result.get("initial_cfg") # Only present for HC usually
                })

        # Calculate metrics
        evals_to_first_crash = [r["first_crash_evals"] for r in runs_data if r["first_crash_evals"] is not None]
        avg_evals_to_crash = np.mean(evals_to_first_crash) if evals_to_first_crash else float('inf')
        
        total_evals = sum(r["evaluations"] for r in runs_data)
        total_time = sum(r["total_time"] for r in runs_data)
        
        # Count distinct failures (by config)
        distinct_fails = []
        seen_configs = []
        for fail in all_fails:
            if fail["cfg"] not in seen_configs:
                distinct_fails.append(fail)
                seen_configs.append(fail["cfg"])
        
        metrics = {
            "avg_evals_to_first_crash": avg_evals_to_crash,
            "total_crashes_found": len(all_fails),
            "distinct_crashes_found": len(distinct_fails),
            "avg_time_per_search": total_time / n_runs,
            "avg_time_per_evaluation": total_time / total_evals if total_evals > 0 else 0,
            "success_rate": len(evals_to_first_crash) / n_runs
        }
        
        self.results[name] = {
            "metrics": metrics,
            "fails": all_fails,
            "distinct_fails": distinct_fails
        }
        
        self._print_metrics(name, metrics)
        self._save_results(name, n_scenarios, n_eval)
        
        return metrics

    def _print_metrics(self, name, metrics):
        print(f"\nResults for {name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    def _save_results(self, name, n_scenarios, n_eval):
        os.makedirs("performance_results", exist_ok=True)
        filename = f"performance_results/{name.lower().replace(' ', '_')}_{n_scenarios}_{n_eval}_results.json"
        with open(filename, "w") as f:
            # We need a custom encoder for numpy types or just convert them
            json.dump(self.results[name], f, indent=2, default=lambda x: float(x) if isinstance(x, np.float64) else x)
        print(f"Results saved to {filename}")

if __name__ == "__main__":
    env_id = "highway-fast-v0"
    policy = load_pretrained_policy("agents/model")
    env, defaults = make_env(env_id)
    
    evaluator = PerformanceEvaluator(env_id, base_cfg, param_spec, policy, defaults)
    
    evaluator.run_evaluation(HillClimbing, "Hill Climbing", n_runs=5, n_scenarios=20, n_eval=5)
    # evaluator.run_evaluation(RandomSearch, "Random Search", n_runs=10, n_scenarios=100)

