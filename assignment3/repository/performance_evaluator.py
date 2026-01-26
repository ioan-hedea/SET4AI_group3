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

    def run_evaluation(self, algo_class: Type, name: str, seed_base: int = 420, n_runs: int = 11, n_scenarios: int = 16, n_eval: int = 1, **kwargs):
        print(f"\n--- Evaluating {name} over {n_runs} runs ---")
        
        runs_data = []
        all_fails = []
        per_run_records = []
        
        for i in range(n_runs):
            print(f"Run {i+1}/{n_runs}...")
            # Use a different seed for each run
            seed = seed_base + i
            
            algo = algo_class(
                self.env_id, 
                copy.deepcopy(self.base_cfg), 
                self.param_spec, 
                self.policy, 
                self.defaults
            )
            
            result = algo.run_search(seed=seed, n_scenarios=n_scenarios, n_eval=n_eval, **kwargs)
            runs_data.append(result)
            
            # Record per-run stats
            run_record = {
                "run": i,
                "evaluations": result["evaluations"],
                "total_time": result["total_time"],
            }
            if "best_fitness" in result:
                run_record["best_fitness"] = result["best_fitness"]
            per_run_records.append(run_record)
            
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
            "distinct_fails": distinct_fails,
            "per_run_stats": per_run_records
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

    evaluator.run_evaluation(HillClimbing, "Hill Climbing", seed_base=0, n_scenarios=50, n_eval=2, patience=3)
    evaluator.run_evaluation(HillClimbing, "Hill Climbing", seed_base=31, n_scenarios=33, n_eval=3, patience=3)
    evaluator.run_evaluation(HillClimbing, "Hill Climbing", seed_base=31 * 2, n_scenarios=25, n_eval=4)
    evaluator.run_evaluation(HillClimbing, "Hill Climbing", seed_base=31 * 3, n_scenarios=20, n_eval=5)
    evaluator.run_evaluation(HillClimbing, "Hill Climbing", seed_base=31 * 4, n_scenarios=17, n_eval=6)
    evaluator.run_evaluation(HillClimbing, "Hill Climbing", seed_base=31 * 5, n_scenarios=14, n_eval=7)
    evaluator.run_evaluation(HillClimbing, "Hill Climbing", seed_base=31 * 6, n_scenarios=13, n_eval=8)
    evaluator.run_evaluation(HillClimbing, "Hill Climbing", seed_base=31 * 7, n_scenarios=11, n_eval=9)
    evaluator.run_evaluation(HillClimbing, "Hill Climbing", seed_base=31 * 8, n_scenarios=10, n_eval=10)
    
    evaluator.run_evaluation(RandomSearch, "Random Search", n_runs=50, n_scenarios=100)
