import copy
import math
import numpy as np
from envs.highway_env_utils import run_episode, record_video_episode
from search.base_search import ScenarioSearch
from search.hill_climbing import compute_objectives_from_time_series, compute_fitness
from tqdm import trange


class GeneticAlgorithm(ScenarioSearch):
    """Genetic Algorithm for scenario search."""

    def run_search(self, iterations=50, population_size=10, seed=42):
        """
        Run genetic algorithm search.
        
        Parameters:
            iterations (n): number of generations
            population_size (p): size of the population
            seed: random seed
        """
        print(f"Running Genetic Algorithm Search (n={iterations}, p={population_size})...")
        rng = np.random.default_rng(seed)
        
        # Hyperparameters
        crossover_chance = 0.5
        mutation_chance = 0.1
        
        # 1. Initialize population of size p
        population = [self.sample_random_config(rng) for _ in range(population_size)]
        
        best_cfg = None
        best_loss = -float('inf')
        best_seed = None
        crash_log = []

        for i in trange(iterations):
            # 2. Evaluate population
            evals = []
            for cfg in population:
                s = int(rng.integers(1e9))
                crashed, ts = run_episode(self.env_id, cfg, self.policy, self.defaults, s)
                obj = compute_objectives_from_time_series(ts)
                # TODO: found bug in code. crashed = True
                # even though filter(lambda e: e['crashed'], ts) = empty
                if crashed:
                    loss = 100.0
                else:
                    loss = min(compute_fitness(obj), 100.0)
                
                evals.append({
                    'cfg': copy.deepcopy(cfg),
                    'loss': loss,
                    'seed': s,
                    'crashed': crashed
                })
                
                # Keep track of best result
                if loss > best_loss:
                    best_loss = loss
                    best_cfg = copy.deepcopy(cfg)
                    best_seed = s
                
                if crashed:
                    print(f"  Collision found! Gen {i}, Loss: {loss}")
                    crash_log.append({"cfg": copy.deepcopy(cfg), "seed": s})
                    record_video_episode(self.env_id, cfg, self.policy, self.defaults, s, out_dir="videos")

            # Save/print results for this generation
            losses = np.array([e['loss'] for e in evals])
            print(f"Gen {i}: Max Loss = {np.max(losses):.4f}")

            # 3. Create roulette spinning wheel (probabilistic selection)
            total_loss = np.sum(losses)
            if total_loss == 0:
                probs = np.ones(population_size) / population_size
            else:
                probs = losses / total_loss

            # 4. Form new population
            new_population = []
            while len(new_population) < population_size:
                # Select a pair
                idx1 = rng.choice(population_size, p=probs)
                idx2 = rng.choice(population_size, p=probs)
                
                # Don't allow pairs of the same individual
                while idx1 == idx2:
                    idx2 = rng.choice(population_size, p=probs)
                
                p1_cfg = evals[idx1]['cfg']
                p2_cfg = evals[idx2]['cfg']
                
                # 0.5 crossover chance
                if rng.random() < crossover_chance:
                    # Single-point crossover
                    c1_cfg, c2_cfg = self._crossover(p1_cfg, p2_cfg, rng)
                else:
                    # Parent stay untouched
                    c1_cfg, c2_cfg = copy.deepcopy(p1_cfg), copy.deepcopy(p2_cfg)
                
                # 0.1 mutation chance (prob checked per parameter in _mutate)
                c1_cfg = self._mutate(c1_cfg, rng, mutation_chance)
                c2_cfg = self._mutate(c2_cfg, rng, mutation_chance)
                
                new_population.append(c1_cfg)
                if len(new_population) < population_size:
                    new_population.append(c2_cfg)
            
            population = new_population

        # Record video for the best found config if it crashed
        if best_cfg and best_loss >= 100: # We used 100 for crash
            record_video_episode(self.env_id, best_cfg, self.policy, self.defaults, best_seed, out_dir="videos")

        return crash_log

    def _crossover(self, p1, p2, rng):
        """Single-point crossover for all parameters."""
        keys = sorted(list(self.param_spec.keys()))
        if len(keys) < 2:
            return copy.deepcopy(p1), copy.deepcopy(p2)
        
        point = rng.integers(1, len(keys))
        c1 = copy.deepcopy(p1)
        c2 = copy.deepcopy(p2)
        
        for i, k in enumerate(keys):
            if i >= point:
                # Swap values
                c1[k] = p2[k]
                c2[k] = p1[k]
        
        # Repair lane constraints if necessary
        for c in [c1, c2]:
            if "lanes_count" in c and "initial_lane_id" in c:
                c["initial_lane_id"] = int(np.clip(c["initial_lane_id"], 0, c["lanes_count"] - 1))
                
        return c1, c2

    def _mutate(self, cfg, rng, mutation_chance):
        """Mutate parameters using Bayesian perturbation."""
        new_cfg = copy.deepcopy(cfg)
        for k, spec in self.param_spec.items():
            if k not in new_cfg:
                continue
            
            if rng.random() < mutation_chance:
                # Bayesian perturbation: Gaussian noise with small variance relative to range
                min_v = spec["min"]
                max_v = spec["max"]
                
                # Infer appropriate small variance (e.g., 5% of range)
                sigma = (max_v - min_v) * 0.05
                noise = rng.normal(0, sigma)
                
                if spec["type"] == "int":
                    new_val = int(np.clip(round(new_cfg[k] + noise), min_v, max_v))
                else:
                    new_val = float(np.clip(new_cfg[k] + noise, min_v, max_v))
                
                new_cfg[k] = new_val
        
        # Repair lane constraints
        if "lanes_count" in new_cfg and "initial_lane_id" in new_cfg:
            new_cfg["initial_lane_id"] = int(np.clip(new_cfg["initial_lane_id"], 0, new_cfg["lanes_count"] - 1))
            
        return new_cfg
