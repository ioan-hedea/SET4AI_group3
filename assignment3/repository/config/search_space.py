param_spec = {
    "vehicles_count":   {"type": "int",   "min": 5,   "max": 60},
    "lanes_count":      {"type": "int",   "min": 3,   "max": 10},
    "initial_spacing":  {"type": "float", "min": 0.5, "max": 5.0},
    "ego_spacing":      {"type": "float", "min": 1.0, "max": 4.0},
    "initial_lane_id":  {"type": "int",   "min": 0,   "max": 4},
}

base_cfg = {
    "duration": 30,
    "simulation_frequency": 25,
    "policy_frequency": 5,
}