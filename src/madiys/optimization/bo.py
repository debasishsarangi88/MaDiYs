from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
from skopt import gp_minimize
from skopt.space import Real


def run_bo(
	objective_fn: Callable[[List[float]], float],
	bounds: List[Tuple[float, float]],
	n_calls: int = 25,
	initial_points: int = 5,
) -> Dict[str, object]:
	"""Run single-objective BO with GP surrogate.

	Returns dict with keys: result, x_best, y_best, history.
	"""
	space = [Real(low, high) for (low, high) in bounds]
	res = gp_minimize(
		objective_fn,
		space,
		n_calls=n_calls,
		n_initial_points=initial_points,
		acq_func="EI",
		random_state=42,
	)
	return {
		"result": res,
		"x_best": res.x,
		"y_best": res.fun,
		"history": {
			"xs": np.asarray(res.x_iters).tolist(),
			"ys": np.asarray(res.func_vals, dtype=float).tolist(),
		},
	}


