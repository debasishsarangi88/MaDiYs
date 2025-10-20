from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import json
import pandas as pd

from ..models import BaselineRegressor
from ..optimization import run_bo
from ..persistence import get_engine, init_db, log_run, log_result, list_runs as _list_runs


@dataclass
class OrchestratorConfig:
	persistence_url: str = "sqlite:///madiys.db"


class Orchestrator:
	"""Coordinates tools: prediction, optimization, persistence."""

	def __init__(self, config: Optional[OrchestratorConfig] = None) -> None:
		self.config = config or OrchestratorConfig()
		self.model: Optional[BaselineRegressor] = None
		self._engine = get_engine(self.config.persistence_url)

	def ensure_db(self) -> None:
		init_db(self._engine)

	def plan(self, goal: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
		steps: List[str] = []
		if goal == "predict":
			steps = ["maybe_train", "predict"]
		elif goal == "optimize":
			steps = ["ensure_model", "optimize", "maybe_persist"]
		else:
			steps = ["noop"]
		return {"goal": goal, "steps": steps, "context": context or {}}

	def train_model(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
		model = BaselineRegressor()
		if y is not None:
			model.fit(X, y)
		self.model = model

	def save_model(self, path: str) -> None:
		if self.model is None:
			raise RuntimeError("No model to save")
		self.model.save(path)

	def load_model(self, path: str) -> None:
		self.model = BaselineRegressor.load(path)

	def predict(self, X: pd.DataFrame) -> List[float]:
		if self.model is None:
			raise RuntimeError("Model not available; train or load first")
		preds = self.model.predict(X)
		return [float(v) for v in preds]

	def optimize(
		self,
		feature_names: List[str],
		bounds: List[Tuple[float, float]],
		n_calls: int = 20,
		initial_points: int = 5,
		log_results: bool = True,
	) -> Dict[str, Any]:
		if self.model is None:
			raise RuntimeError("Model not available; train or load first")

		def objective(x_list: List[float]) -> float:
			row = pd.DataFrame([dict(zip(feature_names, x_list))])[feature_names]
			# negate to maximize
			return float(-self.model.predict(row)[0])

		res = run_bo(objective, bounds, n_calls=n_calls, initial_points=initial_points)
		output = {
			"x_best": dict(zip(feature_names, res["x_best"])),
			"y_best": float(-res["y_best"]),
			"history": {"xs": res["history"]["xs"], "ys": [float(-y) for y in res["history"]["ys"]]},
		}

		if log_results:
			self.ensure_db()
			run_id = log_run(self._engine, name="bo_run", meta_json=json.dumps({"features": feature_names}))
			for x_vals, y_val in zip(output["history"]["xs"], output["history"]["ys"]):
				payload = json.dumps(dict(zip(feature_names, x_vals)))
				log_result(self._engine, run_id, x_json=payload, y_value=float(y_val))
			output["run_id"] = run_id

		return output

	def list_runs(self) -> List[Dict[str, Any]]:
		self.ensure_db()
		return _list_runs(self._engine)


