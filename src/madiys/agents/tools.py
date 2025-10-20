from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Tuple


class PredictionTool(ABC):
	@abstractmethod
	def train(self, features, target) -> Any:  # returns fitted model
		...

	@abstractmethod
	def predict(self, features) -> Iterable[float]:
		...


class OptimizationTool(ABC):
	@abstractmethod
	def optimize(
		self,
		objective_fn,
		bounds: List[Tuple[float, float]],
		n_calls: int = 20,
		initial_points: int = 5,
	) -> Dict[str, Any]:
		...


class PersistenceTool(ABC):
	@abstractmethod
	def log_run(self, name: str, meta: Dict[str, Any]) -> int:
		...

	@abstractmethod
	def log_result(self, run_id: int, x: Dict[str, float], y: float) -> None:
		...

	@abstractmethod
	def list_runs(self) -> List[Dict[str, Any]]:
		...


