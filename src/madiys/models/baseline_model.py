from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class BaselineRegressorConfig:
	n_estimators: int = 300
	random_state: int = 42


class BaselineRegressor:
	"""Baseline tabular regressor using StandardScaler + RandomForest."""

	def __init__(self, config: Optional[BaselineRegressorConfig] = None) -> None:
		self.config = config or BaselineRegressorConfig()
		self.pipeline: Optional[Pipeline] = None

	def fit(self, X: pd.DataFrame, y: Iterable[float]) -> None:
		X_np = np.asarray(X, dtype=float)
		y_np = np.asarray(list(y), dtype=float)
		rf = RandomForestRegressor(
			n_estimators=self.config.n_estimators,
			random_state=self.config.random_state,
		)
		self.pipeline = Pipeline([
			("scale", StandardScaler()),
			("rf", rf),
		])
		self.pipeline.fit(X_np, y_np)

	def predict(self, X: pd.DataFrame) -> np.ndarray:
		if self.pipeline is None:
			raise RuntimeError("Model not fitted")
		return self.pipeline.predict(np.asarray(X, dtype=float))

	def save(self, path: str) -> None:
		if self.pipeline is None:
			raise RuntimeError("Nothing to save; model not fitted")
		joblib.dump({"config": self.config, "pipeline": self.pipeline}, path)

	@classmethod
	def load(cls, path: str) -> "BaselineRegressor":
		obj = joblib.load(path)
		inst = cls(config=obj.get("config"))
		inst.pipeline = obj["pipeline"]
		return inst


