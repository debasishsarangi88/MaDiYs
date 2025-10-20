from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field


class MaterialSample(BaseModel):
	composition: Dict[str, float] = Field(default_factory=dict)
	features: Dict[str, float] = Field(default_factory=dict)
	target: Optional[float] = None


class TargetSpec(BaseModel):
	name: str
	goal: str = Field(description="maximize|minimize|target")
	constraint: Optional[str] = None


