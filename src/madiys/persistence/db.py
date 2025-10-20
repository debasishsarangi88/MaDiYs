from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlmodel import Field, Session, SQLModel, create_engine, select


class ExperimentRun(SQLModel, table=True):
	id: Optional[int] = Field(default=None, primary_key=True)
	name: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	meta_json: str = Field(default="{}")


class CandidateResult(SQLModel, table=True):
	id: Optional[int] = Field(default=None, primary_key=True)
	run_id: int = Field(index=True)
	x_json: str
	y_value: float


def get_engine(url: str = "sqlite:///madiys.db"):
	return create_engine(url)


def init_db(engine) -> None:
	SQLModel.metadata.create_all(engine)


def log_run(engine, name: str, meta_json: str = "{}") -> int:
	with Session(engine) as session:
		run = ExperimentRun(name=name, meta_json=meta_json)
		session.add(run)
		session.commit()
		session.refresh(run)
		return int(run.id)  # type: ignore[arg-type]


def log_result(engine, run_id: int, x_json: str, y_value: float) -> None:
	with Session(engine) as session:
		res = CandidateResult(run_id=run_id, x_json=x_json, y_value=y_value)
		session.add(res)
		session.commit()


def list_runs(engine) -> List[Dict[str, Any]]:
	with Session(engine) as session:
		rows = session.exec(select(ExperimentRun)).all()
		return [
			{"id": r.id, "name": r.name, "created_at": r.created_at.isoformat()}
			for r in rows
		]


