from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
	from mp_api.client import MPRester
except Exception:  # pragma: no cover - optional dependency
	MPRester = None  # type: ignore


class MPCache:
	"""SQLite-based cache for Materials Project queries."""
	
	def __init__(self, cache_path: str = "mp_cache.db"):
		self.cache_path = Path(cache_path)
		self._init_db()
	
	def _init_db(self) -> None:
		"""Initialize cache database."""
		with sqlite3.connect(self.cache_path) as conn:
			conn.execute("""
				CREATE TABLE IF NOT EXISTS mp_cache (
					query_hash TEXT PRIMARY KEY,
					query_params TEXT,
					data_json TEXT,
					created_at TIMESTAMP,
					expires_at TIMESTAMP
				)
			""")
			conn.execute("""
				CREATE INDEX IF NOT EXISTS idx_expires ON mp_cache(expires_at)
			""")
	
	def _get_query_hash(self, apikey: str, fields: List[str], formula: Optional[str], limit: int) -> str:
		"""Generate hash for query parameters."""
		query_str = f"{apikey}:{sorted(fields)}:{formula}:{limit}"
		return hashlib.md5(query_str.encode()).hexdigest()
	
	def get(self, apikey: str, fields: List[str], formula: Optional[str], limit: int) -> Optional[pd.DataFrame]:
		"""Get cached data if available and not expired."""
		query_hash = self._get_query_hash(apikey, fields, formula, limit)
		
		with sqlite3.connect(self.cache_path) as conn:
			cursor = conn.execute("""
				SELECT data_json FROM mp_cache 
				WHERE query_hash = ? AND expires_at > ?
			""", (query_hash, datetime.utcnow()))
			
			row = cursor.fetchone()
			if row:
				data = json.loads(row[0])
				return pd.DataFrame(data)
		return None
	
	def set(self, apikey: str, fields: List[str], formula: Optional[str], limit: int, 
			data: pd.DataFrame, ttl_hours: int = 24) -> None:
		"""Cache data with TTL."""
		query_hash = self._get_query_hash(apikey, fields, formula, limit)
		query_params = json.dumps({
			"fields": fields,
			"formula": formula,
			"limit": limit
		})
		
		now = datetime.utcnow()
		expires_at = now + timedelta(hours=ttl_hours)
		
		with sqlite3.connect(self.cache_path) as conn:
			conn.execute("""
				INSERT OR REPLACE INTO mp_cache 
				(query_hash, query_params, data_json, created_at, expires_at)
				VALUES (?, ?, ?, ?, ?)
			""", (query_hash, query_params, json.dumps(data.to_dict('records')), now, expires_at))
	
	def cleanup_expired(self) -> None:
		"""Remove expired cache entries."""
		with sqlite3.connect(self.cache_path) as conn:
			conn.execute("DELETE FROM mp_cache WHERE expires_at <= ?", (datetime.utcnow(),))


def map_mp_features(df: pd.DataFrame) -> pd.DataFrame:
	"""Map MP fields to model-ready features."""
	mapped_df = df.copy()
	
	# Create feature mappings
	feature_mappings = {
		# Direct mappings (already numeric)
		'formation_energy_per_atom': 'formation_energy_per_atom',
		'band_gap': 'band_gap',
		'density': 'density',
		'volume': 'volume',
		'energy_above_hull': 'energy_above_hull',
		'is_stable': 'is_stable',
		'is_magnetic': 'is_magnetic',
		'total_magnetization': 'total_magnetization',
		'num_elements': 'num_elements',
		'num_sites': 'num_sites',
	}
	
	# Apply direct mappings
	for mp_field, feature_name in feature_mappings.items():
		if mp_field in mapped_df.columns:
			mapped_df[feature_name] = pd.to_numeric(mapped_df[mp_field], errors='coerce')
	
	# Create derived features
	if 'formula_pretty' in mapped_df.columns:
		# Element count features
		element_counts = mapped_df['formula_pretty'].str.extractall(r'([A-Z][a-z]*)').groupby(level=0)[0].value_counts().unstack(fill_value=0)
		for element in element_counts.columns:
			mapped_df[f'count_{element}'] = element_counts[element]
	
	# Create composition-based features
	if 'formula_pretty' in mapped_df.columns:
		# Extract atomic ratios
		formulas = mapped_df['formula_pretty'].fillna('')
		mapped_df['formula_length'] = formulas.str.len()
		mapped_df['has_oxygen'] = formulas.str.contains('O', na=False).astype(int)
		mapped_df['has_metal'] = formulas.str.contains(r'[A-Z][a-z]*', na=False).astype(int)
	
	# Fill NaN values with median for numeric columns
	numeric_cols = mapped_df.select_dtypes(include=[float, int]).columns
	for col in numeric_cols:
		if col not in ['material_id']:  # Skip ID columns
			mapped_df[col] = mapped_df[col].fillna(mapped_df[col].median())
	
	return mapped_df


def fetch_mp_samples(
	apikey: str,
	fields: Optional[List[str]] = None,
	formula: Optional[str] = None,
	limit: int = 200,
	use_cache: bool = True,
	cache_path: str = "mp_cache.db",
	ttl_hours: int = 24,
) -> pd.DataFrame:
	"""Fetch a sample from the Materials Project API with caching and feature mapping.
	
	Args:
		apikey: Materials Project API key
		fields: List of fields to fetch (default includes common properties)
		formula: Optional formula filter (e.g., "Fe2O3")
		limit: Maximum number of records to fetch
		use_cache: Whether to use caching
		cache_path: Path to cache database
		ttl_hours: Cache TTL in hours
	
	Returns:
		DataFrame with mapped features ready for ML
	"""
	if MPRester is None:
		raise RuntimeError("mp-api not installed. Please install mp-api and pymatgen.")

	default_fields = [
		"material_id",
		"formula_pretty",
		"formation_energy_per_atom",
		"band_gap",
		"density",
		"volume",
		"energy_above_hull",
		"is_stable",
		"is_magnetic",
		"total_magnetization",
		"num_elements",
		"num_sites",
		"structure",
	]
	query_fields = fields or default_fields
	
	# Check cache first
	if use_cache:
		cache = MPCache(cache_path)
		cached_data = cache.get(apikey, query_fields, formula, limit)
		if cached_data is not None:
			return map_mp_features(cached_data)
	
	# Fetch from API
	with MPRester(apikey) as mpr:
		criteria: Dict[str, object] = {}
		if formula:
			criteria["formula_pretty"] = formula
			docs = mpr.materials.summary.search(
				formula=criteria["formula_pretty"],
				fields=query_fields,
				chunk_size=limit,
			)
		else:
			docs = mpr.materials.summary.search(fields=query_fields, chunk_size=limit)

		data = [d.dict() for d in docs]
		df = pd.DataFrame(data)
	
	# Cache the result
	if use_cache:
		cache.set(apikey, query_fields, formula, limit, df, ttl_hours)
	
	# Map features and return
	return map_mp_features(df)


def clear_mp_cache(cache_path: str = "mp_cache.db") -> None:
	"""Clear all cached MP data."""
	cache = MPCache(cache_path)
	with sqlite3.connect(cache_path) as conn:
		conn.execute("DELETE FROM mp_cache")


