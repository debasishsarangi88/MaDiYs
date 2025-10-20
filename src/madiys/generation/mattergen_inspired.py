from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

try:
	from pymatgen.core import Structure, Lattice, Element
	from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
	from pymatgen.analysis.structure_matcher import StructureMatcher
except ImportError:
	Structure = None  # type: ignore
	Lattice = None  # type: ignore
	Element = None  # type: ignore
	SpacegroupAnalyzer = None  # type: ignore
	StructureMatcher = None  # type: ignore


class PropertyConstraints(BaseModel):
	"""Property constraints for material generation (inspired by MatterGen)."""
	
	# Chemical constraints
	allowed_elements: List[str] = Field(default_factory=list, description="Allowed chemical elements")
	forbidden_elements: List[str] = Field(default_factory=list, description="Forbidden chemical elements")
	max_elements: int = Field(default=5, description="Maximum number of different elements")
	
	# Structural constraints
	crystal_system: Optional[str] = Field(default=None, description="Target crystal system")
	space_group: Optional[int] = Field(default=None, description="Target space group number")
	
	# Property constraints (ranges)
	band_gap_range: Tuple[float, float] = Field(default=(0.0, 10.0), description="Band gap range (eV)")
	density_range: Tuple[float, float] = Field(default=(1.0, 20.0), description="Density range (g/cm³)")
	formation_energy_range: Tuple[float, float] = Field(default=(-5.0, 2.0), description="Formation energy range (eV/atom)")
	
	# Stability constraints
	min_stability: float = Field(default=0.0, description="Minimum stability threshold")
	require_stable: bool = Field(default=True, description="Require stable materials only")


class MaterialGenerator:
	"""MatterGen-inspired material generator with property constraints."""
	
	def __init__(self, constraints: Optional[PropertyConstraints] = None):
		self.constraints = constraints or PropertyConstraints()
		self.generated_materials: List[Dict] = []
	
	def generate_candidates(
		self, 
		n_candidates: int = 10,
		base_composition: Optional[str] = None,
		use_constraints: bool = True
	) -> List[Dict]:
		"""Generate material candidates with property constraints."""
		
		candidates = []
		
		for _ in range(n_candidates):
			# Generate composition
			composition = self._generate_composition(base_composition)
			
			# Generate crystal structure
			structure = self._generate_structure(composition)
			
			# Estimate properties
			properties = self._estimate_properties(structure, composition)
			
			# Apply constraints if enabled
			if use_constraints and not self._meets_constraints(properties):
				continue
			
			candidate = {
				"composition": composition,
				"structure": structure,
				"properties": properties,
				"formula": self._composition_to_formula(composition),
				"stability_score": self._calculate_stability_score(properties),
			}
			
			candidates.append(candidate)
			self.generated_materials.append(candidate)
		
		return candidates
	
	def _generate_composition(self, base_composition: Optional[str] = None) -> Dict[str, float]:
		"""Generate a chemical composition."""
		if base_composition:
			# Parse base composition (e.g., "Fe2O3" -> {"Fe": 2, "O": 3})
			return self._parse_composition(base_composition)
		
		# Generate random composition
		allowed_elements = self.constraints.allowed_elements or [
			"H", "Li", "Be", "B", "C", "N", "O", "F", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
			"K", "Ca", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
			"Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",
			"Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
			"Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn"
		]
		
		# Filter out forbidden elements
		allowed_elements = [el for el in allowed_elements if el not in self.constraints.forbidden_elements]
		
		# Random number of elements (2 to max_elements)
		n_elements = random.randint(2, min(self.constraints.max_elements, len(allowed_elements)))
		selected_elements = random.sample(allowed_elements, n_elements)
		
		# Random stoichiometry
		composition = {}
		for element in selected_elements:
			composition[element] = random.randint(1, 6)  # Random stoichiometry 1-6
		
		return composition
	
	def _parse_composition(self, formula: str) -> Dict[str, float]:
		"""Parse chemical formula to composition dict."""
		# Simple parser for formulas like "Fe2O3"
		import re
		pattern = r'([A-Z][a-z]*)(\d*)'
		matches = re.findall(pattern, formula)
		
		composition = {}
		for element, count in matches:
			composition[element] = float(count) if count else 1.0
		
		return composition
	
	def _generate_structure(self, composition: Dict[str, float]) -> Optional[Dict]:
		"""Generate crystal structure (simplified)."""
		if Structure is None:
			# Fallback: return basic structure info
			return {
				"lattice_type": "cubic",
				"a": random.uniform(3.0, 15.0),
				"b": random.uniform(3.0, 15.0),
				"c": random.uniform(3.0, 15.0),
				"alpha": 90.0,
				"beta": 90.0,
				"gamma": 90.0,
			}
		
		# Generate random lattice parameters
		a = random.uniform(3.0, 15.0)
		b = random.uniform(3.0, 15.0)
		c = random.uniform(3.0, 15.0)
		alpha = random.uniform(60.0, 120.0)
		beta = random.uniform(60.0, 120.0)
		gamma = random.uniform(60.0, 120.0)
		
		lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
		
		# Generate random atomic positions
		species = []
		coords = []
		for element, count in composition.items():
			for _ in range(int(count)):
				species.append(Element(element))
				coords.append([random.random(), random.random(), random.random()])
		
		structure = Structure(lattice, species, coords)
		
		return {
			"structure": structure,
			"lattice": {
				"a": a, "b": b, "c": c,
				"alpha": alpha, "beta": beta, "gamma": gamma
			},
			"volume": structure.volume,
			"density": structure.density,
		}
	
	def _estimate_properties(self, structure: Dict, composition: Dict[str, float]) -> Dict:
		"""Estimate material properties (simplified models)."""
		
		# Simple property estimation based on composition and structure
		total_atoms = sum(composition.values())
		
		# Band gap estimation (simplified)
		band_gap = random.uniform(0.0, 5.0)  # Random for now
		
		# Density estimation
		if "density" in structure:
			density = structure["density"]
		else:
			density = random.uniform(1.0, 20.0)
		
		# Formation energy estimation
		formation_energy = random.uniform(-3.0, 1.0)
		
		# Stability estimation
		stability = random.uniform(0.0, 1.0)
		
		return {
			"band_gap": band_gap,
			"density": density,
			"formation_energy_per_atom": formation_energy,
			"is_stable": stability > 0.5,
			"stability_score": stability,
			"volume": structure.get("volume", random.uniform(50, 500)),
			"num_elements": len(composition),
			"total_atoms": total_atoms,
		}
	
	def _meets_constraints(self, properties: Dict) -> bool:
		"""Check if material meets property constraints."""
		constraints = self.constraints
		
		# Check band gap range
		band_gap = properties.get("band_gap", 0)
		if not (constraints.band_gap_range[0] <= band_gap <= constraints.band_gap_range[1]):
			return False
		
		# Check density range
		density = properties.get("density", 0)
		if not (constraints.density_range[0] <= density <= constraints.density_range[1]):
			return False
		
		# Check formation energy range
		formation_energy = properties.get("formation_energy_per_atom", 0)
		if not (constraints.formation_energy_range[0] <= formation_energy <= constraints.formation_energy_range[1]):
			return False
		
		# Check stability
		if constraints.require_stable and not properties.get("is_stable", False):
			return False
		
		# Check minimum stability
		stability_score = properties.get("stability_score", 0)
		if stability_score < constraints.min_stability:
			return False
		
		return True
	
	def _calculate_stability_score(self, properties: Dict) -> float:
		"""Calculate stability score for material."""
		# Simple stability score based on formation energy and other factors
		formation_energy = properties.get("formation_energy_per_atom", 0)
		band_gap = properties.get("band_gap", 0)
		
		# Lower formation energy = more stable
		stability = max(0, 1.0 - abs(formation_energy) / 3.0)
		
		# Band gap contributes to stability
		if 0.5 <= band_gap <= 3.0:  # Optimal band gap range
			stability += 0.2
		
		return min(1.0, stability)
	
	def _composition_to_formula(self, composition: Dict[str, float]) -> str:
		"""Convert composition dict to chemical formula."""
		formula_parts = []
		for element, count in sorted(composition.items()):
			if count == 1:
				formula_parts.append(element)
			else:
				formula_parts.append(f"{element}{int(count)}")
		
		return "".join(formula_parts)
	
	def get_generation_stats(self) -> Dict:
		"""Get statistics about generated materials."""
		if not self.generated_materials:
			return {"total": 0}
		
		band_gaps = [m["properties"]["band_gap"] for m in self.generated_materials]
		densities = [m["properties"]["density"] for m in self.generated_materials]
		stabilities = [m["stability_score"] for m in self.generated_materials]
		
		return {
			"total": len(self.generated_materials),
			"band_gap": {"min": min(band_gaps), "max": max(band_gaps), "mean": np.mean(band_gaps)},
			"density": {"min": min(densities), "max": max(densities), "mean": np.mean(densities)},
			"stability": {"min": min(stabilities), "max": max(stabilities), "mean": np.mean(stabilities)},
		}
