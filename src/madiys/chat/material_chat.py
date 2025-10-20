from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from ..generation import PropertyConstraints


def parse_material_request(request: str) -> Tuple[PropertyConstraints, str]:
	"""Parse natural language material request into property constraints.
	
	Args:
		request: Natural language description of desired material
		
	Returns:
		Tuple of (PropertyConstraints, explanation)
	"""
	request_lower = request.lower()
	
	# Initialize default constraints
	constraints = PropertyConstraints()
	explanation_parts = []
	
	# Parse dielectric constant requirements
	if "low dielectric" in request_lower or "low permittivity" in request_lower:
		constraints.band_gap_range = (0.0, 2.0)  # Low band gap for low dielectric
		explanation_parts.append("Set band gap range to 0-2 eV for low dielectric constant")
	
	if "high dielectric" in request_lower or "high permittivity" in request_lower:
		constraints.band_gap_range = (3.0, 10.0)  # Higher band gap
		explanation_parts.append("Set band gap range to 3-10 eV for high dielectric constant")
	
	# Parse conductivity requirements
	if "conductive" in request_lower or "conductor" in request_lower:
		constraints.band_gap_range = (0.0, 1.0)  # Very low band gap
		explanation_parts.append("Set band gap range to 0-1 eV for conductive materials")
	
	if "insulator" in request_lower or "insulating" in request_lower:
		constraints.band_gap_range = (4.0, 10.0)  # High band gap
		explanation_parts.append("Set band gap range to 4-10 eV for insulating materials")
	
	# Parse density requirements
	if "lightweight" in request_lower or "light weight" in request_lower or "low density" in request_lower:
		constraints.density_range = (1.0, 5.0)
		explanation_parts.append("Set density range to 1-5 g/cm³ for lightweight materials")
	
	if "heavy" in request_lower or "high density" in request_lower:
		constraints.density_range = (10.0, 20.0)
		explanation_parts.append("Set density range to 10-20 g/cm³ for heavy materials")
	
	# Parse stability requirements
	if "stable" in request_lower or "stability" in request_lower:
		constraints.require_stable = True
		constraints.min_stability = 0.7
		explanation_parts.append("Set stability requirements for stable materials")
	
	if "metastable" in request_lower:
		constraints.require_stable = False
		constraints.min_stability = 0.3
		explanation_parts.append("Set lower stability threshold for metastable materials")
	
	# Parse specific elements
	element_patterns = {
		"silicon": ["Si"],
		"carbon": ["C"],
		"titanium": ["Ti"],
		"aluminum": ["Al"],
		"iron": ["Fe"],
		"copper": ["Cu"],
		"gold": ["Au"],
		"silver": ["Ag"],
		"oxide": ["O"],
		"nitride": ["N"],
		"carbide": ["C"],
		"silicate": ["Si", "O"],
		"ceramic": ["O", "N", "C"],
		"metal": ["Fe", "Ti", "Al", "Cu", "Ni", "Cr", "Mo", "W"],
		"semiconductor": ["Si", "Ge", "Ga", "As", "In", "P"],
	}
	
	allowed_elements = set()
	for keyword, elements in element_patterns.items():
		if keyword in request_lower:
			allowed_elements.update(elements)
			explanation_parts.append(f"Added {keyword} elements: {', '.join(elements)}")
	
	if allowed_elements:
		constraints.allowed_elements = list(allowed_elements)
	
	# Parse forbidden elements
	forbidden_patterns = {
		"no carbon": ["C"],
		"no silicon": ["Si"],
		"no metal": ["Fe", "Ti", "Al", "Cu", "Ni", "Cr", "Mo", "W"],
		"no oxygen": ["O"],
		"no nitrogen": ["N"],
	}
	
	forbidden_elements = set()
	for keyword, elements in forbidden_patterns.items():
		if keyword in request_lower:
			forbidden_elements.update(elements)
			explanation_parts.append(f"Excluded {keyword} elements: {', '.join(elements)}")
	
	if forbidden_elements:
		constraints.forbidden_elements = list(forbidden_elements)
	
	# Parse industrial applications
	if "industrial" in request_lower or "commercial" in request_lower:
		# Common industrial materials
		constraints.allowed_elements = ["Si", "O", "Al", "Fe", "Ti", "C", "N"]
		constraints.require_stable = True
		explanation_parts.append("Set constraints for industrial-grade materials")
	
	if "aerospace" in request_lower or "aviation" in request_lower:
		constraints.density_range = (1.0, 8.0)  # Lightweight
		constraints.require_stable = True
		explanation_parts.append("Set lightweight and stable constraints for aerospace applications")
	
	if "automotive" in request_lower or "car" in request_lower:
		constraints.allowed_elements = ["Fe", "Al", "Ti", "C", "Si", "O"]
		constraints.require_stable = True
		explanation_parts.append("Set constraints for automotive applications")
	
	if "electronics" in request_lower or "electronic" in request_lower:
		constraints.band_gap_range = (0.5, 3.0)  # Semiconductor range
		constraints.allowed_elements = ["Si", "Ge", "Ga", "As", "In", "P", "O", "N"]
		explanation_parts.append("Set semiconductor constraints for electronic applications")
	
	if "energy" in request_lower or "battery" in request_lower or "solar" in request_lower:
		constraints.band_gap_range = (1.0, 3.5)  # Good for energy applications
		constraints.require_stable = True
		explanation_parts.append("Set constraints for energy applications")
	
	# Parse specific property values
	# Look for specific numbers
	numbers = re.findall(r'\d+\.?\d*', request)
	if numbers:
		# Try to match numbers with properties
		if "band gap" in request_lower or "bandgap" in request_lower:
			try:
				value = float(numbers[0])
				constraints.band_gap_range = (value * 0.8, value * 1.2)
				explanation_parts.append(f"Set band gap around {value} eV")
			except (ValueError, IndexError):
				pass
		
		if "density" in request_lower:
			try:
				value = float(numbers[0])
				constraints.density_range = (value * 0.8, value * 1.2)
				explanation_parts.append(f"Set density around {value} g/cm³")
			except (ValueError, IndexError):
				pass
	
	# Default constraints if nothing specific found
	if not explanation_parts:
		explanation_parts.append("Applied default material constraints")
		constraints.require_stable = True
		constraints.min_stability = 0.5
	
	explanation = " | ".join(explanation_parts)
	return constraints, explanation


class MaterialChatBot:
	"""Chat interface for material discovery requests."""
	
	def __init__(self):
		self.conversation_history = []
		self.example_requests = [
			"Create a material with low dielectric constant currently used industrially",
			"Generate a lightweight ceramic for aerospace applications",
			"Find a conductive material for electronics",
			"Design a stable oxide for high-temperature applications",
			"Create a semiconductor with band gap around 2 eV",
			"Generate a lightweight metal alloy for automotive use",
			"Find an insulating material for electrical applications",
			"Design a stable nitride for industrial coatings",
		]
	
	def process_request(self, request: str) -> Dict:
		"""Process a natural language material request."""
		# Parse the request
		constraints, explanation = parse_material_request(request)
		
		# Add to conversation history
		self.conversation_history.append({
			"user_request": request,
			"constraints": constraints,
			"explanation": explanation,
			"timestamp": "now"
		})
		
		return {
			"constraints": constraints,
			"explanation": explanation,
			"success": True,
			"message": f"Parsed your request: '{request}'\n\n{explanation}"
		}
	
	def get_example_requests(self) -> List[str]:
		"""Get example material requests."""
		return self.example_requests
	
	def get_conversation_history(self) -> List[Dict]:
		"""Get conversation history."""
		return self.conversation_history
	
	def clear_history(self):
		"""Clear conversation history."""
		self.conversation_history = []
