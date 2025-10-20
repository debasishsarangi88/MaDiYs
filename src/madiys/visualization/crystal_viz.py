from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class CrystalVisualizer:
	"""Crystal structure visualization inspired by MatterGen's approach."""
	
	def __init__(self):
		self.colors = {
			"H": "#FFFFFF", "Li": "#CC80FF", "Be": "#C2FF00", "B": "#FFB5B5", "C": "#909090",
			"N": "#3050F8", "O": "#FF0D0D", "F": "#90E050", "Na": "#AB5CF2", "Mg": "#8AFF00",
			"Al": "#BFA6A6", "Si": "#F0C8A0", "P": "#FF8000", "S": "#FFFF30", "Cl": "#1FF01F",
			"K": "#8F40D4", "Ca": "#3DFF00", "Ti": "#BFC2C7", "V": "#A6A6AB", "Cr": "#8A99C7",
			"Mn": "#9C7AC7", "Fe": "#E06633", "Co": "#F090A0", "Ni": "#50D050", "Cu": "#C88033",
			"Zn": "#7D80B0", "Ga": "#C28F8F", "Ge": "#668F8F", "As": "#BD80E3", "Se": "#FFA100",
			"Br": "#A62929", "Rb": "#702EB0", "Sr": "#00FF00", "Y": "#94FFFF", "Zr": "#94E0E0",
			"Nb": "#73C2C9", "Mo": "#54B5B5", "Tc": "#3B9E9E", "Ru": "#248F8F", "Rh": "#0A7D8C",
			"Pd": "#006985", "Ag": "#C0C0C0", "Cd": "#FFD98F", "In": "#A67573", "Sn": "#668080",
			"Sb": "#9E63B5", "Te": "#D47A00", "I": "#940094", "Cs": "#57178F", "Ba": "#00C900",
			"La": "#70D4FF", "Ce": "#FFFFC7", "Pr": "#D9FFC7", "Nd": "#C7FFC7", "Pm": "#A3FFC7",
			"Sm": "#8FFFC7", "Eu": "#61FFC7", "Gd": "#45FFC7", "Tb": "#30FFC7", "Dy": "#1FFFC7",
			"Ho": "#00FF9C", "Er": "#00E675", "Tm": "#00D452", "Yb": "#00BF38", "Lu": "#00AB24",
			"Hf": "#4DC2FF", "Ta": "#4DA6FF", "W": "#2194D6", "Re": "#267DAB", "Os": "#266696",
			"Ir": "#175487", "Pt": "#D0D0E0", "Au": "#FFD123", "Hg": "#B8B8D0", "Tl": "#A6544D",
			"Pb": "#575961", "Bi": "#9E4FB5", "Po": "#AB5C00", "At": "#754F45", "Rn": "#428296"
		}
		
		self.radii = {
			"H": 0.31, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
			"Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02,
			"K": 2.03, "Ca": 1.76, "Ti": 1.40, "V": 1.34, "Cr": 1.22, "Mn": 1.19, "Fe": 1.16, "Co": 1.11,
			"Ni": 1.10, "Cu": 1.12, "Zn": 1.18, "Ga": 1.24, "Ge": 1.21, "As": 1.19, "Se": 1.20, "Br": 1.20,
			"Rb": 2.20, "Sr": 1.95, "Y": 1.78, "Zr": 1.64, "Nb": 1.46, "Mo": 1.39, "Tc": 1.36, "Ru": 1.34,
			"Rh": 1.34, "Pd": 1.37, "Ag": 1.44, "Cd": 1.49, "In": 1.42, "Sn": 1.39, "Sb": 1.39, "Te": 1.38,
			"I": 1.39, "Cs": 2.44, "Ba": 2.15, "La": 1.87, "Ce": 1.82, "Pr": 1.82, "Nd": 1.82, "Pm": 1.81,
			"Sm": 1.80, "Eu": 1.80, "Gd": 1.80, "Tb": 1.78, "Dy": 1.77, "Ho": 1.77, "Er": 1.76, "Tm": 1.75,
			"Yb": 1.94, "Lu": 1.73, "Hf": 1.59, "Ta": 1.46, "W": 1.39, "Re": 1.37, "Os": 1.35, "Ir": 1.33,
			"Pt": 1.33, "Au": 1.34, "Hg": 1.49, "Tl": 1.48, "Pb": 1.47, "Bi": 1.46, "Po": 1.46, "At": 1.45,
			"Rn": 1.45
		}
	
	def _convert_to_float(self, value) -> float:
		"""Convert pymatgen objects to regular Python floats."""
		if value is None:
			return 0.0
		try:
			# Handle pymatgen FloatWithUnit and other special types
			if hasattr(value, 'value'):
				return float(value.value)
			elif hasattr(value, '__float__'):
				return float(value)
			else:
				return float(value)
		except (ValueError, TypeError):
			return 0.0
	
	def visualize_crystal_structure(
		self, 
		structure_data: Dict, 
		composition: Dict[str, float],
		title: str = "Crystal Structure"
	) -> go.Figure:
		"""Create 3D visualization of crystal structure."""
		
		fig = go.Figure()
		
		# Add unit cell
		self._add_unit_cell(fig, structure_data)
		
		# Add atoms
		self._add_atoms(fig, structure_data, composition)
		
		# Add bonds (simplified)
		self._add_bonds(fig, structure_data, composition)
		
		# Update layout
		fig.update_layout(
			title=title,
			scene=dict(
				xaxis_title="X (Å)",
				yaxis_title="Y (Å)",
				zaxis_title="Z (Å)",
				aspectmode="cube",
				camera=dict(
					eye=dict(x=1.5, y=1.5, z=1.5)
				)
			),
			showlegend=True,
			width=800,
			height=600
		)
		
		return fig
	
	def _add_unit_cell(self, fig: go.Figure, structure_data: Dict) -> None:
		"""Add unit cell edges to visualization."""
		if "lattice" not in structure_data:
			return
		
		lattice = structure_data["lattice"]
		a, b, c = lattice["a"], lattice["b"], lattice["c"]
		alpha, beta, gamma = lattice["alpha"], lattice["beta"], lattice["gamma"]
		
		# Convert to Cartesian coordinates (simplified)
		# This is a basic implementation - real conversion would use proper lattice vectors
		vertices = [
			[0, 0, 0], [a, 0, 0], [a, b, 0], [0, b, 0],  # Bottom face
			[0, 0, c], [a, 0, c], [a, b, c], [0, b, c]   # Top face
		]
		
		# Define edges of the unit cell
		edges = [
			[0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
			[4, 5], [5, 6], [6, 7], [7, 4],  # Top face
			[0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
		]
		
		for edge in edges:
			start, end = edge
			fig.add_trace(go.Scatter3d(
				x=[vertices[start][0], vertices[end][0]],
				y=[vertices[start][1], vertices[end][1]],
				z=[vertices[start][2], vertices[end][2]],
				mode='lines',
				line=dict(color='black', width=2),
				showlegend=False,
				name='Unit Cell'
			))
	
	def _add_atoms(self, fig: go.Figure, structure_data: Dict, composition: Dict[str, float]) -> None:
		"""Add atoms to visualization."""
		# Generate random atomic positions (simplified)
		positions = []
		colors = []
		sizes = []
		labels = []
		
		for element, count in composition.items():
			for _ in range(int(count)):
				# Random position in unit cell
				pos = [
					np.random.random() * structure_data.get("lattice", {}).get("a", 10),
					np.random.random() * structure_data.get("lattice", {}).get("b", 10),
					np.random.random() * structure_data.get("lattice", {}).get("c", 10)
				]
				positions.append(pos)
				colors.append(self.colors.get(element, "#808080"))
				sizes.append(self.radii.get(element, 1.0) * 20)  # Scale for visualization
				labels.append(element)
		
		if positions:
			positions = np.array(positions)
			fig.add_trace(go.Scatter3d(
				x=positions[:, 0],
				y=positions[:, 1],
				z=positions[:, 2],
				mode='markers',
				marker=dict(
					size=sizes,
					color=colors,
					opacity=0.8,
					line=dict(width=1, color='black')
				),
				text=labels,
				hovertemplate='%{text}<br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>',
				name='Atoms'
			))
	
	def _add_bonds(self, fig: go.Figure, structure_data: Dict, composition: Dict[str, float]) -> None:
		"""Add chemical bonds to visualization (simplified)."""
		# This is a simplified bond representation
		# Real implementation would calculate actual bond distances
		positions = []
		for element, count in composition.items():
			for _ in range(int(count)):
				pos = [
					np.random.random() * structure_data.get("lattice", {}).get("a", 10),
					np.random.random() * structure_data.get("lattice", {}).get("b", 10),
					np.random.random() * structure_data.get("lattice", {}).get("c", 10)
				]
				positions.append(pos)
		
		if len(positions) > 1:
			positions = np.array(positions)
			# Add some random bonds (simplified)
			for i in range(min(5, len(positions) - 1)):
				j = (i + 1) % len(positions)
				fig.add_trace(go.Scatter3d(
					x=[positions[i][0], positions[j][0]],
					y=[positions[i][1], positions[j][1]],
					z=[positions[i][2], positions[j][2]],
					mode='lines',
					line=dict(color='gray', width=3),
					showlegend=False,
					name='Bonds'
				))
	
	def create_property_plot(self, materials: List[Dict], x_prop: str, y_prop: str) -> go.Figure:
		"""Create scatter plot of material properties."""
		x_values = [self._convert_to_float(m["properties"][x_prop]) for m in materials]
		y_values = [self._convert_to_float(m["properties"][y_prop]) for m in materials]
		formulas = [m["formula"] for m in materials]
		stabilities = [self._convert_to_float(m["stability_score"]) for m in materials]
		
		fig = go.Figure()
		
		fig.add_trace(go.Scatter(
			x=x_values,
			y=y_values,
			mode='markers',
			marker=dict(
				size=10,
				color=stabilities,
				colorscale='Viridis',
				showscale=True,
				colorbar=dict(title="Stability Score")
			),
			text=formulas,
			hovertemplate=f'{x_prop}: %{{x:.2f}}<br>{y_prop}: %{{y:.2f}}<br>Formula: %{{text}}<extra></extra>',
			name='Materials'
		))
		
		fig.update_layout(
			title=f'{y_prop} vs {x_prop}',
			xaxis_title=x_prop,
			yaxis_title=y_prop,
			width=600,
			height=400
		)
		
		return fig
	
	def create_composition_plot(self, materials: List[Dict]) -> go.Figure:
		"""Create plot showing element composition distribution."""
		all_elements = set()
		for material in materials:
			all_elements.update(material["composition"].keys())
		
		element_counts = {element: 0 for element in all_elements}
		for material in materials:
			for element in material["composition"].keys():
				element_counts[element] += 1
		
		fig = go.Figure()
		fig.add_trace(go.Bar(
			x=list(element_counts.keys()),
			y=list(element_counts.values()),
			name='Element Frequency'
		))
		
		fig.update_layout(
			title='Element Composition Distribution',
			xaxis_title='Element',
			yaxis_title='Frequency',
			width=600,
			height=400
		)
		
		return fig
