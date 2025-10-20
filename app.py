"""
MaDiYs - Material Discovery Agent
Hugging Face Space Application with Gradio Interface
"""

import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import json
from typing import List, Dict, Tuple, Optional

# Add src to path
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from madiys.generation import MaterialGenerator, PropertyConstraints
from madiys.visualization import CrystalVisualizer
from madiys.chat import MaterialChatBot, parse_material_request


class MaDiYsApp:
    """Main application class for MaDiYs Hugging Face Space."""
    
    def __init__(self):
        self.chat_bot = MaterialChatBot()
        self.visualizer = CrystalVisualizer()
        self.generated_materials = []
        self.generator = None
    
    def process_chat_request(self, request: str) -> Tuple[str, str]:
        """Process natural language material request."""
        if not request.strip():
            return "Please enter a material request.", ""
        
        try:
            result = self.chat_bot.process_request(request)
            if result["success"]:
                constraints = result["constraints"]
                explanation = result["explanation"]
                
                # Convert constraints to display format
                constraints_text = f"""
**Parsed Constraints:**
- Allowed Elements: {constraints.allowed_elements or 'Any'}
- Forbidden Elements: {constraints.forbidden_elements or 'None'}
- Band Gap Range: {constraints.band_gap_range[0]:.1f} - {constraints.band_gap_range[1]:.1f} eV
- Density Range: {constraints.density_range[0]:.1f} - {constraints.density_range[1]:.1f} g/cm³
- Require Stable: {constraints.require_stable}
- Max Elements: {constraints.max_elements}
"""
                return explanation, constraints_text
            else:
                return "Failed to process request. Please try again.", ""
        except Exception as e:
            return f"Error processing request: {str(e)}", ""
    
    def generate_materials(self, 
                          allowed_elements: str,
                          forbidden_elements: str,
                          band_gap_min: float,
                          band_gap_max: float,
                          density_min: float,
                          density_max: float,
                          n_candidates: int,
                          use_chat_constraints: bool,
                          chat_constraints_json: str) -> Tuple[str, str, str]:
        """Generate materials based on constraints."""
        try:
            # Parse constraints
            if use_chat_constraints and chat_constraints_json:
                constraints_dict = json.loads(chat_constraints_json)
                constraints = PropertyConstraints(**constraints_dict)
            else:
                # Manual constraints
                allowed_list = [e.strip() for e in allowed_elements.split(",") if e.strip()]
                forbidden_list = [e.strip() for e in forbidden_elements.split(",") if e.strip()]
                
                constraints = PropertyConstraints(
                    allowed_elements=allowed_list if allowed_list else None,
                    forbidden_elements=forbidden_list if forbidden_list else None,
                    band_gap_range=(band_gap_min, band_gap_max),
                    density_range=(density_min, density_max),
                    require_stable=True
                )
            
            # Generate materials
            self.generator = MaterialGenerator(constraints)
            self.generated_materials = self.generator.generate_candidates(
                n_candidates=n_candidates,
                use_constraints=True
            )
            
            # Create results table
            if self.generated_materials:
                results_data = []
                for i, material in enumerate(self.generated_materials):
                    props = material["properties"]
                    results_data.append({
                        "ID": i + 1,
                        "Formula": material["formula"],
                        "Band Gap (eV)": f"{props['band_gap']:.3f}",
                        "Density (g/cm³)": f"{props['density']:.2f}",
                        "Formation Energy (eV/atom)": f"{props['formation_energy_per_atom']:.3f}",
                        "Stable": "Yes" if props['is_stable'] else "No",
                        "Stability Score": f"{material['stability_score']:.3f}"
                    })
                
                df = pd.DataFrame(results_data)
                results_table = df.to_string(index=False)
                
                # Statistics
                stats = self.generator.get_generation_stats()
                stats_text = f"""
**Generation Statistics:**
- Total Materials: {stats['total']}
- Average Band Gap: {stats['band_gap']['mean']:.2f} eV
- Average Density: {stats['density']['mean']:.2f} g/cm³
- Average Stability: {stats['stability']['mean']:.2f}
"""
                
                return f"✅ Generated {len(self.generated_materials)} materials successfully!", results_table, stats_text
            else:
                return "❌ No materials generated. Try relaxing constraints.", "", ""
                
        except Exception as e:
            return f"❌ Error generating materials: {str(e)}", "", ""
    
    def create_property_plot(self, x_prop: str, y_prop: str) -> str:
        """Create property scatter plot."""
        if not self.generated_materials:
            return "No materials generated yet."
        
        try:
            fig = self.visualizer.create_property_plot(self.generated_materials, x_prop, y_prop)
            return fig.to_html(include_plotlyjs='cdn')
        except Exception as e:
            return f"Error creating plot: {str(e)}"
    
    def create_crystal_visualization(self, material_idx: int) -> str:
        """Create 3D crystal structure visualization."""
        if not self.generated_materials or material_idx >= len(self.generated_materials):
            return "No materials available or invalid index."
        
        try:
            material = self.generated_materials[material_idx]
            fig = self.visualizer.visualize_crystal_structure(
                material["structure"],
                material["composition"],
                f"Crystal Structure: {material['formula']}"
            )
            return fig.to_html(include_plotlyjs='cdn')
        except Exception as e:
            return f"Error creating visualization: {str(e)}"
    
    def get_example_requests(self) -> List[str]:
        """Get example material requests."""
        return self.chat_bot.get_example_requests()


# Initialize app
app = MaDiYsApp()

# Create Gradio interface
def create_interface():
    """Create the main Gradio interface."""
    
    with gr.Blocks(title="MaDiYs - Material Discovery Agent", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🧪 MaDiYs - AI Material Discovery Agent
        
        Generate novel materials with specific properties using AI-driven design inspired by Microsoft's MatterGen.
        """)
        
        with gr.Tabs():
            # Chat Interface Tab
            with gr.Tab("💬 Chat Interface"):
                gr.Markdown("Describe your material requirements in natural language:")
                
                with gr.Row():
                    with gr.Column():
                        chat_input = gr.Textbox(
                            label="Material Request",
                            placeholder="e.g., 'Create a material with low dielectric constant currently used industrially'",
                            lines=3
                        )
                        chat_button = gr.Button("🤖 Process Request", variant="primary")
                        chat_output = gr.Textbox(label="Interpretation", lines=3, interactive=False)
                        constraints_output = gr.Textbox(label="Parsed Constraints", lines=8, interactive=False)
                    
                    with gr.Column():
                        gr.Markdown("### 💡 Example Requests:")
                        examples = app.get_example_requests()
                        for i, example in enumerate(examples[:5], 1):
                            gr.Markdown(f"{i}. {example}")
                
                chat_button.click(
                    app.process_chat_request,
                    inputs=[chat_input],
                    outputs=[chat_output, constraints_output]
                )
            
            # Material Generation Tab
            with gr.Tab("⚙️ Generate Materials"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Property Constraints")
                        
                        allowed_elements = gr.Textbox(
                            label="Allowed Elements (comma-separated)",
                            value="Fe,O,Ti,Al,Si,C,N",
                            placeholder="e.g., Fe,O,Ti,Al"
                        )
                        forbidden_elements = gr.Textbox(
                            label="Forbidden Elements (comma-separated)",
                            value="",
                            placeholder="e.g., H,Li,Be"
                        )
                        
                        with gr.Row():
                            band_gap_min = gr.Number(label="Band Gap Min (eV)", value=0.0)
                            band_gap_max = gr.Number(label="Band Gap Max (eV)", value=5.0)
                        
                        with gr.Row():
                            density_min = gr.Number(label="Density Min (g/cm³)", value=1.0)
                            density_max = gr.Number(label="Density Max (g/cm³)", value=15.0)
                        
                        n_candidates = gr.Slider(label="Number of Candidates", minimum=5, maximum=50, value=20)
                        
                        use_chat_constraints = gr.Checkbox(label="Use Chat Constraints", value=False)
                        chat_constraints_json = gr.Textbox(label="Chat Constraints JSON", visible=False)
                        
                        generate_button = gr.Button("🚀 Generate Materials", variant="primary")
                    
                    with gr.Column():
                        generation_status = gr.Textbox(label="Status", interactive=False)
                        results_table = gr.Textbox(label="Generated Materials", lines=15, interactive=False)
                        stats_output = gr.Textbox(label="Statistics", lines=8, interactive=False)
                
                generate_button.click(
                    app.generate_materials,
                    inputs=[
                        allowed_elements, forbidden_elements,
                        band_gap_min, band_gap_max,
                        density_min, density_max,
                        n_candidates, use_chat_constraints, chat_constraints_json
                    ],
                    outputs=[generation_status, results_table, stats_output]
                )
            
            # Visualization Tab
            with gr.Tab("📊 Visualizations"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Property Analysis")
                        
                        with gr.Row():
                            x_property = gr.Dropdown(
                                label="X Property",
                                choices=["band_gap", "density", "formation_energy_per_atom"],
                                value="band_gap"
                            )
                            y_property = gr.Dropdown(
                                label="Y Property",
                                choices=["band_gap", "density", "formation_energy_per_atom"],
                                value="density"
                            )
                        
                        plot_button = gr.Button("📈 Create Plot", variant="secondary")
                        property_plot = gr.HTML(label="Property Plot")
                    
                    with gr.Column():
                        gr.Markdown("### 3D Crystal Structure")
                        
                        material_idx = gr.Slider(
                            label="Material Index",
                            minimum=0,
                            maximum=49,
                            value=0,
                            step=1
                        )
                        
                        crystal_button = gr.Button("🔬 Visualize Crystal", variant="secondary")
                        crystal_plot = gr.HTML(label="3D Crystal Structure")
                
                plot_button.click(
                    app.create_property_plot,
                    inputs=[x_property, y_property],
                    outputs=[property_plot]
                )
                
                crystal_button.click(
                    app.create_crystal_visualization,
                    inputs=[material_idx],
                    outputs=[crystal_plot]
                )
            
            # About Tab
            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
                ## MaDiYs - Material Discovery Agent
                
                **MaDiYs** is an AI-driven material discovery agent inspired by Microsoft's MatterGen, 
                featuring property-constrained material generation, 3D crystal structure visualization, 
                and natural language chat interface.
                
                ### Key Features:
                - 🤖 **AI Material Generation**: Generate novel materials with specific property constraints
                - 💬 **Natural Language Interface**: Describe material requirements in plain English
                - 🔬 **3D Crystal Visualization**: Interactive 3D crystal structure viewer
                - 📊 **Property Analysis**: Comprehensive material property analysis
                - 🧪 **Experiment Tracking**: Log and track material discovery experiments
                - 🔗 **Materials Project Integration**: Access real materials data
                
                ### Technical Details:
                - **Framework**: Gradio + Streamlit
                - **ML Models**: RandomForest, Bayesian Optimization (GP)
                - **Visualization**: Plotly, 3D crystal structures
                - **Data Sources**: Materials Project API, custom datasets
                
                ### Usage:
                1. Use the **Chat Interface** to describe your material requirements
                2. Set **Property Constraints** manually or use chat-derived constraints
                3. **Generate Materials** and explore the results
                4. **Visualize** properties and crystal structures
                
                ### Citation:
                If you use this work in your research, please cite:
                ```bibtex
                @software{madiys2024,
                  title={MaDiYs: AI-Driven Material Discovery Agent},
                  author={Your Name},
                  year={2024},
                  url={https://huggingface.co/spaces/your-username/madiys}
                }
                ```
                """)
    
    return demo


# Create and launch the interface
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)
