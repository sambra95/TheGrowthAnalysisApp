"""Generate and save informational plots for popovers.

This module generates static PNG images for informational plots used in
popovers throughout the application. Run this script directly to regenerate
all info plots.
"""

import numpy as np
from pathlib import Path
import growthcurves.plot as gc_plot
from growthcurves.parametric import fit_parametric
from growthcurves.utils import extract_stats_from_fit


def create_annotation_demo_plot_png(save_path: Path):
    """Create and save a demo plot showing what each annotation represents."""
    # Create synthetic growth curve data
    t = np.linspace(0, 20, 100)
    # Logistic growth curve parameters
    K = 1.0  # carrying capacity
    r = 0.5  # growth rate
    N0 = 0.05  # initial population
    y = K / (1 + ((K - N0) / N0) * np.exp(-r * t))

    # Add some noise
    np.random.seed(42)
    y_noisy = y + np.random.normal(0, 0.02, len(y))

    # Fit a Richards model to the noisy data using growthcurves
    fit_result = fit_parametric(t, y_noisy, method="richards")

    # Extract growth stats from the fit object
    growth_stats = extract_stats_from_fit(fit_result, t, y_noisy)

    # Get annotation positions from growth_stats
    exp_start = growth_stats.get("exp_phase_start")
    exp_end = growth_stats.get("exp_phase_end")
    time_umax = growth_stats.get("time_at_umax")
    od_umax = growth_stats.get("od_at_umax")
    od_max = growth_stats.get("max_od")

    # Create base plot using growthcurves
    fig = gc_plot.create_base_plot(t, y_noisy, scale="linear")

    # Annotate plot with all features
    fig = gc_plot.annotate_plot(
        fig,
        phase_boundaries=(exp_start, exp_end) if exp_start and exp_end else None,
        time_umax=time_umax,
        od_umax=od_umax,
        od_max=od_max,
        umax_point=(time_umax, od_umax) if time_umax and od_umax else None,
        fitted_model=fit_result,
        scale="linear",
    )

    # Add text annotations/labels to highlight each feature
    # Build annotations dynamically based on what's available
    annotations = []

    if exp_start is not None:
        annotations.append(
            {
                "x": exp_start,
                "y": od_max if od_max else 0.95,
                "text": "Phase boundary<br>(exp start)",
                "showarrow": False,
                "xanchor": "left",
                "yanchor": "top",
                "bgcolor": "rgba(255, 255, 255, 0.8)",
                "bordercolor": "black",
                "borderwidth": 1,
            }
        )

    if exp_end is not None:
        annotations.append(
            {
                "x": exp_end,
                "y": od_max if od_max else 0.95,
                "text": "Phase boundary<br>(exp end)",
                "showarrow": False,
                "xanchor": "right",
                "yanchor": "top",
                "bgcolor": "rgba(255, 255, 255, 0.8)",
                "bordercolor": "black",
                "borderwidth": 1,
            }
        )

    if time_umax is not None:
        annotations.append(
            {
                "x": time_umax,
                "y": 0.1 if od_max else 0.1,
                "text": "Time at μmax",
                "showarrow": False,
                "xanchor": "center",
                "yanchor": "bottom",
                "bgcolor": "rgba(255, 255, 255, 0.8)",
                "bordercolor": "black",
                "borderwidth": 1,
            }
        )

    if od_umax is not None:
        annotations.append(
            {
                "x": 0.5,
                "y": od_umax,
                "text": "OD at μmax",
                "showarrow": False,
                "xanchor": "left",
                "yanchor": "middle",
                "bgcolor": "rgba(255, 255, 255, 0.8)",
                "bordercolor": "black",
                "borderwidth": 1,
            }
        )

    if od_max is not None:
        annotations.append(
            {
                "x": 19.5,
                "y": od_max,
                "text": "Max OD",
                "showarrow": False,
                "xanchor": "right",
                "yanchor": "middle",
                "bgcolor": "rgba(255, 255, 255, 0.8)",
                "bordercolor": "black",
                "borderwidth": 1,
            }
        )

    if time_umax is not None and od_umax is not None:
        annotations.append(
            {
                "x": time_umax - 2,
                "y": od_umax - (0.15 if od_max else 0.15),
                "text": "μmax point",
                "showarrow": True,
                "arrowhead": 2,
                "arrowsize": 1,
                "arrowwidth": 1.5,
                "arrowcolor": "red",
                "ax": 0,
                "ay": 0,
                "axref": "pixel",
                "ayref": "pixel",
                "bgcolor": "rgba(255, 255, 255, 0.8)",
                "bordercolor": "black",
                "borderwidth": 1,
            }
        )

    # Add fitted model curve label
    annotations.append(
        {
            "x": 10,
            "y": 0.85 if od_max else 0.85,
            "text": "Fitted model curve",
            "showarrow": True,
            "arrowhead": 2,
            "arrowsize": 1,
            "arrowwidth": 1.5,
            "arrowcolor": "black",
            "ax": 0,
            "ay": -30,
            "axref": "pixel",
            "ayref": "pixel",
            "bgcolor": "rgba(255, 255, 255, 0.8)",
            "bordercolor": "black",
            "borderwidth": 1,
        }
    )

    # Add all text annotations
    for annot in annotations:
        fig.add_annotation(annot)

    # Update layout
    fig.update_layout(
        title="Annotation Guide",
        xaxis_title="Time (hours)",
        yaxis_title="OD600 (baseline-corrected)",
        height=400,
        width=800,
        showlegend=False,
    )

    # Save as PNG
    fig.write_image(save_path, format="png", width=800, height=400, scale=2)


def save_all_info_plots():
    """Generate and save all informational plots as PNG files."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    info_plots_dir = project_root / "info_plots"

    # Create directory if it doesn't exist
    info_plots_dir.mkdir(exist_ok=True)

    # Generate and save annotation demo plot
    print("Generating annotation demo plot...")
    annotation_path = info_plots_dir / "annotation_demo.png"
    create_annotation_demo_plot_png(annotation_path)
    print(f"Saved annotation demo plot to {annotation_path}")

    print("\nAll info plots generated successfully!")


if __name__ == "__main__":
    save_all_info_plots()
