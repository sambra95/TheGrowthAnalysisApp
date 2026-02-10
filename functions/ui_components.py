"""Reusable UI components for Streamlit pages."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st


def page_header_with_help(title: str, help_text: str):
    """Render standard page header with help popover."""
    title_col, popover_col = st.columns([9, 2])
    with title_col:
        st.title(title)
    with popover_col:
        st.write("")
        with st.popover("Help", width="stretch"):
            st.markdown(help_text)


def render_sliding_window_viz():
    """Generate sliding window method visualization."""
    t_points = np.linspace(0, 48, 50)
    y_points = 0.05 + 0.95 / (1 + np.exp(-0.2 * (t_points - 24)))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t_points,
            y=y_points,
            mode="markers",
            marker=dict(color="blue", size=6),
            name="Data points",
        )
    )

    window_center_t = 24
    window_half_width = 4
    win_x0 = window_center_t - window_half_width
    win_x1 = window_center_t + window_half_width

    y_at_win = 0.05 + 0.95 / (1 + np.exp(-0.2 * (np.array([win_x0, win_x1]) - 24)))
    box_y_min = min(y_at_win) - 0.08
    box_y_max = max(y_at_win) + 0.08

    fig.add_shape(
        type="rect",
        x0=win_x0,
        x1=win_x1,
        y0=box_y_min,
        y1=box_y_max,
        fillcolor="rgba(0,200,0,0.2)",
        line=dict(color="green", width=2),
    )

    fig.add_annotation(
        x=window_center_t,
        y=box_y_max + 0.08,
        text="Sliding Window",
        showarrow=False,
        font=dict(color="green", size=12),
    )

    arrow_y = box_y_min + (box_y_max - box_y_min) / 2

    fig.add_annotation(
        x=win_x0 - 1,
        y=arrow_y,
        ax=win_x0 - 6,
        ay=arrow_y,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=2,
        arrowcolor="gray",
        text="",
    )

    fig.add_annotation(
        x=win_x1 + 6,
        y=arrow_y,
        ax=win_x1 + 1,
        ay=arrow_y,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=2,
        arrowcolor="gray",
        text="",
    )

    fig.update_layout(
        height=250,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_title="Time (h)",
        yaxis_title="ln(OD)",
        showlegend=False,
    )
    return fig


def render_spline_viz():
    """Generate spline method visualization."""
    t_spline_points = np.linspace(0, 48, 30)
    y_spline_points = np.log(0.05 + 0.95 / (1 + np.exp(-0.2 * (t_spline_points - 24))))

    t_spline_smooth = np.linspace(0, 48, 200)
    y_spline_smooth = np.log(0.05 + 0.95 / (1 + np.exp(-0.2 * (t_spline_smooth - 24))))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=t_spline_points,
            y=y_spline_points,
            mode="markers",
            marker=dict(color="blue", size=6),
            name="Data points",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=t_spline_smooth,
            y=y_spline_smooth,
            mode="lines",
            line=dict(color="green", width=3),
            name="Spline fit",
        )
    )

    fig.update_layout(
        height=250,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_title="Time (h)",
        yaxis_title="ln(OD)",
        showlegend=False,
    )
    return fig


def render_parametric_model_viz(model_type: str):
    """Generate parametric model visualization based on model type."""
    t = np.linspace(0, 48, 200)

    if "logistic" in str(model_type):
        y = 1.0 / (1 + np.exp(-0.15 * (t - 24)))
    elif "gompertz" in str(model_type):
        y = 1.0 * np.exp(-np.exp(-0.15 * (t - 24)))
    elif "richards" in str(model_type):
        nu = 2.0
        y = 1.0 / (1 + nu * np.exp(-0.15 * (t - 24))) ** (1 / nu)
    elif "baranyi" in str(model_type):
        lag_lambda = 5.0
        mu_max_b = 0.15
        K_b = 1.0
        y0_b = 0.05
        A_t = t + (1.0 / mu_max_b) * np.log(
            np.exp(-mu_max_b * t)
            + np.exp(-lag_lambda)
            - np.exp(-mu_max_b * t - lag_lambda)
        )
        y = K_b / (1.0 + ((K_b - y0_b) / y0_b) * np.exp(-mu_max_b * A_t))
    else:
        y = 1.0 / (1 + np.exp(-0.15 * (t - 24)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y, mode="lines", line=dict(color="blue", width=3)))
    fig.update_layout(
        height=200,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_title="Time (h)",
        yaxis_title="OD600",
        yaxis=dict(range=[0, 1.1]),
        showlegend=False,
    )
    return fig


def render_method_visualization(growth_method: str, model_type: str = None):
    """
    Render visualization for selected growth method with description and equation.

    Args:
        growth_method: One of "Sliding Window", "Spline", "Model Fitting"
        model_type: Model type for parametric methods (e.g., "mech_logistic")

    Returns:
        Plotly figure object or None
    """
    if growth_method == "Sliding Window":
        st.markdown("**Sliding Window Method** (Currently Selected)")
        st.latex(r"\ln(N(t)) = N_0 + b\,t")
        st.caption(
            "Local linear regression in moving windows. Calculates growth rate from nearby data points without assuming global curve shape."
        )
        return render_sliding_window_viz()

    elif growth_method == "Spline":
        st.markdown("**Spline Method** (Currently Selected)")
        st.latex(r"\ln(N(t)) = \mathrm{spline}(t)")
        st.caption(
            "Fitted smoothed curve without underlying shape assumptions. Flexible non-parametric approach."
        )
        return render_spline_viz()

    elif growth_method == "Model Fitting" and model_type:
        if "logistic" in str(model_type):
            st.markdown("**Logistic** (Currently Selected)")
            if str(model_type).startswith("mech_"):
                st.latex(r"\frac{dN}{dt} = \mu\left(1-\frac{N}{K}\right)N")
            else:
                st.latex(
                    r"\ln\!\left(\frac{N(t)}{N_0}\right) = \frac{A}{1+\exp\!\left(\frac{4\mu_{\max}(\lambda-t)}{A}+2\right)}"
                )
            st.caption(
                "Classic S-shaped curve with symmetric inflection point. Most commonly used for microbial growth."
            )

        elif "gompertz" in str(model_type):
            model_name = (
                "Modified Gompertz" if "modified" in str(model_type) else "Gompertz"
            )
            st.markdown(f"**{model_name}** (Currently Selected)")
            if str(model_type).startswith("mech_"):
                st.latex(r"\frac{dN}{dt} = \mu\log\!\left(\frac{K}{N}\right)N")
            elif "modified" in str(model_type):
                st.latex(
                    r"\ln\!\left(\frac{N(t)}{N_0}\right)=A\exp\!\left[-\exp\!\left(\frac{\mu_{\max}\exp(1)(\lambda-t)}{A}+1\right)\right]+A\exp\!\left(\alpha(t-t_{\mathrm{shift}})\right)"
                )
            else:
                st.latex(
                    r"\ln\!\left(\frac{N(t)}{N_0}\right)=A\exp\!\left[-\exp\!\left(\frac{\mu_{\max}\exp(1)(\lambda-t)}{A}+1\right)\right]"
                )
            st.caption(
                "Modified Gompertz with baseline offset y₀ and amplitude A = K − y₀. Asymmetric S-curve; often fits bacterial growth better than logistic."
            )

        elif "richards" in str(model_type):
            st.markdown("**Richards** (Currently Selected)")
            if str(model_type).startswith("mech_"):
                st.latex(
                    r"\frac{dN}{dt}=\mu\left(1-\left(\frac{N}{K}\right)^{\beta}\right)N"
                )
            else:
                st.latex(
                    r"\ln\!\left(\frac{N(t)}{N_0}\right)=A\left(1+\nu\exp\!\left(1+\nu+\frac{\mu_{\max}(1+\nu)^{1/\nu}(\lambda-t)}{A}\right)\right)^{-1/\nu}"
                )
            st.caption(
                "Generalized logistic with shape parameter ν. Most flexible - use when other models don't fit well."
            )

        elif "baranyi" in str(model_type):
            st.markdown("**Baranyi-Roberts** (Currently Selected)")
            st.latex(
                r"\frac{dN}{dt}=\mu\frac{\exp(\mu t)}{\exp(\lambda)-1+\exp(\mu t)}\left(1-\frac{N}{K}\right)N"
            )
            st.caption(
                "Baranyi-Roberts model with physiological lag parameter λ. Mechanistic model accounting for cell adaptation during lag phase."
            )

        return render_parametric_model_viz(model_type)

    return None


def render_phase_boundary_visualization(phase_boundary_method: str):
    """
    Render phase boundary method visualization with description and equation.

    Args:
        phase_boundary_method: One of "threshold" or "tangent"

    Returns:
        Path to image file
    """
    if phase_boundary_method == "threshold":
        st.markdown("**Threshold Method** (Currently Selected)")
        st.latex(r"\text{Lag end: } \mu(t) > f_{\text{lag}} \cdot \mu_{\max}")
        st.caption(
            "Uses threshold fractions of μ_max to identify phase transitions. Adjustable sensitivity via cutoff parameters."
        )
        return "info_plots/threshold_demo.png"
    else:  # tangent
        st.markdown("**Tangent Method** (Currently Selected)")
        st.latex(
            r"\text{Tangent at } \mu_{\max} \text{ intersects baseline and plateau}"
        )
        st.caption(
            "Geometric definition based on tangent line at maximum growth rate. No arbitrary thresholds required."
        )
        return "info_plots/tangent_demo.png"
