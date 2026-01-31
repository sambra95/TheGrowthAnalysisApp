# Info Plots

This directory contains pre-generated PNG images used in informational popovers throughout the application.

## Regenerating Plots

To regenerate all info plots (e.g., after making changes to the plot styling or content), run:

```bash
python functions/info_plots.py
```

## Current Plots

- `annotation_demo.png`: Demonstrates the different annotation types available for well-level plots (phase boundaries, μmax point, fitted model curve, etc.)

## Why Pre-generate?

These plots are static educational content that don't change based on user data. Pre-generating them as PNG files:
- Improves page load performance
- Reduces computational overhead
- Ensures consistent appearance across sessions
