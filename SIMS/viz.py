"""
Visualization Helper Functions for Ultrasound Beamforming Simulations

This module provides shared plotting functions to create subplots for the generated
pressure fields. These functions use titles, captions (colorbar labels), and colormaps
supplied via lists. The function `plot_panels` uses zip to iterate over these collections,
ensuring that each subplot is configured as expected.

Usage Example:
    from SIMS.viz import plot_panels
    fig, axs = plot_panels(x, z, 
                           fields=[coherent_field, rss_field],
                           titles=["Coherent Field", "RSS Field"],
                           cb_labels=["Pressure (Pa)", "Pressure (Pa)"],
                           cmaps=["RdBu", "inferno"])
    plt.show()
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

def plot_panels(x, z, fields, titles, cb_labels, cmaps, figsize=(20, 8)):
    """
    Plot multiple field panels in a single figure.

    Args:
        x (ndarray): Lateral coordinate grid (in meters).
        z (ndarray): Axial coordinate grid (in meters).
        fields (list of ndarray): List of field arrays to plot.
        titles (list of str): List of title strings for each subplot.
        cb_labels (list of str): List of colorbar label strings for each subplot.
        cmaps (list of str): List of colormap names for each subplot.
        figsize (tuple): Figure size (default: (20, 8)).

    Returns:
        tuple: (fig, axs) where fig is the created matplotlib figure and axs is the axes array.
    """
    num_panels = len(fields)
    fig, axs = plt.subplots(1, num_panels, figsize=figsize)

    # Wrap a single-axis into a list for consistent iteration
    if num_panels == 1:
        axs = [axs]

    for ax, field, title, cb_label, cmap in zip(axs, fields, titles, cb_labels, cmaps):
        # Determine normalization and levels based on field data
        if np.any(field < 0):
            # Use TwoSlopeNorm for fields containing negative values (e.g., coherent field)
            limit = np.percentile(np.abs(field), 99)
            norm = TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit)
            levels = np.linspace(-limit, limit, 200)
        else:
            # For envelope fields (non-negative)
            limit = np.percentile(field, 99)
            norm = plt.Normalize(vmin=0, vmax=limit)
            levels = np.linspace(0, limit, 200)

        # Plot contourf on a grid scaled to mm
        cf = ax.contourf(x * 1000, z * 1000, field, levels=levels, cmap=cmap,
                         extend='both', norm=norm)
        # Add a contour at level 0 for clarity
        ax.contour(x * 1000, z * 1000, field, levels=[0],
                   colors='black', linestyles='--', linewidths=1)
        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel("Lateral Distance (mm)")
        ax.set_ylabel("Axial Distance (mm)")
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')
        fig.colorbar(cf, ax=ax, label=cb_label)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, axs 