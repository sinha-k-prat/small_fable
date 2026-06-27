"""Minimal installable metadata for the `residual_orrery` package.

`pip install -e .` from the repo root makes `import residual_orrery` and
`python -m residual_orrery ...` work. Runtime deps are intentionally loose so the
package installs cleanly on both the pinned legacy stack and a modern Colab.

ONE version-sensitive pin: matplotlib < 3.8. The 3D pane-blackening in
animate.py uses ``ax.w_xaxis`` / ``w_yaxis`` / ``w_zaxis``, which were removed in
matplotlib 3.8. Any 3.x < 3.8 (incl. Colab's 3.7) keeps that alias.
"""

from setuptools import setup, find_packages

setup(
    name="residual_orrery",
    version="0.1.0",
    description="Twin-sphere residual-stream orrery: Qwen2.5 0.5B vs 1.5B writer-direction routing.",
    packages=find_packages(include=["residual_orrery", "residual_orrery.*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.2",
        "transformers>=4.44",
        "numpy",
        "matplotlib<3.8",   # ax.w_xaxis alias removed in 3.8 (see animate._blacken_3d_axes)
        "scikit-learn",
        "Pillow",
        "imageio",
    ],
    entry_points={
        "console_scripts": [
            "residual-orrery = residual_orrery.cli:main",
        ],
    },
)
