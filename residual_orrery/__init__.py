"""residual_orrery — a mechanistic-interpretability visualization.

The residual stream is rendered as an *orrery*: down_proj "writer-direction" stars
fixed on the unit sphere S^2, and a marker that hops through the residual trajectory,
lighting up each layer's stars as that layer fires. Qwen2.5 0.5B vs 1.5B side by side.

Module dependency graph (enforced):
    examples ─┐
    models  ──┼─► collect ──► project ──► animate
              └───────────────► compare ─────┘ ──► cli ──► __main__

`project` and `animate` are torch-free (numpy + sklearn / numpy + matplotlib +
imageio only) so they iterate fast on cached .npz without loading a model.
"""

from .examples import EXAMPLES, build_input_ids
from .project import fit_sphere_frame, project_run, SphereFrame, ProjectedRun
from .animate import (
    render_compare_gif,
    render_single_gif,
    GlowStyle,
    STYLE,
    slerp,
    build_schedule,
)

# models/collect/compare pull in torch+transformers; import lazily so that the
# torch-free path (project/animate on cached npz) stays importable even if a
# heavy dep hiccups. They are still importable directly as submodules.
try:  # pragma: no cover - exercised when torch present
    from .models import load_model, ModelBundle, MODEL_IDS
    from .collect import (
        collect_run,
        collect_all,
        collect_all_cached,
        RunCollection,
        NodeKind,
        TrajNode,
    )
    from .compare import build_compare, run_compare, CompareResult

    _TORCH_OK = True
except Exception as _exc:  # pragma: no cover
    _TORCH_OK = False
    _IMPORT_ERROR = _exc

__version__ = "0.1.0"

__all__ = [
    "EXAMPLES",
    "build_input_ids",
    "fit_sphere_frame",
    "project_run",
    "SphereFrame",
    "ProjectedRun",
    "render_compare_gif",
    "render_single_gif",
    "GlowStyle",
    "STYLE",
    "slerp",
    "build_schedule",
    "__version__",
]

if _TORCH_OK:
    __all__ += [
        "load_model",
        "ModelBundle",
        "MODEL_IDS",
        "collect_run",
        "collect_all",
        "collect_all_cached",
        "RunCollection",
        "NodeKind",
        "TrajNode",
        "build_compare",
        "run_compare",
        "CompareResult",
    ]
