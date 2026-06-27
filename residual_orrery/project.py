"""project.py — joint PCA(3) over the union + L2 sphere-normalize.

numpy + sklearn ONLY — NO torch. Operates purely on RunCollection-shaped data so it
iterates fast on cached .npz without loading a model.

One PCA(3) per model over the UNION of {all trajectory points (chosen + optionally the
other 4 prompts, fit-only), all drawn top-K writer columns, unembed rows} -> one shared
R^3 frame. Then L2-normalize each 3D vector to S^2.

Per-model frames are intentional: 0.5B (H=896) and 1.5B (H=1536) cannot share a PCA
basis without a fabricated alignment map. Side-by-side compares routing *shape*.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA


def _unit_rows(V, eps=1e-8):
    """L2-normalize each row of [*, H] to the unit hypersphere in R^H."""
    V = np.asarray(V, dtype=np.float32)
    n = np.linalg.norm(V, axis=-1, keepdims=True)
    return V / np.maximum(n, eps)


@dataclass
class SphereFrame:
    pca: PCA
    mean_: np.ndarray  # == pca.mean_, kept explicit
    explained_var_ratio: np.ndarray
    H: int

    def project(self, V):  # [*, H] -> [*, 3]
        # L2-normalize in R^H BEFORE PCA: Qwen2 residual stream has a few huge-norm
        # outlier dims that otherwise hijack the PCA mean/axes, collapsing the small-norm
        # down_proj writer columns to a single point. Normalizing first makes PCA capture
        # DIRECTIONAL structure so trajectory points AND writer stars both spread on S^2.
        V = np.asarray(V, dtype=np.float32)
        single = V.ndim == 1
        if single:
            V = V[None, :]
        Y = self.pca.transform(_unit_rows(V))
        return Y[0] if single else Y

    def project_sphere(self, V, eps=1e-8):
        Y = self.project(V)
        Y = np.atleast_2d(Y)
        n = np.linalg.norm(Y, axis=-1, keepdims=True)
        out = Y / np.maximum(n, eps)
        return out


@dataclass
class ProjectedRun:
    tag: str
    traj_sphere: np.ndarray  # [P, 3] on S^2, ordered hop path
    node_kinds: list  # parallel: (kind_str, layer)
    stars_sphere: dict  # layer -> [K, 3] writer stars on S^2
    stars_a: dict  # layer -> [K] raw |a_j| (glow magnitude; normed at render)
    unembed_sphere: np.ndarray  # [3] target star on S^2
    pred_token_str: str
    N: int
    topk: int
    # ---- v2 additive: carry generated answer + correctness onto the projection ----
    answer_text: str = ""
    is_correct: object = None    # True/False/None
    gold: str = ""


def _build_union(runs, subsample_cols, seed):
    """Stack [M, H] float32: all trajectory h, all drawn writer cols, all unembed dirs."""
    H = runs[0].H
    parts = []
    for rc in runs:
        for nd in rc.nodes:
            parts.append(np.asarray(nd.h, np.float32)[None, :])  # [1, H]
    col_parts = []
    for rc in runs:
        for L in sorted(rc.down_cols):
            col_parts.append(np.asarray(rc.down_cols[L], np.float32))  # [K, H]
    if col_parts:
        cols = np.concatenate(col_parts, axis=0)  # [sum K, H]
        if subsample_cols is not None and cols.shape[0] > subsample_cols:
            rng = np.random.RandomState(seed)
            sel = rng.choice(cols.shape[0], size=subsample_cols, replace=False)
            cols = cols[sel]
        parts.append(cols)
    for rc in runs:
        parts.append(np.asarray(rc.unembed_dir, np.float32)[None, :])
    X = np.concatenate(parts, axis=0).astype(np.float32)
    assert X.ndim == 2 and X.shape[1] == H, X.shape
    # match SphereFrame.project: fit PCA on L2-normalized (directional) rows.
    return _unit_rows(X)


def fit_sphere_frame(runs, subsample_cols=1500, seed=0):
    """Fit ONE PCA(3) over the union of everything drawn for this model."""
    if not isinstance(runs, (list, tuple)):
        runs = [runs]
    X = _build_union(runs, subsample_cols, seed)  # [M, H]
    n_comp = min(3, X.shape[0], X.shape[1])
    # svd_solver="full" is deterministic; do NOT pass random_state (only randomized uses it).
    pca = PCA(n_components=n_comp, svd_solver="full").fit(X)
    if n_comp < 3:  # pathological tiny case; pad components so transform yields 3-D
        pca = _pad_to_3(pca, X.shape[1])
    return SphereFrame(
        pca=pca,
        mean_=pca.mean_.copy().astype(np.float32),
        explained_var_ratio=pca.explained_variance_ratio_.copy(),
        H=runs[0].H,
    )


def _pad_to_3(pca, H):
    """Pad a <3-component PCA up to 3 components with zero rows (degenerate safety)."""
    comp = pca.components_
    k = comp.shape[0]
    if k >= 3:
        return pca
    pad = np.zeros((3 - k, H), dtype=comp.dtype)
    pca.components_ = np.concatenate([comp, pad], axis=0)
    evr = pca.explained_variance_ratio_
    pca.explained_variance_ratio_ = np.concatenate(
        [evr, np.zeros(3 - k, dtype=evr.dtype)]
    )
    ev = pca.explained_variance_
    pca.explained_variance_ = np.concatenate([ev, np.zeros(3 - k, dtype=ev.dtype)])
    pca.n_components_ = 3
    return pca


def project_run(frame, run):
    """Project a single RunCollection through ``frame`` onto S^2."""
    traj_H = np.stack([np.asarray(nd.h, np.float32) for nd in run.nodes], axis=0)  # [P,H]
    traj_sphere = frame.project_sphere(traj_H)  # [P, 3]
    assert traj_sphere.shape == (len(run.nodes), 3), traj_sphere.shape

    node_kinds = [(nd.kind.value, int(nd.layer)) for nd in run.nodes]

    stars_sphere, stars_a = {}, {}
    for L in sorted(run.down_cols):
        stars_sphere[L] = frame.project_sphere(run.down_cols[L])  # [K, 3]
        stars_a[L] = np.asarray(run.topk_a[L], np.float32)  # [K]
        assert stars_sphere[L].shape[0] == stars_a[L].shape[0]

    unembed_sphere = frame.project_sphere(run.unembed_dir[None, :])[0]  # [3]
    # visual identity: the UNEMBED trajectory node and the target star coincide.
    assert np.allclose(unembed_sphere, traj_sphere[-1], atol=1e-5), (
        "unembed node must coincide with target star",
    )

    return ProjectedRun(
        tag=run.tag,
        traj_sphere=traj_sphere,
        node_kinds=node_kinds,
        stars_sphere=stars_sphere,
        stars_a=stars_a,
        unembed_sphere=unembed_sphere,
        pred_token_str=run.pred_token_str,
        N=run.N,
        topk=run.topk,
        # duck-typed getattr so even a smoke run without these fields projects cleanly.
        answer_text=getattr(run, "answer_text", ""),
        is_correct=getattr(run, "is_correct", None),
        gold=getattr(run, "gold", ""),
    )
