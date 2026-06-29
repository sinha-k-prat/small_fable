#!/usr/bin/env python3
"""
probe_lens.py — THE GATE. Is "answer forms only late" a raw-lens ARTIFACT or real late computation?

For each model, at the answer position, we read each layer two ways:
  RAW LENS   : argmax of the model's own norm+lm_head prefix-matches the capital (what we had -> a CLIFF)
  PROBE      : a cross-validated linear probe trained to predict the CAPITAL (and the bridge COUNTRY)
               from that layer's residual -> "is the answer LINEARLY DECODABLE here, even if the raw
               lens can't read it?"  (a task-specific tuned lens; sklearn, no model training -> robust)

If the PROBE decodes the answer EARLY where the raw lens is flat -> the early/mid layers DO compute it
(in a concept space the unembedding can't read) -> the layers are NOT wasted and "late" was an artifact.
If even the probe is flat until the end -> the model genuinely computes late on this task.

Dataset: many cities per country (city != capital) so each capital/country is a real probe CLASS.
Prints the 3 pre-committed OUTCOMES. Defensive per-model (one failing is skipped).

Colab:  pip install -q bitsandbytes accelerate scikit-learn
        python tools/probe_lens.py --models 0.5B 1.5B 3B 7B --out probe
"""
import argparse, gc, json, traceback
import numpy as np

# capital -> ([non-capital cities in that country], country).  >=4 cities each -> probe has examples/class.
DATA = {
    "Berlin": (["Munich", "Hamburg", "Frankfurt", "Cologne", "Stuttgart"], "Germany"),
    "Paris": (["Lyon", "Marseille", "Nice", "Bordeaux", "Toulouse"], "France"),
    "Rome": (["Milan", "Naples", "Turin", "Florence", "Venice"], "Italy"),
    "Madrid": (["Barcelona", "Valencia", "Seville", "Bilbao", "Malaga"], "Spain"),
    "Tokyo": (["Osaka", "Kyoto", "Nagoya", "Yokohama", "Sapporo"], "Japan"),
    "Canberra": (["Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide"], "Australia"),
    "Ottawa": (["Toronto", "Vancouver", "Montreal", "Calgary", "Edmonton"], "Canada"),
    "Beijing": (["Shanghai", "Guangzhou", "Shenzhen", "Chengdu", "Wuhan"], "China"),
    "Ankara": (["Istanbul", "Izmir", "Bursa", "Antalya", "Adana"], "Turkey"),
    "Moscow": (["Saint Petersburg", "Novosibirsk", "Kazan", "Sochi", "Samara"], "Russia"),
}


def examples():
    out = []
    for cap, (cities, country) in DATA.items():
        for city in cities:
            out.append((city, country, cap))
    return out


def _fewshot(city):
    return ("<|im_start|>system\nAnswer with only the capital city name, nothing else.<|im_end|>\n"
            "<|im_start|>user\nBoston is a city in a country. What is that country's capital?<|im_end|>\n"
            "<|im_start|>assistant\nWashington<|im_end|>\n"
            f"<|im_start|>user\n{city} is a city in a country. What is that country's capital?<|im_end|>\n"
            "<|im_start|>assistant\n")


def _load(name, want_4bit):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    kw = dict(torch_dtype=torch.float16, device_map="auto")
    if want_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    return tok, AutoModelForCausalLM.from_pretrained(name, **kw).eval()


def _match(tok, tid, target):
    s = tok.decode([int(tid)]).strip().lower()
    return len(s) >= 2 and target.lower().startswith(s)


def collect(tok, model):
    """Return feats[L+1] = [n,d] residuals per layer, raw[L+1] = capital argmax-acc per layer,
    plus capital & country labels."""
    import torch
    dev = next(model.parameters()).device
    base = model.model
    ex = examples()
    feats, raw, ycap, ycty = None, None, [], []
    with torch.no_grad():
        for city, country, capital in ex:
            ids = tok(_fewshot(city), return_tensors="pt").input_ids.to(dev)
            hs = model(ids, output_hidden_states=True, use_cache=False).hidden_states
            L = len(hs)
            if feats is None:
                feats = [[] for _ in range(L)]; raw = np.zeros(L)
            for li, h in enumerate(hs):
                v = h[0, -1, :].float().cpu().numpy()
                feats[li].append(v)
                am = int(model.lm_head(base.norm(h[:, -1, :])).float()[0].argmax())
                raw[li] += _match(tok, am, capital)
            ycap.append(capital); ycty.append(country)
    n = len(ex)
    return [np.asarray(f) for f in feats], raw / n, np.array(ycap), np.array(ycty)


def probe_curve(feats, y):
    """5-fold CV linear-probe accuracy per layer (standardized + L2-regularized logistic)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    _, counts = np.unique(y, return_counts=True)
    nsplit = int(max(2, min(5, counts.min())))   # can't have more folds than smallest class
    cv = StratifiedKFold(n_splits=nsplit, shuffle=True, random_state=0)
    accs = []
    for X in feats:
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=0.5, max_iter=2000, multi_class="auto"))
        try:
            accs.append(float(cross_val_score(clf, X, y, cv=cv).mean()))
        except Exception:
            accs.append(float("nan"))
    return np.asarray(accs)


def _formation(curve, frac=0.6):
    """first layer-fraction where curve >= frac * its max (where the signal 'forms'); -1 if never."""
    c = np.nan_to_num(curve); L = len(c) - 1
    thr = frac * np.nanmax(c)
    for i in range(len(c)):
        if c[i] >= thr and c[i] > 1.5 / len(set(range(10))):  # above chance-ish
            return round(i / L, 3)
    return -1.0


def run(models, out, force_4bit):
    import torch
    res, ok = {}, []
    for tag in models:
        name = f"Qwen/Qwen2.5-{tag}-Instruct"
        want_4bit = force_4bit or tag in ("3B", "7B", "14B", "32B")
        print(f"\n=== {name} (4bit={want_4bit}) ===")
        try:
            tok, model = _load(name, want_4bit)
            feats, raw, ycap, ycty = collect(tok, model)
            p_cap = probe_curve(feats, ycap)
            res[tag] = {
                "raw_lens_cap": [round(float(x), 3) for x in raw],
                "probe_cap": [round(float(x), 3) for x in p_cap],
                "raw_form": _formation(raw), "probe_cap_form": _formation(p_cap),
                "probe_cap_max": round(float(np.nanmax(p_cap)), 3),
                "raw_final": round(float(raw[-1]), 3),
            }
            r = res[tag]
            print(f"  raw-lens capital forms @depth {r['raw_form']} (final {r['raw_final']:.2f}) | "
                  f"PROBE capital forms @depth {r['probe_cap_form']} (max {r['probe_cap_max']:.2f}) | "
                  f"(probe = task-specific tuned lens)")
            ok.append(tag)
            del model; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception:
            print(f"  !! {tag} FAILED:\n  " + traceback.format_exc().splitlines()[-1])
    json.dump(res, open(out + ".json", "w"), indent=2)
    _plot(res, ok, out + ".png")
    _verdict(res, ok)


def _verdict(res, models):
    print("\n" + "=" * 70 + "\nVERDICT — is 'computes late' a raw-lens ARTIFACT?\n" + "=" * 70)
    gaps = []
    for m in models:
        r = res.get(m)
        if not r:
            continue
        gap = (r["raw_form"] - r["probe_cap_form"]) if (r["raw_form"] >= 0 and r["probe_cap_form"] >= 0) else None
        gaps.append(gap)
        print(f"  {m}: raw-lens forms @{r['raw_form']}  vs  PROBE forms @{r['probe_cap_form']}  "
              f"(probe max {r['probe_cap_max']:.2f})")
    big = [g for g in gaps if g is not None and g > 0.15]
    decodable = any((res.get(m, {}).get("probe_cap_max", 0) > 0.5) for m in models)
    print("\n  ---- pre-committed outcome ----")
    if decodable and big:
        print("  (1) PROBE decodes the answer EARLY where the raw lens is flat -> early/mid layers DO\n"
              "      compute it (concept space); 'late' was a LENS ARTIFACT; layers NOT wasted.\n"
              "      => GO: measure solving-vs-repair with the probe, then the head hunt.")
    elif decodable:
        print("  (3) answer is decodable but not much earlier than the raw lens -> computation really is\n"
              "      late-ish; weak mid-stage. Try a harder multi-hop or accept 'solving scales'.")
    else:
        print("  (2) even a trained probe can't decode mid-stack -> the model genuinely computes LATE on\n"
              "      this task (coasting). This line is dead here -> PIVOT (annealed-scratchpad experiment).")


def _plot(res, models, png):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] {e}"); return
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, m in enumerate(models):
        r = res.get(m)
        if not r:
            continue
        xs = np.linspace(0, 1, len(r["probe_cap"]))
        c = cmap(i / max(len(models) - 1, 1))
        ax.plot(xs, r["probe_cap"], "-", color=c, label=f"{m} PROBE capital")
        ax.plot(xs, r["raw_lens_cap"], ":", color=c, alpha=.6, label=f"{m} raw-lens capital")
    ax.set_xlabel("depth (layer fraction)"); ax.set_ylabel("accuracy")
    ax.set_title("Probe (solid) vs raw lens (dotted): if the probe rises EARLY where the raw lens is\n"
                 "flat, the early layers compute the answer (concept space) -> 'late' was an artifact")
    ax.set_ylim(0, 1.02); ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(png, dpi=130); plt.close(fig)
    print(f"wrote {png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["0.5B", "1.5B", "3B", "7B"])
    ap.add_argument("--force_4bit", action="store_true")
    ap.add_argument("--out", default="probe")
    a = ap.parse_args()
    run(a.models, a.out, a.force_4bit)


if __name__ == "__main__":
    main()
