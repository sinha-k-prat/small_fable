#!/usr/bin/env python3
"""
scale_study2.py — solving-vs-repair across scale, done RIGHT: a genuine MULTI-STEP task (2-hop capital)
+ robust PER-LAYER ACCURACY (argmax, sign-based -> immune to the lens-magnitude noise that collapsed
scale_study.py's per-example delta-gate). Forward-only, quantization-safe (model's own norm+lm_head).

2-HOP task: "{city} is a city in a country. The capital of that country is" -> gold = capital.
The model must chain city -> COUNTRY (bridge) -> CAPITAL. So per layer we track BOTH:
  capital accuracy  (gold = the answer)          -> forms LATE
  country  accuracy (bridge = the intermediate)  -> forms MID, then handed off to the capital
A non-trivial curve at last (unlike the binary tasks): the bridge rising then the answer overtaking it
is the multi-step signature, and the late capital gain over the mid is the REFINEMENT/repair stage.
1-HOP control: "The capital of {country} is" (direct recall; forms earlier, less refinement).

Per model we report the per-layer accuracy CURVE and:
  solve_mid   = capital acc at 50% depth          (the early/instinct answer)
  solve_final = capital acc at the last layer      (SOLVING)
  refine      = solve_final - solve_mid            (late-stage gain = REPAIR/second-hop completion)
  bridge_peak = best country acc over mid layers   (did it even retrieve the intermediate?)
  overwrite   = best_mid capital acc - final       (overthinking gap; >0 = had it, lost it)

VERDICT across sizes: does SOLVING (final) grow, does REFINE (final-mid) grow, and does the 2-hop
gap over the 1-hop control widen (multi-step needs more late refinement) -> the separable signal.

Colab:  pip install -q bitsandbytes accelerate
        python tools/scale_study2.py --models 0.5B 1.5B 3B 7B --out scale2
NOTE: per-layer ACCURACY (argmax) is robust to lens-magnitude miscalibration; a trained tuned lens
(Belrose) would sharpen intermediate faithfulness further but needs per-model training (deferred).
"""
import argparse, gc, json, traceback
import numpy as np

# city -> (country, capital).  city != capital so it is genuinely 2-hop; capitals are clean tokens.
TRIPLES = [
    ("Munich", "Germany", "Berlin"), ("Sydney", "Australia", "Canberra"),
    ("Barcelona", "Spain", "Madrid"), ("Milan", "Italy", "Rome"),
    ("Lyon", "France", "Paris"), ("Saint Petersburg", "Russia", "Moscow"),
    ("Busan", "South Korea", "Seoul"), ("Osaka", "Japan", "Tokyo"),
    ("Shanghai", "China", "Beijing"), ("Toronto", "Canada", "Ottawa"),
    ("Istanbul", "Turkey", "Ankara"), ("Zurich", "Switzerland", "Bern"),
    ("Lagos", "Nigeria", "Abuja"), ("Casablanca", "Morocco", "Rabat"),
    ("Auckland", "New Zealand", "Wellington"), ("Karachi", "Pakistan", "Islamabad"),
    ("Alexandria", "Egypt", "Cairo"), ("Thessaloniki", "Greece", "Athens"),
    ("Salzburg", "Austria", "Vienna"), ("Porto", "Portugal", "Lisbon"),
    ("Bergen", "Norway", "Oslo"), ("Krakow", "Poland", "Warsaw"),
    ("Rotterdam", "Netherlands", "Amsterdam"), ("Gothenburg", "Sweden", "Stockholm"),
    ("Antwerp", "Belgium", "Brussels"),
]


# One-shot exemplar (Boston->USA->Washington; NOT in TRIPLES, so no gold leaks) forces a DIRECT,
# single-token-style answer instead of the model restating "The capital is ...".
def _fewshot(task, city, country):
    sys = "<|im_start|>system\nAnswer with only the capital city name, nothing else.<|im_end|>\n"
    if task == "2hop":
        demo_q = "Boston is a city in a country. What is that country's capital?"
        q = f"{city} is a city in a country. What is that country's capital?"
    else:
        demo_q = "What is the capital of the United States?"
        q = f"What is the capital of {country}?"
    return (sys
            + f"<|im_start|>user\n{demo_q}<|im_end|>\n<|im_start|>assistant\nWashington<|im_end|>\n"
            + f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n")


def _match(tok, tid, target):
    """True if the model's top token is a (>=2 char) PREFIX of `target` -> robust to 'Mad'->Madrid."""
    s = tok.decode([int(tid)]).strip().lower()
    return len(s) >= 2 and target.lower().startswith(s)


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


def _first_id(tok, word):
    for s in (" " + word, word):
        ids = tok(s, add_special_tokens=False).input_ids
        if ids:
            return ids[0]
    return None


def _prefix_ids(tok, entities):
    """entity -> tensor of vocab ids whose (>=2-char) stripped form is a prefix of the entity. Handles
    however the model tokenizes a name ('Mad' for Madrid), and restricts the lens to relevant tokens."""
    import torch
    id2s = {}
    for t, i in tok.get_vocab().items():
        s = t.replace("Ġ", " ").replace("▁", " ").strip().lower()
        if len(s) >= 2:
            id2s[i] = s
    out = {}
    for e in entities:
        el = e.lower()
        ids = [i for i, s in id2s.items() if el.startswith(s)]
        out[e] = torch.tensor(ids) if ids else None
    return out


def per_layer_ranks(tok, model, task, pref):
    """Per layer return TWO curves over the triples:
      cap_lead : fraction where the CAPITAL outscores the bridge COUNTRY (restricted lens) -> the hop-2
                 handoff (country leads mid -> capital overtakes late); SMOOTH, visible mid-stack.
      cap_acc  : fraction where the global argmax prefix-matches the capital (strict 'would say it')."""
    import torch
    dev = next(model.parameters()).device
    base = model.model
    cap_lead, cap_acc, n = None, None, 0
    with torch.no_grad():
        for city, country, capital in TRIPLES:
            cpi, tpi = pref.get(capital), pref.get(country)
            if cpi is None or tpi is None:
                continue
            cpi, tpi = cpi.to(dev), tpi.to(dev)
            ids = tok(_fewshot(task, city, country), return_tensors="pt").input_ids.to(dev)
            hs = model(ids, output_hidden_states=True, use_cache=False).hidden_states
            L = len(hs)
            if cap_lead is None:
                cap_lead, cap_acc = np.zeros(L), np.zeros(L)
            for li, h in enumerate(hs):
                lo = model.lm_head(base.norm(h[:, -1, :])).float()[0]
                cap_lead[li] += float(lo[cpi].max() > lo[tpi].max())
                cap_acc[li] += _match(tok, int(lo.argmax()), capital)
            n += 1
    return cap_lead / max(n, 1), cap_acc / max(n, 1), n


def metrics(cl, ca):
    """cl = cap-leads-country curve (hop handoff), ca = strict argmax==capital curve (solving)."""
    L = len(ca) - 1
    mid = int(round(0.5 * L))
    cross = next((li / L for li in range(L + 1) if cl[li] > 0.5), -1.0)
    return {
        "curve_lead": [round(float(x), 3) for x in cl],
        "curve_acc": [round(float(x), 3) for x in ca],
        "solve_final": round(float(ca[-1]), 3),
        "solve_mid": round(float(ca[mid]), 3),
        "refine": round(float(ca[-1] - ca[mid]), 3),
        "lead_final": round(float(cl[-1]), 3),
        "bridge_min": round(float(cl[1:-1].min()) if L > 1 else float(cl[-1]), 3),
        "crossover_depth": round(float(cross), 3),
        "overwrite": round(float((ca[1:-1].max() if L > 1 else ca[-1]) - ca[-1]), 3),
    }


def run(models, out, force_4bit):
    import torch
    res, ok = {}, []
    for tag in models:
        name = f"Qwen/Qwen2.5-{tag}-Instruct"
        want_4bit = force_4bit or tag in ("3B", "7B", "14B", "32B")
        print(f"\n=== {name}  (4bit={want_4bit}) ===")
        try:
            tok, model = _load(name, want_4bit)
            ents = set()
            for _, co, cap in TRIPLES:
                ents.update([co, cap])
            pref = _prefix_ids(tok, ents)
            for task in ("2hop", "1hop"):
                cl, ca, n = per_layer_ranks(tok, model, task, pref)
                m = metrics(cl, ca); res[f"{tag}/{task}"] = m
                print(f"  [{task}] n={n} solve_final={m['solve_final']:.2f} solve_mid={m['solve_mid']:.2f} "
                      f"REFINE={m['refine']:+.2f} | bridge_min={m['bridge_min']:.2f} "
                      f"lead_final={m['lead_final']:.2f} crossover={m['crossover_depth']:.2f}")
            ok.append(tag)
            del model; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception:
            print(f"  !! {tag} FAILED:\n  " + traceback.format_exc().splitlines()[-1])
    json.dump(res, open(out + ".json", "w"), indent=2)
    _plot(res, ok, out + ".png")
    _verdict(res, ok)
    return res


def _trend(xs):
    xs = [x for x in xs if x is not None and not np.isnan(x)]
    return (xs[-1] - xs[0]) if len(xs) >= 2 else 0.0


def _verdict(res, models):
    g = lambda m, t, k: res.get(f"{m}/{t}", {}).get(k)
    print("\n" + "=" * 66 + "\nVERDICT  (2-hop multi-step task)\n" + "=" * 66)
    sf = [g(m, "2hop", "solve_final") for m in models]
    lf = [g(m, "2hop", "lead_final") for m in models]
    bm = [g(m, "2hop", "bridge_min") for m in models]
    cx = [g(m, "2hop", "crossover_depth") for m in models]
    print(f"  sizes              : {models}")
    print(f"  SOLVING (final acc): {sf}   <- does it actually output the capital")
    print(f"  capital>country end: {lf}   <- did the hop-2 handoff complete")
    print(f"  bridge_min (mid)   : {bm}   <- LOW = the country (bridge) genuinely leads mid-stack")
    print(f"  crossover depth    : {cx}   <- where capital overtakes country (-1 = never)")
    ds, dl = _trend(sf), _trend(lf)
    bridge_seen = any((b is not None and b < 0.45) for b in bm)
    print("\n  ---- conclusions ----")
    print(f"  Solving grows with scale?                    {'YES' if ds > 0.05 else 'no'}")
    print(f"  Bridge (country) actually forms mid-stack?   {'YES' if bridge_seen else 'no'}")
    print(f"  Hop-2 completion (capital>country) grows?    {'YES' if dl > 0.05 else 'no'}")
    if bridge_seen and dl > 0.05:
        print("  => the model retrieves the intermediate mid-stack and bigger models COMPLETE the second\n"
              "     hop more reliably -> a separable late-stage (refine/repair) skill that scales.")
    elif ds > 0.05 and not bridge_seen:
        print("  => looks like one-step recall, no visible mid-stage bridge -> not a repair story here.")
    else:
        print("  => weak/ambiguous signal; widen TRIPLES, try a harder multi-hop, or add a tuned lens.")


def _plot(res, models, png):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] skipped ({e})"); return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    cmap = plt.get_cmap("viridis")
    for ax, task in zip(axes, ["2hop", "1hop"]):
        for i, m in enumerate(models):
            r = res.get(f"{m}/{task}")
            if not r:
                continue
            xs = np.linspace(0, 1, len(r["curve_lead"]))
            c = cmap(i / max(len(models) - 1, 1))
            ax.plot(xs, r["curve_lead"], "-", color=c, label=f"{m} capital>country")
            ax.plot(xs, r["curve_acc"], ":", color=c, alpha=.55, label=f"{m} argmax=capital")
        ax.axhline(0.5, color="gray", lw=.6)
        ax.set_title(task); ax.set_xlabel("depth (layer fraction)")
        ax.set_ylabel("fraction"); ax.set_ylim(0, 1.02); ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.suptitle("2-hop handoff: capital>country (solid) should DIP mid (bridge leads) then rise late.  "
                 "Does the late completion scale?")
    fig.tight_layout(); fig.savefig(png, dpi=130); plt.close(fig)
    print(f"wrote {png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["0.5B", "1.5B", "3B", "7B"])
    ap.add_argument("--force_4bit", action="store_true")
    ap.add_argument("--out", default="scale2")
    a = ap.parse_args()
    run(a.models, a.out, a.force_4bit)


if __name__ == "__main__":
    main()
