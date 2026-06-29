#!/usr/bin/env python3
"""
patch_test.py — THE CAUSAL settler. Is the mid-stack answer REAL (model computes it) or a probe
artifact (probe read the input)? Causal tracing (Meng et al. / ROME), per model, across scale.

Per example (city -> country -> capital):
  CLEAN     : run normally, cache the last-token residual at every layer; record if output = capital.
  CORRUPTED : add noise to the CITY token embeddings so the model can't identify the country -> the
              capital answer is destroyed (output wrong).
  PATCH(L)  : redo the CORRUPTED run but overwrite the last-token residual at layer L with the CLEAN
              one, let the rest of the model run, and check if the correct capital is RESTORED.

restoration[L] = P(correct capital | corrupted + clean-patch at L). The layer where restoration RISES
is where the answer is CAUSALLY carried. If a MID layer restores it -> the answer is genuinely computed
mid-stack (NOT the probe reading the input) -> kills the confound. The onset depth vs model size is the
scale story (does the big model compute it earlier).

Controls: clean_acc (upper ref), corrupted_acc (lower ref ~0). Defensive per-model. 4-bit for 3B/7B.

Colab:  pip install -q bitsandbytes accelerate
        python tools/patch_test.py --models 0.5B 1.5B 3B 7B --noise 3.0 --out patch
"""
import argparse, gc, json, traceback
import numpy as np

DATA = {
    "Berlin": (["Munich", "Hamburg", "Frankfurt"], "Germany"),
    "Paris": (["Lyon", "Marseille", "Nice"], "France"),
    "Rome": (["Milan", "Naples", "Turin"], "Italy"),
    "Madrid": (["Barcelona", "Valencia", "Seville"], "Spain"),
    "Tokyo": (["Osaka", "Kyoto", "Nagoya"], "Japan"),
    "Ottawa": (["Toronto", "Vancouver", "Montreal"], "Canada"),
    "Beijing": (["Shanghai", "Guangzhou", "Shenzhen"], "China"),
    "Moscow": (["Novosibirsk", "Kazan", "Sochi"], "Russia"),
}


def examples():
    return [(city, country, cap) for cap, (cities, country) in DATA.items() for city in cities]


def _prompt(city):
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


def _city_positions(tok, prompt, city):
    enc = tok(prompt, return_offsets_mapping=True, add_special_tokens=False)
    start = prompt.rfind(city); end = start + len(city)
    return [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < end and b > start], enc["input_ids"]


def trace_one(tok, model, city, country, capital, noise, gen):
    """Return (clean_ok, corr_ok, restoration[L+1]) for one example."""
    import torch
    dev = next(model.parameters()).device
    emb = model.get_input_embeddings()
    prompt = _prompt(city)
    cpos, ids = _city_positions(tok, prompt, city)
    ids = torch.tensor([ids], device=dev)
    base_emb = emb(ids)                                   # [1,T,d]
    # fixed corruption for this example (same noise across the layer sweep)
    sd = base_emb.std().item()
    noise_vec = torch.zeros_like(base_emb)
    if cpos:
        g = torch.Generator(device="cpu").manual_seed(hash(city) % (2**31))
        n = torch.randn(base_emb.shape, generator=g).to(dev) * (noise * sd)
        for p in cpos:
            noise_vec[0, p] = n[0, p]
    corr_emb = base_emb + noise_vec

    def cap_ok(logits):
        return _match(tok, int(logits[0, -1].argmax()), capital)

    with torch.no_grad():
        clean = model(inputs_embeds=base_emb, output_hidden_states=True, use_cache=False)
        clean_ok = cap_ok(clean.logits)
        clean_last = [h[0, -1, :].detach().clone() for h in clean.hidden_states]   # per layer
        corr = model(inputs_embeds=corr_emb, use_cache=False)
        corr_ok = cap_ok(corr.logits)

        L = len(clean_last)
        restoration = np.zeros(L)
        layers = model.model.layers
        for j in range(len(layers)):                      # patch OUTPUT of decoder layer j == hs[j+1]
            handle = None
            def hook(mod, inp, out, _j=j):
                t = out[0] if isinstance(out, tuple) else out
                t[:, -1, :] = clean_last[_j + 1]
                return out
            handle = layers[j].register_forward_hook(hook)
            try:
                pl = model(inputs_embeds=corr_emb, use_cache=False).logits
                restoration[j + 1] = cap_ok(pl)
            finally:
                handle.remove()
        restoration[0] = corr_ok                          # layer-0 slot = no patch baseline
    return clean_ok, corr_ok, restoration


def run(models, noise, out, force_4bit):
    import torch
    res, ok = {}, []
    ex = examples()
    for tag in models:
        name = f"Qwen/Qwen2.5-{tag}-Instruct"
        want_4bit = force_4bit or tag in ("3B", "7B", "14B", "32B")
        print(f"\n=== {name} (4bit={want_4bit}) ===")
        try:
            tok, model = _load(name, want_4bit)
            clean_acc, corr_acc, rest = [], [], None
            for city, country, capital in ex:
                co, cr, r = trace_one(tok, model, city, country, capital, noise, None)
                clean_acc.append(co); corr_acc.append(cr)
                rest = r if rest is None else rest + r
            rest = rest / len(ex)
            onset = next((j / (len(rest) - 1) for j in range(len(rest)) if rest[j] > 0.5 * rest.max()), -1.0)
            res[tag] = {
                "clean_acc": round(float(np.mean(clean_acc)), 3),
                "corrupted_acc": round(float(np.mean(corr_acc)), 3),
                "restoration": [round(float(x), 3) for x in rest],
                "restore_max": round(float(rest.max()), 3),
                "restore_onset_depth": round(float(onset), 3),
            }
            r = res[tag]
            print(f"  clean={r['clean_acc']:.2f} corrupted={r['corrupted_acc']:.2f} | "
                  f"restoration max={r['restore_max']:.2f} onset@depth {r['restore_onset_depth']}")
            ok.append(tag)
            del model; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception:
            print(f"  !! {tag} FAILED:\n  " + traceback.format_exc().splitlines()[-1])
    json.dump(res, open(out + ".json", "w"), indent=2)
    _plot(res, ok, out + ".png")
    _verdict(res, ok)


def _verdict(res, models):
    print("\n" + "=" * 70 + "\nVERDICT — is the mid-stack answer CAUSAL (real) or a probe artifact?\n" + "=" * 70)
    for m in models:
        r = res.get(m)
        if not r:
            continue
        print(f"  {m}: clean={r['clean_acc']:.2f} corrupted={r['corrupted_acc']:.2f}  "
              f"restoration onset@depth {r['restore_onset_depth']} (max {r['restore_max']:.2f})")
    mids = [res[m]["restore_onset_depth"] for m in models if m in res and 0 <= res[m]["restore_onset_depth"] < 0.7]
    print("\n  ---- pre-committed outcome ----")
    if mids:
        print("  CAUSAL: patching a MID layer restores the correct capital -> the answer is genuinely\n"
              "  computed mid-stack (NOT the probe reading the input). The probe result is REAL. Onset\n"
              "  depth vs size is the scale story. => proceed to the preservation (steering) test.")
    else:
        print("  restoration only at the very end -> the answer is NOT causally carried mid-stack; the\n"
              "  probe was likely reading the input -> the scale benefit is plain SOLVING, not repair.")


def _plot(res, models, png):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] {e}"); return
    cmap = plt.get_cmap("plasma")
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, m in enumerate(models):
        r = res.get(m)
        if not r:
            continue
        xs = np.linspace(0, 1, len(r["restoration"]))
        ax.plot(xs, r["restoration"], "-o", ms=3, color=cmap(i / max(len(models) - 1, 1)),
                label=f"{m} (clean {r['clean_acc']:.2f})")
    ax.set_xlabel("layer patched (depth fraction)"); ax.set_ylabel("restoration (P correct capital)")
    ax.set_title("Causal tracing: patch CLEAN mid-layer activation into a CORRUPTED run.\n"
                 "Restoration rising mid-stack = the answer is CAUSALLY computed there (not a probe artifact)")
    ax.set_ylim(0, 1.02); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(png, dpi=130); plt.close(fig)
    print(f"wrote {png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["0.5B", "1.5B", "3B", "7B"])
    ap.add_argument("--noise", type=float, default=3.0, help="city-embedding corruption (x emb std)")
    ap.add_argument("--force_4bit", action="store_true")
    ap.add_argument("--out", default="patch")
    a = ap.parse_args()
    run(a.models, a.noise, a.out, a.force_4bit)


if __name__ == "__main__":
    main()
