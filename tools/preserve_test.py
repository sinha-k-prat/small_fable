#!/usr/bin/env python3
"""
preserve_test.py — THE FINAL TEST. On the questions the model gets WRONG, did it NEARLY have the
right answer (preservation / repair) or was the right answer nowhere (skill)?

We restrict to the candidate capitals (prefix-token sets -> robust to tokenization). For each example
we rank all capitals by the model's final-layer logit. On WRONG examples (the model's top capital is
not the correct one), we ask: where does the CORRECT capital rank among the candidates?
  - correct is a consistent RUNNER-UP (median rank ~2, far below chance) -> the model FAVORED the truth
    and barely missed -> PRESERVATION / repair ("nearly said it").
  - correct sits at ~CHANCE rank -> the model had no preference for the truth -> SKILL (computed wrong).

Control: chance rank = (n_caps+1)/2. We also report the median rank of a RANDOM non-correct capital
(should be ~chance). Forward-only, quantization-safe, defensive per-model.

Colab:  pip install -q bitsandbytes accelerate
        python tools/preserve_test.py --models 0.5B 1.5B 3B 7B --out preserve
"""
import argparse, gc, json, traceback
import numpy as np

DATA = {  # capital -> ([non-capital cities], country)
    "Berlin": (["Munich", "Hamburg", "Frankfurt", "Cologne"], "Germany"),
    "Paris": (["Lyon", "Marseille", "Nice", "Bordeaux"], "France"),
    "Rome": (["Milan", "Naples", "Turin", "Florence"], "Italy"),
    "Madrid": (["Barcelona", "Valencia", "Seville", "Bilbao"], "Spain"),
    "Tokyo": (["Osaka", "Kyoto", "Nagoya", "Yokohama"], "Japan"),
    "Ottawa": (["Toronto", "Vancouver", "Montreal", "Calgary"], "Canada"),
    "Beijing": (["Shanghai", "Guangzhou", "Shenzhen", "Chengdu"], "China"),
    "Moscow": (["Novosibirsk", "Kazan", "Sochi", "Samara"], "Russia"),
    "Ankara": (["Istanbul", "Izmir", "Bursa", "Antalya"], "Turkey"),
    "Canberra": (["Sydney", "Melbourne", "Brisbane", "Perth"], "Australia"),
}
CAPITALS = list(DATA.keys())


def examples():
    return [(city, cap) for cap, (cities, _) in DATA.items() for city in cities]


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


def _prefix_ids(tok):
    import torch
    id2s = {}
    for t, i in tok.get_vocab().items():
        s = t.replace("Ġ", " ").replace("▁", " ").strip().lower()
        if len(s) >= 2:
            id2s[i] = s
    out = {}
    for cap in CAPITALS:
        cl = cap.lower()
        ids = [i for i, s in id2s.items() if cl.startswith(s)]
        out[cap] = torch.tensor(ids) if ids else None
    return out


def run(models, out, force_4bit):
    import torch
    res, ok = {}, []
    ex = examples()
    rng = np.random.default_rng(0)
    n_caps = len(CAPITALS)
    chance = (n_caps + 1) / 2.0
    for tag in models:
        name = f"Qwen/Qwen2.5-{tag}-Instruct"
        want_4bit = force_4bit or tag in ("3B", "7B", "14B", "32B")
        print(f"\n=== {name} (4bit={want_4bit}) ===")
        try:
            tok, model = _load(name, want_4bit)
            pref = _prefix_ids(tok)
            dev = next(model.parameters()).device
            correct_ranks, random_ranks, top_correct = [], [], []
            with torch.no_grad():
                for city, capital in ex:
                    ids = tok(_prompt(city), return_tensors="pt").input_ids.to(dev)
                    lo = model(ids, use_cache=False).logits[0, -1].float()
                    scores = {c: float(lo[pref[c].to(dev)].max()) for c in CAPITALS if pref[c] is not None}
                    order = sorted(scores, key=scores.get, reverse=True)  # best first
                    rank = order.index(capital) + 1
                    is_top = (order[0] == capital)
                    top_correct.append(is_top)
                    if not is_top:                       # WRONG: model's top capital != correct
                        correct_ranks.append(rank)
                        others = [c for c in CAPITALS if c != capital]
                        rc = rng.choice(others)
                        random_ranks.append(order.index(rc) + 1)
            acc = float(np.mean(top_correct))
            med_correct = float(np.median(correct_ranks)) if correct_ranks else None
            med_random = float(np.median(random_ranks)) if random_ranks else None
            runner_up = float(np.mean([r == 2 for r in correct_ranks])) if correct_ranks else None
            res[tag] = {"capital_acc": round(acc, 3), "n_wrong": len(correct_ranks),
                        "chance_rank": round(chance, 2),
                        "median_correct_rank_when_wrong": med_correct,
                        "median_random_rank_when_wrong": med_random,
                        "frac_correct_is_runnerup": runner_up}
            r = res[tag]
            print(f"  capital_acc={acc:.2f}  wrong={r['n_wrong']} | when WRONG: correct rank "
                  f"median={med_correct} vs chance={chance:.1f} vs random={med_random} | "
                  f"runner-up frac={runner_up}")
            ok.append(tag)
            del model; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception:
            print(f"  !! {tag} FAILED:\n  " + traceback.format_exc().splitlines()[-1])
    json.dump(res, open(out + ".json", "w"), indent=2)
    _plot(res, ok, out + ".png", chance)
    _verdict(res, ok, chance)


def _verdict(res, models, chance):
    print("\n" + "=" * 70 + "\nVERDICT — on WRONG answers: did the model NEARLY have the truth?\n" + "=" * 70)
    leans = []
    for m in models:
        r = res.get(m)
        if not r or r["median_correct_rank_when_wrong"] is None:
            continue
        mc = r["median_correct_rank_when_wrong"]
        leans.append(mc)
        verdict = "PRESERVATION (nearly had it)" if mc <= chance - 1.0 else "SKILL (no preference)"
        print(f"  {m}: when wrong, correct ranks median {mc} (chance {chance:.1f}, random "
              f"{r['median_random_rank_when_wrong']})  -> {verdict}")
    print("\n  ---- pre-committed outcome ----")
    if leans and np.median(leans) <= chance - 1.0:
        print("  PRESERVATION: on wrong answers the correct capital is a consistent RUNNER-UP (well below\n"
              "  chance) -> the model FAVORED the truth and barely missed -> it 'nearly said it'. The\n"
              "  knows-but-doesn't-say / repair story holds. => build the anti-overwrite regularizer.")
    else:
        print("  SKILL: on wrong answers the correct capital sits near CHANCE rank (no better than a random\n"
              "  capital) -> the model had NO preference for the truth -> it genuinely computed the wrong\n"
              "  answer. The repair story does NOT hold; scale buys SKILL. => report the honest negative,\n"
              "  pivot to the annealed-scratchpad experiment.")


def _plot(res, models, png, chance):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] {e}"); return
    xs = list(range(len(models)))
    mc = [res.get(m, {}).get("median_correct_rank_when_wrong", np.nan) for m in models]
    mr = [res.get(m, {}).get("median_random_rank_when_wrong", np.nan) for m in models]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(xs, mc, "-o", color="tab:green", label="correct capital rank (when wrong)")
    ax.plot(xs, mr, "-o", color="tab:gray", label="random capital rank (control)")
    ax.axhline(chance, color="red", ls="--", label=f"chance = {chance:.1f}")
    ax.set_xticks(xs); ax.set_xticklabels(models); ax.set_xlabel("model size")
    ax.set_ylabel("rank among capitals (lower = nearer the top)")
    ax.set_title("On WRONG answers: is the correct capital a runner-up (preservation) or at chance (skill)?\n"
                 "green well below red = preservation; green ≈ red ≈ chance = skill")
    ax.invert_yaxis(); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(png, dpi=130); plt.close(fig)
    print(f"wrote {png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["0.5B", "1.5B", "3B", "7B"])
    ap.add_argument("--force_4bit", action="store_true")
    ap.add_argument("--out", default="preserve")
    a = ap.parse_args()
    run(a.models, a.out, a.force_4bit)


if __name__ == "__main__":
    main()
