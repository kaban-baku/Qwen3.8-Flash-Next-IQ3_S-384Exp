"""群ごとに正規化した経路の重みから専門家を選ぶ。"""
import argparse
import json
import struct
from pathlib import Path

import numpy as np

N_LAYER, N_EXPERT = 48, 512


def load_scores(outdir):
    index = json.loads((outdir / "index.json").read_text(encoding="utf-8"))
    scores, counts = {}, {}
    for n, row in index.items():
        group = row["d"]
        scores.setdefault(group, np.zeros((N_LAYER, N_EXPERT), dtype=np.float64))
        counts.setdefault(group, 0)
        with (outdir / f"c{int(n):03d}.trace").open("rb") as f:
            while header := f.read(8):
                layer, k = struct.unpack("<ii", header)
                assert 0 <= layer < N_LAYER and 0 < k <= N_EXPERT
                ids = np.frombuffer(f.read(4*k), dtype="<i4")
                weights = np.frombuffer(f.read(4*k), dtype="<f4")
                assert len(ids) == len(weights) == k
                assert np.all((0 <= ids) & (ids < N_EXPERT))
                assert np.all((0 <= weights) & (weights <= 1)), "経路の重みが float32 の 0〜1 ではありません"
                # 2026-11-04 実測：このエンジン（0.1.52）の --dump-routing は重みを 0 で書く。重みが全部 0 のレコードは
                # 回数（1 専門家 1 票）で数える（REAP の「重み付き」ではなく頻度基準。Davd-b01 の freq、REAP-256 の imatrix 回数と同じ）
                np.add.at(scores[group][layer], ids, weights if weights.any() else np.ones(k))
                counts[group] += 1
    for group in scores:
        assert counts[group] > 0, f"{group}: レコードがありません"
        scores[group] /= counts[group] / N_LAYER
    return scores


def top(values, k):
    return np.argsort(-values, kind="stable")[:k].tolist()


def select(scores, weights, k, floor):
    if not 0 < k <= N_EXPERT or not 0 <= floor <= N_EXPERT:
        raise ValueError("K は 1〜512、floor は 0〜512")
    combined = sum(weights[g] * s for g, s in scores.items())
    layers, swaps = [], 0
    for layer in range(N_LAYER):
        mandatory = set().union(*(set(top(s[layer], floor)) for s in scores.values()))
        if len(mandatory) > k:
            raise ValueError(f"層 {layer}: 各群の上位 floor の和集合 {len(mandatory)} 個を K={k} に収められません")
        chosen = set(top(combined[layer], k))
        for expert in sorted(mandatory - chosen):
            candidates = chosen - mandatory
            if not candidates:
                raise ValueError(f"層 {layer}: 入れ替え先が足りません")
            victim = min(candidates, key=lambda e: (combined[layer, e], e))
            chosen.remove(victim)
            chosen.add(expert)
            swaps += 1
        layers.append(sorted(chosen))
    return layers, swaps


def masses(scores, layers):
    result = {}
    for group, score in scores.items():
        total = score.sum(axis=1)
        kept = np.array([score[l, es].sum() for l, es in enumerate(layers)])
        result[group] = np.divide(kept, total, out=np.zeros_like(total), where=total != 0)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("outdir", type=Path)
    ap.add_argument("-K", type=int, action="append", default=[])
    ap.add_argument("--weights", default="")
    ap.add_argument("--floor", type=int, default=16)
    ap.add_argument("--curve", action="store_true")
    a = ap.parse_args()
    if not a.K and not a.curve:
        ap.error("-K または --curve が必要です")
    scores = load_scores(a.outdir)
    weights = dict.fromkeys(scores, 1.0)
    weights.update({g: float(w) for g, w in (item.split("=") for item in a.weights.split(",") if item)})
    if a.curve:
        print("K | " + " | ".join(scores))
        print("---|" + "---|" * len(scores))
        for k in (128, 192, 256, 320, 384, 448):
            layers, _ = select(scores, weights, k, a.floor)
            mass = masses(scores, layers)
            print(str(k) + " | " + " | ".join(f"{mass[g].mean():.6f}" for g in scores))
    for k in a.K:
        layers, swaps = select(scores, weights, k, a.floor)
        data = {"n_expert": k, "layers": layers, "floor": a.floor, "weights": weights}
        (a.outdir / f"keep_{k}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        mass = masses(scores, layers)
        lines = [f"K={k}、入れ替え {swaps} 件", "", "群 | 0 | 8 | 16 | 24 | 32 | 40 | 47 | 全体の平均",
                 "---|---|---|---|---|---|---|---|---"]
        for g, values in mass.items():
            lines.append(g + " | " + " | ".join(f"{values[l]:.6f}" for l in (0,8,16,24,32,40,47))
                         + f" | {values.mean():.6f}")
        (a.outdir / f"report_{k}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"keep_{k}.json / report_{k}.md、入れ替え {swaps} 件")


if __name__ == "__main__":
    main()
