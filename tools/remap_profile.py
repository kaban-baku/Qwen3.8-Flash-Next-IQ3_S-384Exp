"""Strata の専門家プロファイル（512 専門家の全ペアの順位）を刈り込み後の番号に写し替える。
残さない専門家のペアは落とし、順位はそのまま詰める。--traces を付けると、写した記録の頻度順を先頭に置く（make_profile の --reorder と同じ）。
usage: python remap_profile.py <keep_K.json> <out.bin> [--base data/expert-profile.bin] [--traces DIR]"""
import argparse, json, struct, sys
import os
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.environ.get("STRATA_DIR", r"C:\Strata"), "tools"))
import make_profile as mp  # read_profile / read_trace / write_profile / rank_profile

ap = argparse.ArgumentParser()
ap.add_argument("keep"); ap.add_argument("out")
ap.add_argument("--base", default=os.path.join(os.environ.get("STRATA_DIR", r"C:\Strata"), "data", "expert-profile.bin"))
ap.add_argument("--traces", default=None, help="写し替え済みの記録のフォルダ（traces_remap_K）")
a = ap.parse_args()

keep = json.loads(Path(a.keep).read_text(encoding="utf-8-sig"))
K = keep["n_expert"]
new_index = [{e: i for i, e in enumerate(layer)} for layer in keep["layers"]]
base = mp.read_profile(a.base, 512)
mapped = [(l, new_index[l][e]) for l, e in base if e in new_index[l]]
freq = defaultdict(int)
if a.traces:
    for t in sorted(Path(a.traces).glob("*.trace")):
        for p, c in mp.read_trace(t, K).items():
            freq[p] += c
ranked, counts = mp.rank_profile(mapped, freq, K, no_base=False, reorder=bool(a.traces))
mp.write_profile(a.out, ranked, K)
assert mp.read_profile(a.out, K) == ranked
print(f"{a.out}: {len(ranked)} ペア（記録 {counts['trace']}・付属 {counts['base']}・補完 {counts['fill']}）、付属の {len(base)} ペアのうち {len(mapped)} が残る")
