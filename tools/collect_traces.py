"""文集を符号化し、1 本ずつ経路を記録する。"""
import argparse
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.environ.get("STRATA_DIR", r"C:\Strata"), "tools"))
from collect_profile_traces import drop, load_tokenizer


def encode_prompt(row):
    text = ""
    if row.get("system"):
        text += f"<|im_start|>system\n{row['system']}<|im_end|>\n"
    text += f"<|im_start|>user\n{row['p']}<|im_end|>\n<|im_start|>assistant\n"
    if not row.get("think", False):
        text += "<think>\n\n</think>\n\n"
    return text


def record_count(path):
    count = 0
    with path.open("rb") as f:
        while header := f.read(8):
            layer, k = struct.unpack("<ii", header)
            assert 0 <= layer < 48 and 0 < k <= 512
            ids = f.read(4 * k)
            weights = struct.unpack(f"<{k}f", f.read(4 * k))
            assert len(ids) == 4 * k
            assert all(0 <= w <= 1 for w in weights), "経路の重みが float32 の 0〜1 ではありません"
            count += 1
    return count


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--corpus", default="corpus.json")
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--only", help="カンマ区切りのプロンプト番号")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8-sig"))
    corpus = json.loads(Path(a.corpus).read_text(encoding="utf-8-sig"))
    tok = load_tokenizer(cfg["tokenizer"])
    a.outdir.mkdir(parents=True, exist_ok=True)
    index_path = a.outdir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    only = {int(n) for n in a.only.split(",")} if a.only else None
    args = drop(drop(drop(cfg["args"], "--serve"), "--batch", True), "--prefill-head-alias")   # 一発実行（generate）では --prefill-head-alias が使えない（2026-11-04 実測）
    env = dict(os.environ, **{k: str(v) for k, v in cfg.get("env", {}).items()})
    for n, row in enumerate(corpus):
        if only is not None and n not in only:
            continue
        trace = (a.outdir / f"c{n:03d}.trace").resolve()
        tf = (a.outdir / f"c{n:03d}.tok").resolve()
        ids = tok.encode(encode_prompt(row), parse_special=True)
        if not trace.exists():
            tf.write_text(",".join(map(str, ids)), encoding="utf-8")
            print(f"{n}: {row['d']} プロンプト {len(ids)} トークン", flush=True)
            if a.dry_run:
                continue
            cmd = [cfg["exe"]] + args + ["--tokens-file", str(tf), "--max-new", str(a.max_new),
                                        "--greedy", "--dump-routing", str(trace)]
            result = subprocess.run(cmd, cwd=cfg.get("cwd") or None, env=env,
                                    capture_output=True, text=True)
            if result.returncode or not trace.exists():
                raise SystemExit(f"失敗 (rc {result.returncode}): {result.stderr[-1000:]}")
        if not a.dry_run:
            index[str(n)] = {"d": row["d"], "prompt_tokens": len(ids),
                             "generated_tokens": record_count(trace) / 48}
            index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
