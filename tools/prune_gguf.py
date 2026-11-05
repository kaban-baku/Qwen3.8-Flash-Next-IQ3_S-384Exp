"""GGUF の専門家とルーターの行をバイト単位で切り出す。"""
import argparse
import os
import hashlib
import json
import random
import re
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.environ.get("STRATA_DIR", r"C:\Strata"), "tools"))
from gguf_reader import GGUFFile

BUFFER = 8 * 1024 * 1024
EXPERT = re.compile(r"^blk\.(\d+)\.ffn_(?:gate|up|down)_exps\.weight$")
ROUTER = re.compile(r"^blk\.(\d+)\.ffn_gate_inp\.weight$")


def align(n, alignment):
    return (n + alignment - 1) // alignment * alignment


def string_bytes(text):
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def metadata_bytes(path, k):
    """メタデータの値を走査し、expert_count の値の位置だけを書き換える。"""
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    with path.open("rb") as f:
        magic, version, nt, nk = struct.unpack("<IIQQ", f.read(24))
        assert magic == 0x46554747 and version == 3

        def string():
            length, = struct.unpack("<Q", f.read(8))
            return f.read(length)

        def skip(type_id):
            if type_id == 8:
                length, = struct.unpack("<Q", f.read(8))
                f.seek(length, 1)
            elif type_id == 9:
                elem, count = struct.unpack("<IQ", f.read(12))
                if elem == 8:
                    for _ in range(count):
                        skip(elem)
                else:
                    f.seek(sizes[elem] * count, 1)
            else:
                f.seek(sizes[type_id], 1)

        value_pos = None
        for _ in range(nk):
            key = string().decode("utf-8")
            type_id, = struct.unpack("<I", f.read(4))
            if key == "qwen4exp.expert_count":
                if type_id != 4:
                    raise ValueError("qwen4exp.expert_count が u32 ではありません")
                value_pos = f.tell()
            skip(type_id)
        if value_pos is None:
            raise ValueError("qwen4exp.expert_count がありません")
        end = f.tell()
        f.seek(0)
        raw = bytearray(f.read(end))
        struct.pack_into("<I", raw, value_pos, k)
        return bytes(raw)


def tensor_sizes(model):
    offsets = sorted(t.offset for t in model.tensors)
    ends = dict(zip(offsets, offsets[1:] + [model.path.stat().st_size - model.data_start]))
    return {t.name: t.expected_bytes() if t.expected_bytes() is not None
            else ends[t.offset] - t.offset for t in model.tensors}


def layout(model, keep):
    k = keep["n_expert"]
    n = model.metadata["qwen4exp.expert_count"]
    nl = model.metadata["qwen4exp.block_count"]
    layers = keep["layers"]
    if len(layers) != nl or not 0 < k <= n:
        raise ValueError("keep の層数または専門家数が元と一致しません")
    for es in layers:
        if len(es) != k or es != sorted(set(es)) or not all(0 <= e < n for e in es):
            raise ValueError("keep の専門家番号は K 個の昇順・重複なし・元の範囲内が必要です")
    sizes = tensor_sizes(model)
    offset, entries = 0, []
    for t in model.tensors:
        shape = list(t.shape)
        size = sizes[t.name]
        match = EXPERT.fullmatch(t.name) or ROUTER.fullmatch(t.name)
        if match:
            axis = 2 if EXPERT.fullmatch(t.name) else 1
            if shape[axis] != n or size % n:
                raise ValueError(f"{t.name}: 専門家数またはバイト数が不整合")
            if axis == 1 and (t.type_id != 30 or size // n != shape[0] * 2):
                raise ValueError(f"{t.name}: BF16 ルーターではありません")
            shape[axis] = k
            size = size // n * k
        offset = align(offset, model.alignment)
        entries.append((t, shape, offset, size, int(match[1]) if match else None))
        offset += size
    return entries, sizes


def copy_bytes(src, out, count):
    while count:
        raw = src.read(min(BUFFER, count))
        if not raw:
            raise ValueError("テンソルデータが途中で終わっています")
        out.write(raw)
        count -= len(raw)


def prune(src, keep, out):
    if src.resolve() == out.resolve():
        raise ValueError("元と出力に同じパスは指定できません")
    model = GGUFFile(src)
    entries, sizes = layout(model, keep)
    header = bytearray(metadata_bytes(src, keep["n_expert"]))
    for t, shape, offset, size, layer in entries:
        header += string_bytes(t.name) + struct.pack("<I", len(shape))
        header += struct.pack(f"<{len(shape)}Q", *shape) + struct.pack("<IQ", t.type_id, offset)
    data_start = align(len(header), model.alignment)
    start = time.monotonic()
    layer_bytes = {}
    last_at = {int(m[1]): i for i, (t, _, _, _, _) in enumerate(entries)
               if (m := re.match(r"blk\.(\d+)\.", t.name))}
    with src.open("rb", buffering=BUFFER) as f, out.open("wb", buffering=BUFFER) as dest:
        dest.write(header)
        dest.write(b"\0" * (data_start - len(header)))
        for i, (t, shape, offset, size, layer) in enumerate(entries):
            dest.write(b"\0" * (data_start + offset - dest.tell()))
            if layer is None:
                f.seek(model.data_start + t.offset)
                copy_bytes(f, dest, size)
            else:
                unit = sizes[t.name] // model.metadata["qwen4exp.expert_count"]
                for expert in keep["layers"][layer]:
                    f.seek(model.data_start + t.offset + expert * unit)
                    copy_bytes(f, dest, unit)
            layer_match = re.match(r"blk\.(\d+)\.", t.name)
            if layer_match:
                l = int(layer_match[1])
                layer_bytes[l] = layer_bytes.get(l, 0) + size
                if i == last_at[l]:
                    print(f"層 {l} | {layer_bytes[l]} バイト | {time.monotonic()-start:.2f} 秒", flush=True)


def digest(f, offset, count):
    f.seek(offset)
    h = hashlib.sha256()
    while count:
        raw = f.read(min(BUFFER, count))
        if not raw:
            raise ValueError("照合中にデータが途切れました")
        h.update(raw)
        count -= len(raw)
    return h.digest()


def verify(src, keep, out):
    original, result = GGUFFile(src), GGUFFile(out)
    entries, sizes = layout(original, keep)
    checks = []
    same_names = [t.name for t in original.tensors] == [t.name for t in result.tensors]
    checks.append(("テンソル数・名前の並び", same_names))
    checks.append(("expert_count", result.metadata["qwen4exp.expert_count"] == keep["n_expert"]))
    checks.append(("メタデータのバイト列", metadata_bytes(src, keep["n_expert"])
                   == metadata_bytes(out, keep["n_expert"])))
    checks.append(("shape・型・offset", same_names and all(
        t.shape == shape and t.type_id == orig.type_id and t.offset == offset
        for (orig, shape, offset, size, layer), t in zip(entries, result.tensors))))
    by_name = {t.name: t for t in result.tensors}
    rng = random.Random(42)
    with src.open("rb", buffering=BUFFER) as f, out.open("rb", buffering=BUFFER) as dest:
        for t, shape, offset, size, layer in entries:
            other = by_name.get(t.name)
            if other is None:
                checks.append((t.name, False))
                continue
            if layer is None:
                ok = digest(f, original.data_start+t.offset, sizes[t.name]) == digest(
                    dest, result.data_start+other.offset, sizes[t.name])
                checks.append((t.name + " SHA-256", ok))
            else:
                es = keep["layers"][layer]
                positions = {0, len(es)-1, rng.randrange(len(es))}
                unit = sizes[t.name] // original.metadata["qwen4exp.expert_count"]
                ok = all(digest(f, original.data_start+t.offset+es[p]*unit, unit) == digest(
                    dest, result.data_start+other.offset+p*unit, unit) for p in sorted(positions))
                checks.append((t.name + " 先頭・末尾・無作為", ok))
    last = entries[-1]
    checks.append(("末尾に余りなし", out.stat().st_size == result.data_start+last[2]+last[3]))
    print("確認項目 | 結果\n---|---")
    for name, ok in checks:
        print(f"{name} | {'PASS' if ok else 'FAIL'}")
    if not all(ok for _, ok in checks):
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--keep", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    keep = json.loads(a.keep.read_text(encoding="utf-8-sig"))
    if a.verify:
        verify(a.src, keep, a.out)
    else:
        prune(a.src, keep, a.out)


if __name__ == "__main__":
    main()
