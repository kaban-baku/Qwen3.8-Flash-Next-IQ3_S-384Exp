"""2 層・8 専門家・embedding 16・expert_ff 8 の合成 GGUF。"""
import argparse
import json
import struct
from pathlib import Path


def string(text):
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def make(path):
    metadata = [("general.architecture", 8, string("qwen4exp")),
                ("general.alignment", 4, struct.pack("<I", 64)),
                ("qwen4exp.block_count", 4, struct.pack("<I", 2)),
                ("qwen4exp.expert_count", 4, struct.pack("<I", 8)),
                ("qwen4exp.expert_used_count", 4, struct.pack("<I", 2)),
                ("test.strings", 9, struct.pack("<IQ", 8, 2) + string("日本語") + string("tiny"))]
    tensors = [("token_embd.weight", [16, 3], 1, bytes(range(96)))]
    for layer in range(2):
        for kind in ("gate", "up", "down"):
            shape = [8, 16, 8] if kind == "down" else [16, 8, 8]
            # 型42のパスも確認する。量子化済みバイト列の意味はこの試験では使わない。
            type_id = 1 if layer == 0 else 42
            unit = 16*8*2 if type_id == 1 else 16*8//64*18
            data = b"".join(bytes([(layer*80 + len(tensors)*8 + e) % 256])*unit for e in range(8))
            tensors.append((f"blk.{layer}.ffn_{kind}_exps.weight", shape, type_id, data))
        data = b"".join(bytes([e+layer*16])*32 for e in range(8))
        tensors.append((f"blk.{layer}.ffn_gate_inp.weight", [16, 8], 30, data))
        tensors.append((f"blk.{layer}.attn_norm.weight", [16], 1, bytes([layer+5])*32))
    header = bytearray(struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(metadata)))
    for key, type_id, raw in metadata:
        header += string(key) + struct.pack("<I", type_id) + raw
    offset, offsets = 0, []
    for name, shape, type_id, data in tensors:
        offset = (offset+63)//64*64
        offsets.append(offset)
        header += string(name) + struct.pack("<I", len(shape)) + struct.pack(f"<{len(shape)}Q", *shape)
        header += struct.pack("<IQ", type_id, offset)
        offset += len(data)
    data_start = (len(header)+63)//64*64
    with path.open("wb") as f:
        f.write(header)
        f.write(b"\0"*(data_start-f.tell()))
        for (_, _, _, data), off in zip(tensors, offsets):
            f.write(b"\0"*(data_start+off-f.tell()))
            f.write(data)
    keep = {"n_expert": 4, "layers": [[0, 2, 5, 7], [1, 3, 4, 6]], "floor": 0, "weights": {}}
    path.with_name("tiny_keep_4.json").write_text(json.dumps(keep, indent=2), encoding="utf-8")
    print(f"{path}: {path.stat().st_size} バイト")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "tiny.gguf")
    make(ap.parse_args().out)
