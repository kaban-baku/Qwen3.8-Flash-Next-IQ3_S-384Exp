"""経路記録（--dump-routing）の専門家番号を、刈り込み後の番号に写し替える（残さない専門家はレコードから落とす）。
usage: python remap_traces.py <traces dir> <keep_K.json> <out dir>
レコード = <ii（layer, k）> + k 個の int32 + k 個の float32（make_profile.read_trace と同じ）。"""
import json, struct, sys
from pathlib import Path

src, keep_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
keep = json.loads(keep_path.read_text(encoding="utf-8-sig"))
new_index = [{e: i for i, e in enumerate(layer)} for layer in keep["layers"]]
out.mkdir(parents=True, exist_ok=True)
total = kept = 0
for trace in sorted(src.glob("*.trace")):
    blob = trace.read_bytes()
    off, parts = 0, []
    while off + 8 <= len(blob):
        layer, k = struct.unpack_from("<ii", blob, off); off += 8
        ids = struct.unpack_from("<%di" % k, blob, off); off += 4 * k
        ws = struct.unpack_from("<%df" % k, blob, off); off += 4 * k
        pairs = [(new_index[layer][e], w) for e, w in zip(ids, ws) if e in new_index[layer]]
        total += k; kept += len(pairs)
        if pairs:
            parts.append(struct.pack("<ii", layer, len(pairs)) + struct.pack("<%di" % len(pairs), *[p[0] for p in pairs])
                         + struct.pack("<%df" % len(pairs), *[p[1] for p in pairs]))
    (out / trace.name).write_bytes(b"".join(parts))
print(f"写した記録 {len(list(src.glob('*.trace')))} 本、経路 {kept}/{total}（{kept/total*100:.1f}% が残る）")
