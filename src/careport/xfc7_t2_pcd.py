"""Reader for TruckScenes PCD files (DATA binary_compressed: LZF over a column-major field layout).

Pure Python, no new dependency. LZF format (liblzf): a control byte c < 32 starts a literal run of c + 1 bytes;
otherwise length = c >> 5 (7 means: add the next byte), offset = ((c & 31) << 8) + next byte + 1 back from the
output end, and length + 2 bytes are copied (possibly overlapping)."""
import struct

import numpy as np

_NP = {("F", 4): np.float32, ("F", 8): np.float64, ("U", 1): np.uint8, ("U", 2): np.uint16, ("U", 4): np.uint32,
       ("U", 8): np.uint64, ("I", 1): np.int8, ("I", 2): np.int16, ("I", 4): np.int32, ("I", 8): np.int64}


def lzf_decompress(data: bytes, out_len: int) -> bytes:
    out = bytearray()
    i, n = 0, len(data)
    while i < n:
        c = data[i]
        i += 1
        if c < 32:
            out += data[i:i + c + 1]
            i += c + 1
        else:
            ln = c >> 5
            if ln == 7:
                ln += data[i]
                i += 1
            ref = len(out) - ((c & 31) << 8) - data[i] - 1
            i += 1
            ln += 2
            if ref < 0:
                raise ValueError("LZF back-reference before the start of the output")
            if ref + ln <= len(out):
                out += out[ref:ref + ln]
            else:                                   # overlapping copy
                for k in range(ln):
                    out.append(out[ref + k])
    if len(out) != out_len:
        raise ValueError(f"LZF output {len(out)} bytes, expected {out_len}")
    return bytes(out)


def read_pcd(path):
    """{field: array} for a PCD file with DATA binary_compressed or binary."""
    blob = open(path, "rb").read()
    k = blob.index(b"DATA")
    end = blob.index(b"\n", k)
    head = {}
    for line in blob[:end].decode("ascii").splitlines():
        parts = line.split()
        if parts:
            head[parts[0]] = parts[1:]
    fields, sizes, types = head["FIELDS"], [int(s) for s in head["SIZE"]], head["TYPE"]
    n = int(head["POINTS"][0])
    body = blob[end + 1:]
    kind = head["DATA"][0]
    if kind == "binary_compressed":
        csize, usize = struct.unpack("<II", body[:8])
        raw = lzf_decompress(body[8:8 + csize], usize)
        out, off = {}, 0
        for f, s, t in zip(fields, sizes, types):
            out[f] = np.frombuffer(raw, dtype=_NP[(t, s)], count=n, offset=off).copy()
            off += s * n
        return out
    if kind == "binary":
        dt = np.dtype([(f, _NP[(t, s)]) for f, s, t in zip(fields, sizes, types)])
        rec = np.frombuffer(body, dtype=dt, count=n)
        return {f: rec[f].copy() for f in fields}
    raise ValueError(f"unsupported PCD DATA {kind}")
