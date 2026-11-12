import os
import struct
import tempfile
import unittest

import numpy as np


def _lzf_literals(raw: bytes) -> bytes:
    """A valid LZF stream made of literal runs only (max 32 bytes each)."""
    out = bytearray()
    for i in range(0, len(raw), 32):
        chunk = raw[i:i + 32]
        out.append(len(chunk) - 1)
        out += chunk
    return bytes(out)


class LzfTest(unittest.TestCase):
    def test_literal_and_back_reference(self):
        from careport.xfc7_t2_pcd import lzf_decompress
        stream = bytes([3]) + b"abcd" + bytes([(2 << 5) | 0, 3])          # copy 4 bytes from 4 back
        self.assertEqual(lzf_decompress(stream, 8), b"abcdabcd")

    def test_overlapping_run_and_long_length(self):
        from careport.xfc7_t2_pcd import lzf_decompress
        # 'a' then a back-reference of length 7 + 3 + 2 = 12 at offset 1 (overlapping copy)
        stream = bytes([0]) + b"a" + bytes([(7 << 5) | 0, 3, 0])
        self.assertEqual(lzf_decompress(stream, 13), b"a" * 13)

    def test_size_mismatch_raises(self):
        from careport.xfc7_t2_pcd import lzf_decompress
        with self.assertRaises(ValueError):
            lzf_decompress(bytes([1]) + b"ab", 5)


class PcdTest(unittest.TestCase):
    def test_reads_binary_compressed_column_major(self):
        from careport.xfc7_t2_pcd import read_pcd
        n = 3
        x = np.array([1.0, 2.0, 3.0], np.float32)
        y = np.array([4.0, 5.0, 6.0], np.float32)
        t = np.array([10, 20, 30], np.uint64)
        raw = x.tobytes() + y.tobytes() + t.tobytes()
        comp = _lzf_literals(raw)
        header = (f"VERSION 0.7\nFIELDS x y timestamp\nSIZE 4 4 8\nTYPE F F U\nCOUNT 1 1 1\nWIDTH {n}\nHEIGHT 1\n"
                  f"VIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary_compressed\n").encode()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.pcd")
            open(p, "wb").write(header + struct.pack("<II", len(comp), len(raw)) + comp)
            pc = read_pcd(p)
        np.testing.assert_array_equal(pc["x"], x)
        np.testing.assert_array_equal(pc["y"], y)
        np.testing.assert_array_equal(pc["timestamp"], t)


if __name__ == "__main__":
    unittest.main()
