"""XF-C7 T3, before the freeze (SAMPLE_PIN addendum 12): check that the protocol draft's camera tables are the fill
tool's rows, unchanged and in order (header, separator and the seven camera rows; leading spaces ignored).
Usage, from the selection folder: python -I -X utf8 t3_replication_2026-11-04/code_checks/t3_s3_table_check.py
  PILOT_PROTOCOL_XF-C7_T3_v1_0_DRAFT.md t3_replication_2026-11-04/s3_fill/t3_s3_values.md"""
import sys


def tool_tables(md_text):
    out = []
    for block in md_text.split("### ")[1:]:
        name, body = block.split("\n", 1)
        rows = [l for l in body.splitlines() if l.startswith(("| ring_", "| Camera", "|---"))]
        if rows:
            out.append((name.strip(), rows))
    return out


def protocol_tables(text):
    tables, cur = [], []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(("| Camera", "|---", "| ring_")):
            cur.append(s)
        elif cur:
            tables.append(cur)
            cur = []
    if cur:
        tables.append(cur)
    return [t for t in tables if t[0].startswith("| Camera")]


def main(argv):
    proto, md = argv[1], argv[2]
    tool = tool_tables(open(md, encoding="utf-8").read())
    prot = protocol_tables(open(proto, encoding="utf-8").read())
    if len(tool) != len(prot):
        print(f"tool has {len(tool)} tables, protocol {len(prot)}: DIFFERENT")
        return 1
    bad = 0
    for (name, rows), t in zip(tool, prot):
        same = rows == t
        bad += not same
        print(name, f"({len(rows)} lines)", "IDENTICAL" if same else "DIFFERENT")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
