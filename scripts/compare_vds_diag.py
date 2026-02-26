#!/usr/bin/env python3
import sys
from pathlib import Path


def parse_kv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("KV|"):
            continue
        parts = line.split("|", 2)
        if len(parts) != 3:
            continue
        _, k, v = parts
        out[k.strip()] = v.strip()
    return out


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: compare_vds_diag.py <report_a.txt> <report_b.txt>")
        return 2
    a = Path(sys.argv[1]).expanduser().resolve()
    b = Path(sys.argv[2]).expanduser().resolve()
    if not a.exists() or not b.exists():
        print("Both files must exist")
        return 2

    da = parse_kv(a)
    db = parse_kv(b)
    keys = sorted(set(da.keys()) | set(db.keys()))

    print(f"Compare A={a}")
    print(f"Compare B={b}\n")

    changed = 0
    for k in keys:
        va = da.get(k, "<missing>")
        vb = db.get(k, "<missing>")
        if va != vb:
            changed += 1
            print(f"- {k}")
            print(f"  A: {va}")
            print(f"  B: {vb}")

    if changed == 0:
        print("No KV differences found.")
    else:
        print(f"\nTotal differences: {changed}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
