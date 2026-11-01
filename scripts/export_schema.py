"""Write spec.schema.json (repo root and inside the package) from the spec model.

python scripts/export_schema.py            # rewrite both files
python scripts/export_schema.py --check    # exit 1 if either file is out of date (used by tests and CI)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from agenticreq.schema import SCHEMA_FILE, spec_json_schema

ROOT = Path(__file__).resolve().parent.parent
TARGETS = [ROOT / SCHEMA_FILE, ROOT / "src" / "agenticreq" / SCHEMA_FILE]


def main(argv: list[str]) -> int:
    text = json.dumps(spec_json_schema(), indent=2) + "\n"
    if "--check" in argv:
        stale = [t for t in TARGETS if not t.is_file() or t.read_text(encoding="utf-8") != text]
        for t in stale:
            print(f"out of date: {t.relative_to(ROOT)}  (run: python scripts/export_schema.py)")
        return 1 if stale else 0
    for t in TARGETS:
        t.write_text(text, encoding="utf-8")
        print(f"wrote {t.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
