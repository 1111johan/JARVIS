import os
import re

from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def main() -> None:
    os.makedirs("data/cif", exist_ok=True)
    os.makedirs(".cache/atomgptlab", exist_ok=True)
    os.environ.setdefault("ATOMGPTLAB_CACHE", os.path.abspath(".cache/atomgptlab"))

    print("Downloading JARVIS-DFT dft_2d dataset...")
    records = data("dft_2d")
    print("Total materials:", len(records))

    saved = 0
    skipped = 0
    failed = 0

    for idx, item in enumerate(records):
        try:
            atoms = Atoms.from_dict(item["atoms"])
            jid = safe_name(item.get("jid", f"material_{idx}"))
            formula = safe_name(item.get("formula", "unknown"))
            out = os.path.join("data", "cif", f"{jid}_{formula}.cif")
            if os.path.exists(out):
                skipped += 1
                continue
            atoms.write_cif(out)
            saved += 1
        except Exception as exc:
            failed += 1
            print(f"Failed item {item.get('jid', idx)}: {exc}")

    print("Saved CIF files:", saved)
    print("Skipped existing CIF files:", skipped)
    print("Failed items:", failed)
    print("Output folder: data/cif/")


if __name__ == "__main__":
    main()
