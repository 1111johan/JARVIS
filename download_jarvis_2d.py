import csv
import os
import re
from typing import Any, Dict, List, Set

from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data
from pymatgen.core import Composition


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def parse_float(value: Any):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().lower()
    if s in {"", "na", "none", "nan", "null"}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def normalize_formula(value: Any) -> str:
    raw = str(value or "unknown").replace(" ", "")
    if raw.lower() in {"", "unknown", "na", "none", "nan"}:
        return "unknown"
    try:
        return Composition(raw).reduced_formula.replace(" ", "")
    except Exception:
        return raw


def has_icsd(icsd_value: Any) -> int:
    if icsd_value is None:
        return 0
    if isinstance(icsd_value, (list, tuple, set, dict)):
        return int(len(icsd_value) > 0)
    if isinstance(icsd_value, (int, float)):
        return int(icsd_value > 0)
    s = str(icsd_value).strip().lower()
    return int(s not in {"", "na", "none", "nan", "null", "[]", "{}", "0"})


def load_experimental_formula_set(path: str) -> Set[str]:
    if not os.path.exists(path):
        return set()
    formulas: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            formulas.add(normalize_formula(row.get("formula", "")))
    formulas.discard("unknown")
    return formulas


def synthesis_data_label(row: Dict[str, Any], experimental_formulas: Set[str]) -> int:
    from_formula = int(row["formula_reduced"] in experimental_formulas)
    from_icsd = int(row["has_icsd"])
    return int(max(from_formula, from_icsd))


def write_metadata_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fields = list(rows[0].keys())
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    os.makedirs("data/cif", exist_ok=True)
    os.makedirs(".cache/atomgptlab", exist_ok=True)
    os.environ.setdefault("ATOMGPTLAB_CACHE", os.path.abspath(".cache/atomgptlab"))

    exp_label_file = os.path.join("data", "experimental_2d_labels.csv")
    experimental_formulas = load_experimental_formula_set(exp_label_file)

    print("Downloading JARVIS-DFT dft_2d dataset...")
    records = data("dft_2d")
    print("Total materials:", len(records))

    saved = 0
    skipped = 0
    failed = 0
    metadata_rows: List[Dict[str, Any]] = []

    for idx, item in enumerate(records):
        try:
            atoms = Atoms.from_dict(item["atoms"])
            jid = safe_name(item.get("jid", f"material_{idx}"))
            formula_reduced = normalize_formula(item.get("formula", "unknown"))
            formula_tag = safe_name(formula_reduced)
            # Deduplicate by JID to avoid creating extra CIF copies with different formula tags.
            existing = sorted(
                [p for p in os.listdir(os.path.join("data", "cif")) if p.startswith(f"{jid}_") and p.endswith(".cif")]
            )
            if existing:
                file_name = existing[0]
                out = os.path.join("data", "cif", file_name)
                skipped += 1
            else:
                file_name = f"{jid}_{formula_tag}.cif"
                out = os.path.join("data", "cif", file_name)
                atoms.write_cif(out)
                saved += 1

            row = {
                "jid": jid,
                "formula_raw": str(item.get("formula", "unknown")),
                "formula_reduced": formula_reduced,
                "cif_file": file_name,
                "cif_path": out.replace("\\", "/"),
                "formation_energy_peratom": parse_float(item.get("formation_energy_peratom")),
                "ehull": parse_float(item.get("ehull")),
                "exfoliation_energy": parse_float(item.get("exfoliation_energy")),
                "optb88vdw_bandgap": parse_float(item.get("optb88vdw_bandgap")),
                "hse_gap": parse_float(item.get("hse_gap")),
                "icsd_raw": str(item.get("icsd", "")),
                "has_icsd": has_icsd(item.get("icsd")),
                "reference": str(item.get("reference", "")),
            }
            row["exp_formula_label"] = int(row["formula_reduced"] in experimental_formulas)
            row["synthesis_data_label"] = synthesis_data_label(
                row=row, experimental_formulas=experimental_formulas
            )
            metadata_rows.append(row)
        except Exception as exc:
            failed += 1
            print(f"Failed item {item.get('jid', idx)}: {exc}")

    metadata_path = os.path.join("data", "jarvis_dft_2d_metadata.csv")
    metadata_rows = sorted(metadata_rows, key=lambda x: x["jid"])
    write_metadata_csv(rows=metadata_rows, out_path=metadata_path)

    print("Saved CIF files:", saved)
    print("Skipped existing CIF files:", skipped)
    print("Failed items:", failed)
    print("Output folder: data/cif/")
    print("Metadata file:", metadata_path)
    print("Experimental formula labels loaded:", len(experimental_formulas))


if __name__ == "__main__":
    main()
