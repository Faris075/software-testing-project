"""
database/seed.py
================
Seeds the SQLite database with:
  1. Synthetic cohort data — 15 students × 5 subjects × 2 years (2024, 2025)
  2. Real VLE dataset     — 160 student instances from data/vle/original/original.csv
                           + test split from data/vle/test/test.csv

Run from the project root:
    python database/seed.py
"""

import sqlite3
import pathlib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = pathlib.Path(__file__).parent.parent
DB_PATH     = ROOT / "database" / "students.db"
SCHEMA_PATH = ROOT / "database" / "schema.sql"
RAW_CSV     = ROOT / "data" / "students_raw.csv"
VLE_ORIG    = ROOT / "data" / "vle" / "original" / "original.csv"
VLE_TEST    = ROOT / "data" / "vle" / "test" / "test.csv"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def apply_schema(conn: sqlite3.Connection) -> None:
    with open(SCHEMA_PATH, "r", encoding="utf-8") as fh:
        conn.executescript(fh.read())
    conn.commit()


# ---------------------------------------------------------------------------
# Part 1 — Synthetic cohort (marks dataset)
# ---------------------------------------------------------------------------

STUDENT_NAMES = [
    "Ali", "Sara", "Omar", "Lena", "Tariq",
    "Maya", "Khalid", "Nour", "Reem", "Jad",
    "Hana", "Zaid", "Dina", "Faris", "Layla",
]
SUBJECTS = ["Math", "Physics", "CS", "English", "Statistics"]

# Intentional missing value positions (row_index, subject_index) — 0-based
MISSING_2024 = [(2, 1), (5, 3), (7, 0), (11, 2), (13, 1)]
MISSING_2025 = [(9, 4), (14, 3)]


def _build_marks_array(seed_val: int, missing_pos: list) -> np.ndarray:
    """Generate 15×5 mark matrix with optional NaN cells."""
    rng = np.random.default_rng(seed_val)
    marks = rng.integers(50, 96, size=(15, 5)).astype(float)
    for r, c in missing_pos:
        marks[r, c] = np.nan
    return marks


def seed_synthetic_cohort(conn: sqlite3.Connection) -> pd.DataFrame:
    """Insert synthetic students + marks; save students_raw.csv; return wide DataFrame."""

    marks_2024 = _build_marks_array(42,  MISSING_2024)
    marks_2025 = np.clip(
        _build_marks_array(42, []) + np.random.default_rng(99).integers(-8, 13, size=(15, 5)),
        0, 100
    ).astype(float)
    for r, c in MISSING_2025:
        marks_2025[r, c] = np.nan

    # -- Build wide DataFrame for CSV export ---------------------------------
    rows = []
    for i, name in enumerate(STUDENT_NAMES):
        student_id = f"S{i+1:03d}"
        for year, matrix in [(2024, marks_2024), (2025, marks_2025)]:
            row = {"student_id": student_id, "name": name, "year": year}
            for j, subj in enumerate(SUBJECTS):
                row[subj] = matrix[i, j]
            rows.append(row)

    df = pd.DataFrame(rows)

    # Save raw CSV (data/ folder created if needed)
    (ROOT / "data").mkdir(exist_ok=True)
    df.to_csv(RAW_CSV, index=False)
    print(f"[seed] Saved {RAW_CSV}")

    # -- Insert into SQLite --------------------------------------------------
    cursor = conn.cursor()

    # students table
    cursor.executemany(
        "INSERT OR IGNORE INTO students (student_id, name) VALUES (?, ?)",
        [(f"S{i+1:03d}", name) for i, name in enumerate(STUDENT_NAMES)],
    )

    # marks table
    mark_rows = []
    for i, name in enumerate(STUDENT_NAMES):
        student_id = f"S{i+1:03d}"
        for year, matrix in [(2024, marks_2024), (2025, marks_2025)]:
            for j, subj in enumerate(SUBJECTS):
                val = matrix[i, j]
                mark_rows.append((student_id, year, subj, None if np.isnan(val) else float(val)))

    cursor.executemany(
        "INSERT INTO marks (student_id, year, subject, mark) VALUES (?, ?, ?, ?)",
        mark_rows,
    )

    conn.commit()
    print(f"[seed] Synthetic cohort: {len(STUDENT_NAMES)} students, "
          f"{len(mark_rows)} mark rows inserted.")
    return df


# ---------------------------------------------------------------------------
# Part 2 — Real VLE dataset
# ---------------------------------------------------------------------------

# Mapping from raw CSV column names → clean DB column names
VLE_COLUMN_MAP = {
    "Gender":                              "gender",
    "Age":                                 "age",
    "POLAR4 Quintile":                     "polar4_quintile",
    "POLAR3 Quintile":                     "polar3_quintile",
    "Adult HE 2001 Quintile":              "adult_he_2001_quintile",
    "Adult HE 2011 Quintile":              "adult_he_2011_quintile",
    "TUNDRA MSOA Quintile":                "tundra_msoa_quintile",
    "TUNDRA LSOA Quintile":               "tundra_lsoa_quintile",
    "Gaps GCSE Quintile ":                 "gaps_gcse_quintile",     # note trailing space
    "Gaps GCSE Ethnicity Quintile":        "gaps_gcse_ethnicity_quintile",
    "Uni Connect target ward":             "uni_connect_target_ward",
    "attending from home?":                "attending_from_home",
    "distance to university (km)":         "distance_to_uni_km",
    "Count of Module Area Logins":         "logins",
    "Total Hours in Module Area":          "total_hours",
    "% of Average Hours in Module Area":   "pct_avg_hours",
    "# of presence":                       "presence_count",
    "# of Absence":                        "absence_count",
    "Percent Attended":                    "pct_attended",
    'label (fail=1, pass=0)':              "label",
}

DB_COLS = [
    "gender", "age", "logins", "total_hours", "pct_avg_hours",
    "presence_count", "absence_count", "pct_attended",
    "attending_from_home", "distance_to_uni_km",
    "polar4_quintile", "polar3_quintile",
    "adult_he_2001_quintile", "adult_he_2011_quintile",
    "tundra_msoa_quintile", "tundra_lsoa_quintile",
    "gaps_gcse_quintile", "gaps_gcse_ethnicity_quintile",
    "uni_connect_target_ward", "label", "split",
]

INSERT_SQL = f"""
    INSERT INTO vle_students
    ({', '.join(DB_COLS)})
    VALUES ({', '.join(['?'] * len(DB_COLS))})
"""


def _load_vle_csv(path: pathlib.Path, split_label: str) -> pd.DataFrame:
    """Load and normalise one VLE CSV file."""
    df = pd.read_csv(path)
    # Drop any fully-empty trailing columns (artefact of some exports)
    df = df.loc[:, ~df.columns.str.fullmatch(r"Unnamed.*")]
    # Strip whitespace from headers
    df.columns = df.columns.str.strip()
    # Re-apply strip to the map keys as well when looking up
    col_map = {k.strip(): v for k, v in VLE_COLUMN_MAP.items()}
    df = df.rename(columns=col_map)
    # Keep only columns we care about
    keep = [c for c in DB_COLS if c != "split"]
    df = df[[c for c in keep if c in df.columns]]
    # Convert to numeric, coercing errors to NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["split"] = split_label
    return df


def seed_vle_dataset(conn: sqlite3.Connection) -> None:
    """Load original.csv (original split) + test.csv (test split) into vle_students."""

    frames = []

    if VLE_ORIG.exists():
        frames.append(_load_vle_csv(VLE_ORIG, "original"))
    else:
        print(f"[seed] WARNING: {VLE_ORIG} not found — skipping VLE original split.")

    if VLE_TEST.exists():
        frames.append(_load_vle_csv(VLE_TEST, "test"))
    else:
        print(f"[seed] WARNING: {VLE_TEST} not found — skipping VLE test split.")

    if not frames:
        print("[seed] No VLE data found. Skipping.")
        return

    combined = pd.concat(frames, ignore_index=True)

    # Deduplicate by all feature columns (same row may appear in both files for original)
    feature_cols = [c for c in DB_COLS if c != "split"]
    combined = combined.drop_duplicates(subset=feature_cols)

    # Drop rows where label is missing — label is NOT NULL in schema
    before = len(combined)
    combined = combined.dropna(subset=["label"])
    dropped = before - len(combined)
    if dropped:
        print(f"[seed] Dropped {dropped} VLE row(s) with missing label.")

    cursor = conn.cursor()
    inserted = 0
    for _, row in combined.iterrows():
        values = []
        for col in DB_COLS:
            val = row.get(col, None)
            # Replace NaN with None for SQLite NULL
            values.append(None if pd.isna(val) else val)
        cursor.execute(INSERT_SQL, values)
        inserted += 1

    conn.commit()
    print(f"[seed] VLE dataset: {inserted} rows inserted into vle_students.")
    print(f"       Class distribution — "
          f"pass(0): {int((combined['label'] == 0).sum())}, "
          f"fail(1): {int((combined['label'] == 1).sum())}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"[seed] Connecting to {DB_PATH}")
    conn = get_connection()

    print("[seed] Applying schema...")
    apply_schema(conn)

    print("[seed] Seeding synthetic cohort marks...")
    seed_synthetic_cohort(conn)

    print("[seed] Seeding real VLE dataset...")
    seed_vle_dataset(conn)

    conn.close()
    print("[seed] Done. Database ready at:", DB_PATH)


if __name__ == "__main__":
    main()
