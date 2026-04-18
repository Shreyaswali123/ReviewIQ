"""
ingest.py — Robust data ingestion for ReviewIQ
Handles: arbitrary column names, multi-encoding CSVs, nested JSON,
         missing fields, empty rows, and pasted text.
"""

import pandas as pd
import json
import io
import re
from typing import Union

# ── Column alias registry ──────────────────────────────────────────────────────
# Maps our standard internal field names → every likely real-world column variant.
COLUMN_ALIASES: dict[str, list[str]] = {
    "review_id": [
        "review_id", "id", "reviewid", "rev_id", "review_number",
        "uid", "uuid", "record_id", "index", "row_id",
    ],
    "review_text": [
        "review_text", "text", "review", "body", "content", "comment",
        "feedback", "description", "review_body", "reviewtext", "review text",
        "message", "review_content", "customer_review", "user_review",
        "review_comment", "comments", "customer_feedback", "review_description",
        "review_details", "opinion", "remarks",
    ],
    "product_name": [
        "product_name", "product", "name", "item", "item_name",
        "product_title", "title", "productname", "asin", "sku",
        "item_title", "product_id", "product_label", "item_id",
    ],
    "category": [
        "category", "cat", "type", "product_category", "department",
        "section", "genre", "product_type", "item_category",
        "categoryname", "category_name", "vertical", "segment",
    ],
    "star_rating": [
        "star_rating", "rating", "stars", "score", "review_rating",
        "rate", "overall_rating", "ratings", "star", "review_score",
        "user_rating", "overall", "out_of_5", "stars_given",
    ],
    "review_date": [
        "review_date", "date", "timestamp", "created_at", "review_time",
        "post_date", "submitted_at", "date_posted", "created",
        "published_at", "review_on", "date_of_review", "posted_on",
    ],
    "reviewer_id": [
        "reviewer_id", "reviewer", "user_id", "author", "username",
        "customer_id", "user", "userid", "author_id", "customer",
        "profile_name", "reviewer_name", "buyer_id",
    ],
}

OPTIONAL_DEFAULTS = {
    "product_name": "Unknown Product",
    "category":     "unknown",
    "star_rating":  None,
    "review_date":  "2025-01-01",
}

# ── Encoding helpers ───────────────────────────────────────────────────────────

def _decode_bytes(file_bytes: bytes) -> str:
    """
    Tries common encodings in order. Falls back to chardet, then lossy
    replacement so we never hard-crash on a bad file.
    """
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"):
        try:
            return file_bytes.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    # Last resort: chardet
    try:
        import chardet
        detected = chardet.detect(file_bytes)
        enc = detected.get("encoding") or "utf-8"
        return file_bytes.decode(enc, errors="replace")
    except ImportError:
        pass
    return file_bytes.decode("utf-8", errors="replace")


# ── Column-name normalisation ──────────────────────────────────────────────────

def _normalise(col: str) -> str:
    """Lowercase, strip, collapse spaces/hyphens/dots to underscores."""
    return re.sub(r"[\s\-\.]+", "_", col.strip().lower())


def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames arbitrary column names to our internal schema using COLUMN_ALIASES.
    Unmapped columns are kept as-is (they won't interfere).
    """
    # Build a map: normalised_column_name → original_column_name
    norm_to_orig = {_normalise(c): c for c in df.columns}

    rename_map: dict[str, str] = {}
    for standard, aliases in COLUMN_ALIASES.items():
        if standard in rename_map.values():
            continue  # already mapped
        for alias in aliases:
            key = _normalise(alias)
            if key in norm_to_orig and norm_to_orig[key] not in rename_map:
                rename_map[norm_to_orig[key]] = standard
                break

    return df.rename(columns=rename_map)


def _auto_detect_text_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    If no recognised text column exists, pick the column with the longest
    average string length (most likely to be free-form review text).
    """
    if "review_text" in df.columns:
        return df

    best_col, best_avg = None, 0
    for col in df.columns:
        try:
            avg = df[col].dropna().astype(str).str.len().mean()
            if avg > best_avg:
                best_avg = avg
                best_col = col
        except Exception:
            pass

    if best_col and best_avg >= 20:
        print(f"⚠️  Auto-detected '{best_col}' as review_text column (avg {best_avg:.0f} chars)")
        df = df.rename(columns={best_col: "review_text"})
    else:
        raise ValueError(
            f"Could not locate a review-text column. Found: {list(df.columns)}. "
            "Expected one of: text, review, body, content, comment, feedback."
        )
    return df


# ── Record validation & normalisation ─────────────────────────────────────────

def _validate_and_fill(records: list, source_label: str = "") -> list:
    """
    • Removes records with no usable review text.
    • Auto-generates IDs when absent.
    • Fills optional fields with defaults.
    • Coerces star_rating to int-or-None.
    """
    clean = []
    skipped = 0

    for i, r in enumerate(records):
        text = str(r.get("review_text", "")).strip()
        # Skip blank or sentinel-value rows
        if not text or text.lower() in ("nan", "none", "null", "n/a", ""):
            skipped += 1
            continue

        # Auto-generate missing IDs
        prefix = source_label or "rec"
        if not r.get("review_id") or str(r.get("review_id", "")).lower() in ("nan", "none", ""):
            r["review_id"] = f"{prefix}_{i:06d}"
        if not r.get("reviewer_id") or str(r.get("reviewer_id", "")).lower() in ("nan", "none", ""):
            r["reviewer_id"] = f"USER_{i:06d}"

        # Fill optional defaults
        for field, default in OPTIONAL_DEFAULTS.items():
            val = r.get(field)
            if val is None or str(val).strip().lower() in ("nan", "none", "null", "n/a", ""):
                r[field] = default

        # Coerce star_rating
        try:
            r["star_rating"] = int(float(str(r["star_rating"])))
            if not 1 <= r["star_rating"] <= 5:
                r["star_rating"] = None
        except (ValueError, TypeError):
            r["star_rating"] = None

        clean.append(r)

    if skipped:
        print(f"ℹ️  Skipped {skipped} empty/invalid rows")
    return clean


# ── Public loaders ─────────────────────────────────────────────────────────────

def load_csv(file_bytes: bytes) -> list:
    """
    Loads reviews from a CSV.
    Handles: arbitrary column names, multiple encodings, missing fields,
             empty rows, files with only a header row.
    """
    text = _decode_bytes(file_bytes)

    try:
        df = pd.read_csv(io.StringIO(text), dtype=str, low_memory=False)
    except Exception as e:
        raise ValueError(f"Failed to parse CSV: {e}")

    if df.empty:
        print("⚠️  CSV file is empty.")
        return []

    df = _map_columns(df)
    df = _auto_detect_text_column(df)
    df = df.fillna("")

    records = _validate_and_fill(df.to_dict(orient="records"), source_label="csv")
    print(f"✅ Loaded {len(records)} valid reviews from CSV")
    return records


def load_json(json_data: Union[list, dict]) -> list:
    """
    Parses JSON payloads.
    Handles: bare list, {data:[...]}, {reviews:[...]}, {results:[...]}, single dict.
    """
    if isinstance(json_data, list):
        raw = json_data
    elif isinstance(json_data, dict):
        for key in ("data", "reviews", "results", "items", "records", "rows"):
            if key in json_data and isinstance(json_data[key], list):
                raw = json_data[key]
                break
        else:
            raw = [json_data]  # treat single dict as one record
    else:
        raise ValueError(f"Unsupported JSON root type: {type(json_data).__name__}")

    # Normalise keys on every record
    normalised = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        norm = {_normalise(k): v for k, v in r.items()}
        # Apply alias mapping
        mapped: dict = {}
        for standard, aliases in COLUMN_ALIASES.items():
            for alias in aliases:
                key = _normalise(alias)
                if key in norm:
                    mapped[standard] = norm[key]
                    break
        # Preserve unmapped fields
        alias_keys = {_normalise(a) for als in COLUMN_ALIASES.values() for a in als}
        for k, v in norm.items():
            if k not in alias_keys:
                mapped.setdefault(k, v)
        normalised.append(mapped)

    return _validate_and_fill(normalised, source_label="json")


def load_pasted(text: str) -> list:
    """Handles pasted reviews split by newlines. Skips blank lines."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    records = [
        {
            "review_id":   f"manual_{i:04d}",
            "review_text": line,
            "product_name": "Unknown Product",
            "category":    "unknown",
            "star_rating": None,
            "review_date": "2025-01-01",
            "reviewer_id": f"USER_{i:04d}",
        }
        for i, line in enumerate(lines)
    ]
    return _validate_and_fill(records, source_label="paste")


def load_demo_dataset() -> list:
    """Loads all synthetic CSVs as one combined list."""
    import os
    all_reviews: list = []
    paths = [
        "data/synthetic/smartwatch_reviews.csv",
        "data/synthetic/protein_bar_reviews.csv",
        "data/synthetic/running_shoes_reviews.csv",
    ]
    for path in paths:
        if os.path.exists(path):
            with open(path, "rb") as f:
                all_reviews.extend(load_csv(f.read()))
        else:
            print(f"WARNING: {path} not found. Run phase1.py first.")
    return all_reviews