#!/usr/bin/env python3
"""
Download full BEA NIUnderlyingDetail tables U20404 and U20405 (monthly) and save as CSV.

Tables:
- U20404: PCE price indexes by type of product (2.4.4U)
- U20405: PCE nominal expenditures by type of product (2.4.5U)
- APIDatasetMetaData (for NIUnderlyingDetail): includes hierarchy fields such as
  ParentLineNumber/Tier/Path for line drill-down.

API docs: https://apps.bea.gov/api/_pdf/bea_web_service_api_user_guide.pdf
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Set

import requests

API_KEY_FILE = Path(__file__).with_name("bea_api_key.txt")
ENDPOINT = "https://apps.bea.gov/api/data"

DATASET = "NIUnderlyingDetail"
FREQUENCY = "M"   # Monthly
YEAR = "ALL"      # Full history
TABLES = ["U20404", "U20405"]
METADATA_DATASET = "APIDatasetMetaData"
HIERARCHY_SOURCE_TABLE = "U20405"
RETRIEVE_API_DATA = True  # False -> skip API calls and regenerate hierarchy from existing metadata CSV
REQUEST_TIMEOUT_SECONDS = 120
MAX_REQUEST_ATTEMPTS = 6
BASE_BACKOFF_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 90.0


def load_api_key(key_file: Path) -> str:
    def normalize_key(raw: str) -> str:
        # Remove UTF-8 BOM if present and trim surrounding whitespace.
        return str(raw or "").lstrip("\ufeff").strip()

    env_key = normalize_key(os.getenv("BEA_API_KEY", ""))
    if env_key:
        return env_key

    try:
        file_key = normalize_key(key_file.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Missing BEA API key. Set BEA_API_KEY env var or create key file: {key_file}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"Failed reading BEA API key file: {key_file}") from exc

    if not file_key:
        raise RuntimeError(f"BEA API key file is empty: {key_file}")
    return file_key


API_KEY = load_api_key(API_KEY_FILE)


def extract_error(j: Dict[str, Any]) -> Dict[str, Any] | None:
    beaapi = (j or {}).get("BEAAPI") or {}
    results = beaapi.get("Results") if isinstance(beaapi, dict) else None

    if isinstance(results, dict):
        err = results.get("Error")
        if isinstance(err, dict):
            return err

    err = beaapi.get("Error") if isinstance(beaapi, dict) else None
    if isinstance(err, dict):
        return err

    return None


def extract_rows(j: Dict[str, Any]) -> List[Dict[str, Any]]:
    beaapi = (j or {}).get("BEAAPI") or {}
    results = beaapi.get("Results") if isinstance(beaapi, dict) else None

    # Most common shape: BEAAPI.Results.Data -> [row, ...]
    if isinstance(results, dict):
        data = results.get("Data")
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]

    # Some metadata responses may put rows directly in Results.
    if isinstance(results, list):
        return [x for x in results if isinstance(x, dict)]

    # Last-resort: pick the largest list-of-dicts found under Results.
    if isinstance(results, dict):
        list_candidates = [v for v in results.values() if isinstance(v, list)]
        if list_candidates:
            best = max(list_candidates, key=len)
            return [x for x in best if isinstance(x, dict)]

    raise RuntimeError("Unexpected response shape: could not find row list in BEAAPI.Results")


def as_list(x: Any) -> List[Any]:
    if isinstance(x, list):
        return x
    if isinstance(x, dict):
        return [x]
    return []


def flatten_metadata_rows(j: Dict[str, Any], dataset_name: str) -> List[Dict[str, Any]]:
    """
    Flatten APIDatasetMetaData payload into line-level rows.
    Expected shape (based on BEA docs/packages):
      BEAAPI.Datasets[].APITable[].Line[]
    """
    beaapi = (j or {}).get("BEAAPI") or {}

    datasets_obj = beaapi.get("Datasets")
    if datasets_obj is None and isinstance(beaapi.get("Results"), dict):
        datasets_obj = beaapi["Results"].get("Datasets")

    out: List[Dict[str, Any]] = []
    for ds in as_list(datasets_obj):
        if not isinstance(ds, dict):
            continue

        ds_name = pick_field(ds, "DatasetName", "Datasetname", "datasetname", "Dataset")
        if dataset_name and ds_name and str(ds_name).lower() != dataset_name.lower():
            continue

        md_updated = pick_field(ds, "MetaDataUpdated", "MetadataUpdated")
        tables_obj = pick_field(ds, "APITable", "ApiTable", "APITables")

        for tab in as_list(tables_obj):
            if not isinstance(tab, dict):
                continue

            table_name = pick_field(tab, "TableName", "Table")
            table_id = pick_field(tab, "TableID", "TableId")
            release_date = pick_field(tab, "ReleaseDate")
            next_release_date = pick_field(tab, "NextReleaseDate")
            lines_obj = pick_field(tab, "Line", "Lines")

            for ln in as_list(lines_obj):
                if not isinstance(ln, dict):
                    continue

                row = dict(ln)
                if "DatasetName" not in row and ds_name is not None:
                    row["DatasetName"] = ds_name
                if "TableName" not in row and table_name is not None:
                    row["TableName"] = table_name
                if "TableID" not in row and table_id is not None:
                    row["TableID"] = table_id
                if "MetaDataUpdated" not in row and md_updated is not None:
                    row["MetaDataUpdated"] = md_updated
                if "ReleaseDate" not in row and release_date is not None:
                    row["ReleaseDate"] = release_date
                if "NextReleaseDate" not in row and next_release_date is not None:
                    row["NextReleaseDate"] = next_release_date
                out.append(row)

    return out


def fetch_table(table_name: str) -> List[Dict[str, Any]]:
    params = {
        "UserID": API_KEY,
        "method": "GetData",
        "DataSetName": DATASET,
        "TableName": table_name,
        "Frequency": FREQUENCY,
        "Year": YEAR,
        "ResultFormat": "JSON",
    }

    j = request_json_with_backoff(params, label=f"{DATASET} {table_name}")

    err = extract_error(j)
    if err:
        code = err.get("APIErrorCode")
        desc = err.get("APIErrorDescription")
        raise RuntimeError(f"BEA API error {code}: {desc}")

    return extract_rows(j)


def retry_delay_seconds(response: requests.Response | None, attempt: int) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                value = float(str(retry_after).strip())
                if value > 0:
                    return min(MAX_BACKOFF_SECONDS, value)
            except Exception:
                pass
    return min(MAX_BACKOFF_SECONDS, BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)))


def build_request_url(params: Dict[str, Any]) -> str:
    try:
        req = requests.Request("GET", ENDPOINT, params=params)
        prepared = req.prepare()
        return prepared.url or ENDPOINT
    except Exception:
        return ENDPOINT


def request_json_with_backoff(params: Dict[str, Any], label: str) -> Dict[str, Any]:
    last_error: Exception | None = None
    request_url = build_request_url(params)
    for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
        try:
            resp = requests.get(ENDPOINT, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= MAX_REQUEST_ATTEMPTS:
                break
            sleep_for = retry_delay_seconds(None, attempt)
            print(
                f"  {label}: request error ({exc}); retrying in {sleep_for:.1f}s "
                f"[attempt {attempt}/{MAX_REQUEST_ATTEMPTS}] | URL: {request_url}",
                flush=True,
            )
            time.sleep(sleep_for)
            continue

        if resp.status_code == 429 or 500 <= resp.status_code <= 599:
            last_error = requests.HTTPError(
                f"{resp.status_code} {resp.reason}",
                response=resp,
            )
            if attempt >= MAX_REQUEST_ATTEMPTS:
                break
            sleep_for = retry_delay_seconds(resp, attempt)
            print(
                f"  {label}: HTTP {resp.status_code}; retrying in {sleep_for:.1f}s "
                f"[attempt {attempt}/{MAX_REQUEST_ATTEMPTS}] | URL: {resp.url or request_url}",
                flush=True,
            )
            time.sleep(sleep_for)
            continue

        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(f"{label}: {exc}; URL: {resp.url or request_url}") from exc

        return resp.json()

    raise RuntimeError(
        f"{label}: exhausted retries after {MAX_REQUEST_ATTEMPTS} attempts; "
        f"last error: {last_error}; URL: {request_url}"
    )


def fetch_niunderlyingdetail_metadata() -> List[Dict[str, Any]]:
    variants = [
        {"datasetname": METADATA_DATASET, "dataset": DATASET},
        {"DataSetName": METADATA_DATASET, "Dataset": DATASET},
    ]
    errors: List[str] = []

    for extra_params in variants:
        params = {
            "UserID": API_KEY,
            "method": "GetData",
            "ResultFormat": "JSON",
            **extra_params,
        }

        try:
            j = request_json_with_backoff(params, label=f"{METADATA_DATASET} {extra_params}")

            err = extract_error(j)
            if err:
                code = err.get("APIErrorCode")
                desc = err.get("APIErrorDescription")
                errors.append(f"{extra_params}: BEA API error {code}: {desc}")
                continue

            rows = flatten_metadata_rows(j, DATASET)
            if rows:
                return rows

            # Fallback to generic extractor in case BEA changes metadata shape.
            try:
                generic_rows = extract_rows(j)
                if generic_rows:
                    return generic_rows
            except Exception:
                pass

            beaapi = (j or {}).get("BEAAPI") or {}
            bea_keys = ",".join(sorted(beaapi.keys())) if isinstance(beaapi, dict) else "N/A"
            errors.append(
                f"{extra_params}: request succeeded but no parseable metadata rows "
                f"(BEAAPI keys: {bea_keys})"
            )
        except Exception as exc:
            errors.append(f"{extra_params}: {exc}")

    joined = "\n  - ".join(errors)
    raise RuntimeError(f"Failed to fetch NIUnderlyingDetail metadata.\n  - {joined}")


def write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Union of keys across rows (preserve a reasonable, stable order if present)
    preferred_order = [
        "Dataset",
        "TableName",
        "Table",
        "LineNumber",
        "ParentLineNumber",
        "Tier",
        "Path",
        "LineDescription",
        "TimePeriod",
        "Year",
        "Frequency",
        "Unit",
        "DataValue",
        "NoteRef",
        "SeriesCode",
        "MetricName",
        "CL_UNIT",
        "CL_UNIT_MULT",
    ]
    keys = set()
    for row in rows:
        keys.update(row.keys())

    fieldnames = [k for k in preferred_order if k in keys] + sorted([k for k in keys if k not in preferred_order])

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"CSV not found: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def pick_field(row: Dict[str, Any], *names: str) -> Any:
    for n in names:
        if n in row:
            return row[n]
    lower = {k.lower(): v for k, v in row.items()}
    for n in names:
        v = lower.get(n.lower())
        if v is not None:
            return v
    return None


def normalize_table_name(value: Any) -> str | None:
    if value in (None, ""):
        return None
    s = str(value).strip()
    if not s:
        return None
    s_up = s.upper()
    if s_up.startswith("U") and s_up[1:].isdigit():
        return s_up
    if s.isdigit():
        return f"U{s}"
    return s_up


def parse_line_number(value: Any) -> int | None:
    if value in (None, ""):
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return int(s)
    except Exception:
        return None


def resolve_table_name_from_row(row: Dict[str, Any], wanted_tables: Set[str]) -> str | None:
    for candidate in (
        pick_field(row, "TableName", "Table"),
        pick_field(row, "TableID", "TableId"),
    ):
        normalized = normalize_table_name(candidate)
        if normalized is not None and normalized in wanted_tables:
            return normalized
    return None


def extract_metadata_line_sets(metadata_rows: List[Dict[str, Any]], tables: List[str]) -> Dict[str, Set[int]]:
    target_tables = [normalize_table_name(t) for t in tables]
    target_tables = [t for t in target_tables if t is not None]
    wanted_tables = set(target_tables)

    out: Dict[str, Set[int]] = {t: set() for t in target_tables}
    for row in metadata_rows:
        table_name = resolve_table_name_from_row(row, wanted_tables)
        if table_name is None:
            continue
        line_no = parse_line_number(pick_field(row, "LineNumber"))
        if line_no is None:
            continue
        out[table_name].add(line_no)
    return out


def assert_same_line_sets(line_sets: Dict[str, Set[int]], tables: List[str]) -> Set[int]:
    if not tables:
        return set()

    base_table = tables[0]
    if base_table not in line_sets:
        raise RuntimeError(f"Missing line-set for {base_table}.")

    base = line_sets[base_table]
    for t in tables[1:]:
        if t not in line_sets:
            raise RuntimeError(f"Missing line-set for {t}.")
        cur = line_sets[t]
        if cur == base:
            continue
        missing_in_t = sorted(base - cur)
        extra_in_t = sorted(cur - base)
        raise RuntimeError(
            "LineNumber mismatch between "
            f"{base_table} and {t}: "
            f"{base_table} has {len(base):,} unique lines, {t} has {len(cur):,}. "
            f"Missing in {t} (first 12): {missing_in_t[:12]} | "
            f"Extra in {t} (first 12): {extra_in_t[:12]}"
        )
    return base


def build_hierarchy_rows(
    metadata_rows: List[Dict[str, Any]],
    tables: List[str],
    hierarchy_source_table: str,
) -> List[Dict[str, Any]]:
    target_tables = [normalize_table_name(t) for t in tables]
    target_tables = [t for t in target_tables if t is not None]
    wanted_tables = set(target_tables)
    source_table = normalize_table_name(hierarchy_source_table)
    if source_table is None:
        raise RuntimeError("Invalid hierarchy source table.")
    if source_table not in wanted_tables:
        raise RuntimeError(
            f"Hierarchy source table {source_table} is not in target tables: {sorted(wanted_tables)}"
        )

    by_table: Dict[str, Dict[int, Dict[str, Any]]] = {t: {} for t in target_tables}

    for row in metadata_rows:
        table_name = resolve_table_name_from_row(row, wanted_tables)
        if table_name is None:
            continue

        line_no = parse_line_number(pick_field(row, "LineNumber"))
        if line_no is None:
            continue

        # Keep first row per (table, line); metadata should be unique at this grain.
        by_table[table_name].setdefault(line_no, row)

    source_rows = by_table.get(source_table, {})
    if not source_rows:
        raise RuntimeError(f"No metadata rows found for hierarchy source table {source_table}.")

    source_lines = set(source_rows.keys())
    for t in target_tables:
        cur_lines = set(by_table.get(t, {}).keys())
        if cur_lines != source_lines:
            missing_in_t = sorted(source_lines - cur_lines)
            extra_in_t = sorted(cur_lines - source_lines)
            raise RuntimeError(
                "Metadata LineNumber mismatch between "
                f"{source_table} and {t}: "
                f"{source_table} has {len(source_lines):,}, {t} has {len(cur_lines):,}. "
                f"Missing in {t} (first 12): {missing_in_t[:12]} | "
                f"Extra in {t} (first 12): {extra_in_t[:12]}"
            )

    out: List[Dict[str, Any]] = []
    for line_no in sorted(source_lines):
        src = source_rows[line_no]
        out.append(
            {
                "LineNumber": str(line_no),
                "ParentLineNumber": pick_field(src, "ParentLineNumber"),
                "Tier": pick_field(src, "Tier"),
                "Path": pick_field(src, "Path"),
                "LineDescription": pick_field(src, "LineDescription"),
                "SeriesCode": pick_field(src, "SeriesCode"),
            }
        )

    return out


def main() -> int:
    out_dir = Path(".").resolve()
    metadata_path = out_dir / f"{DATASET}_{METADATA_DATASET}.csv"
    if RETRIEVE_API_DATA:
        for t in TABLES:
            print(f"Fetching {DATASET} {t} (Frequency={FREQUENCY}, Year={YEAR}) ...", flush=True)
            rows = fetch_table(t)
            out_path = out_dir / f"{DATASET}_{t}_freq-{FREQUENCY}_year-{YEAR}.csv"
            print(f"  -> {len(rows):,} rows; writing {out_path.name}", flush=True)
            write_csv(rows, out_path)

            # Be polite to the API
            time.sleep(2.5)

        print(f"Fetching {METADATA_DATASET} for dataset={DATASET} ...", flush=True)
        metadata_rows = fetch_niunderlyingdetail_metadata()
        print(f"  -> {len(metadata_rows):,} rows; writing {metadata_path.name}", flush=True)
        write_csv(metadata_rows, metadata_path)
    else:
        print(
            f"Skipping API retrieval (RETRIEVE_API_DATA=False); loading {metadata_path.name} ...",
            flush=True,
        )
        metadata_rows = read_csv(metadata_path)
        print(f"  -> loaded {len(metadata_rows):,} metadata rows from disk", flush=True)

    metadata_line_sets = extract_metadata_line_sets(metadata_rows, TABLES)
    common_lines = assert_same_line_sets(metadata_line_sets, TABLES)
    print(
        "Verified metadata LineNumber parity across "
        f"{', '.join(TABLES)}: {len(common_lines):,} unique lines.",
        flush=True,
    )

    hierarchy_rows = build_hierarchy_rows(
        metadata_rows,
        TABLES,
        hierarchy_source_table=HIERARCHY_SOURCE_TABLE,
    )
    hierarchy_path = out_dir / f"{DATASET}_hierarchy.csv"
    print(
        f"  -> {len(hierarchy_rows):,} hierarchy rows; writing {hierarchy_path.name} "
        f"(hierarchy sourced from {HIERARCHY_SOURCE_TABLE})",
        flush=True,
    )
    write_csv(hierarchy_rows, hierarchy_path)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
