# dr2_to_dr3_conversion.py
# Robust DR2 -> DR3 mapper with streaming IO and stable synchronous TAP queries.
# Mallory: defaults to ARI mirror (no '/tap-server'); flip to ESA if you prefer.

import os
import sys
import time
import random
import gc
from typing import Optional

import pandas as pd
import requests
from astropy.table import Table
from astroquery.utils.tap.core import TapPlus

# -----------------------------
# ENDPOINT SELECTION (pick one)
# -----------------------------
TAP_URL = "https://gaia.ari.uni-heidelberg.de/tap"            # ARI mirror (no /tap-server; usually no login)
# TAP_URL = "https://gea.esac.esa.int/tap-server/tap"         # ESA official (uploads require GAIA_USER/GAIA_PASS)

# Official DR2->DR3 cross-ID table names used by ESA and mirrors.
NEIGHBOUR_TABLES = ["gaiaedr3.dr2_neighbourhood", "gaiadr3.dr2_neighbourhood"]

# -----------------------------
# TAP helpers (robust + small)
# -----------------------------
def get_tap(url: str) -> TapPlus:
    """Create a TAP client bound to a specific endpoint; login only if ESA endpoint."""
    tap = TapPlus(url=url)
    if "gea.esac.esa.int" in url:
        user, pwd = os.getenv("GAIA_USER"), os.getenv("GAIA_PASS")
        if not user or not pwd:
            raise RuntimeError(
                "GAIA_USER/GAIA_PASS must be set to use ESA uploads.\n"
                "Tip (quote $ if present): export GAIA_PASS='myP@ss$with$'"
            )
        tap.login(user=user, password=pwd)
    return tap

def upload_with_retry(tap: TapPlus, table: Table, temp_name: str,
                      max_tries: int = 5, base_sleep: float = 1.0) -> None:
    """Upload an Astropy Table; retry on HTTP 5xx."""
    for attempt in range(max_tries):
        try:
            tap.upload_table(upload_resource=table, table_name=temp_name)
            return
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response is not None else None
            if code and 500 <= code < 600:
                sleep = base_sleep * (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(sleep)
                continue
            raise
    raise RuntimeError(f"Upload failed after {max_tries} attempts at {tap.baseurl}.")

def run_adql_sync(tap: TapPlus, adql: str,
                  max_tries: int = 3, base_sleep: float = 1.0):
    """
    Run the ADQL query synchronously (avoids the async 'result' 404).
    If the service is flaky, retry the whole sync request a few times.
    """
    last = None
    for attempt in range(max_tries):
        try:
            job = tap.launch_job(adql, dump_to_file=False)  # SYNCHRONOUS
            return job.get_results()
        except requests.exceptions.HTTPError as e:
            last = e
            code = e.response.status_code if e.response is not None else None
            if code and 500 <= code < 600:
                time.sleep(base_sleep * (2 ** attempt) + random.uniform(0, 0.5))
                continue
            raise
        except Exception as e:
            last = e
            time.sleep(base_sleep * (2 ** attempt) + random.uniform(0, 0.5))
            continue
    raise last if last else RuntimeError("Synchronous TAP query failed after retries.")

def best_dr3_adql(temp_name: str, neighbour_table: str, upload_schema: str) -> str:
    """
    Official DR2->DR3 crossmatch: pick DR3 row with MIN(angular_distance).
    Prefer TAP_UPLOAD (portable), fallback to tap_upload if needed.
    """
    return f"""
    WITH best AS (
      SELECT i.tic_id, i.dr2_source_id, MIN(n.angular_distance) AS min_ang
      FROM {upload_schema}.{temp_name} AS i
      JOIN {neighbour_table} AS n ON n.dr2_source_id = i.dr2_source_id
      GROUP BY i.tic_id, i.dr2_source_id
    )
    SELECT b.tic_id, b.dr2_source_id, n.dr3_source_id
    FROM best AS b
    JOIN {neighbour_table} AS n
      ON n.dr2_source_id = b.dr2_source_id
     AND n.angular_distance = b.min_ang
    ORDER BY b.tic_id
    """

def process_chunk_one_endpoint(df_chunk: pd.DataFrame, temp_name: str) -> pd.DataFrame:
    """
    Upload + query on the configured TAP endpoint. Returns:
    ['tic_id', 'dr2_source_id', 'dr3_source_id'].
    """
    # Build upload table
    sub = df_chunk[["tic_id", "dr2_source_id"]].dropna()
    up = Table(names=["tic_id", "dr2_source_id"], dtype=["int64", "int64"])
    for tic, dr2 in zip(sub["tic_id"].astype("int64"), sub["dr2_source_id"].astype("int64")):
        up.add_row([int(tic), int(dr2)])

    tap = get_tap(TAP_URL)
    upload_with_retry(tap, up, temp_name)

    last_err: Optional[Exception] = None
    for tbl in NEIGHBOUR_TABLES:
        for upload_schema in ("TAP_UPLOAD", "tap_upload"):
            try:
                q = best_dr3_adql(temp_name, tbl, upload_schema)
                res = run_adql_sync(tap, q)  # <-- synchronous
                # best-effort cleanup (do not crash if deletion fails)
                try:
                    tap.delete_user_table(table_name=temp_name)
                except Exception:
                    pass

                df = res.to_pandas()
                if df.empty:
                    return pd.DataFrame(columns=["tic_id", "dr2_source_id", "dr3_source_id"])
                df["tic_id"] = df["tic_id"].astype("int64")
                df["dr2_source_id"] = df["dr2_source_id"].astype("int64")
                df["dr3_source_id"] = df["dr3_source_id"].astype("Int64")  # nullable int
                return df[["tic_id", "dr2_source_id", "dr3_source_id"]]
            except Exception as e:
                last_err = e
                continue

    raise last_err if last_err else RuntimeError("Query failed on TAP endpoint.")

# -----------------------------
# Streaming writer (CSV/Parquet)
# -----------------------------
def _append_csv(path: str, df_out: pd.DataFrame) -> None:
    header = not os.path.exists(path) or os.path.getsize(path) == 0
    df_out.to_csv(path, mode="a", header=header, index=False)

def _write_parquet_part(out_dir: str, part_idx: int, df_out: pd.DataFrame) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fn = os.path.join(out_dir, f"part-{part_idx:06d}.parquet")
    df_out.to_parquet(fn, index=False)

def stream_map_csv_to_output(
    in_csv: str,
    out_path: str,
    chunk_size: int = 25_000,
    sleep_s: float = 1.0,
    out_format: str = "csv",
    read_cols: tuple = ("tic_id", "dr2_source_id"),
) -> None:
    """
    Stream an input CSV with columns ['tic_id','dr2_source_id'] and write results incrementally.
    - out_format='csv': append to a single CSV file
    - out_format='parquet': write one parquet file per chunk under directory 'out_path'
    """
    reader = pd.read_csv(in_csv, names=list(read_cols), index_col = 0, header= 0, chunksize=chunk_size)

    for idx, df_chunk in enumerate(reader):
        temp_name = f"dr2_input_{int(time.time())}_{idx}"
        print(f"[chunk {idx}] TAP={TAP_URL} rows={len(df_chunk)} ...", flush=True)

        mapped = process_chunk_one_endpoint(df_chunk, temp_name=temp_name)

        if out_format.lower() == "csv":
            _append_csv(out_path, mapped)
        elif out_format.lower() == "parquet":
            _write_parquet_part(out_path, idx, mapped)
        else:
            raise ValueError("out_format must be 'csv' or 'parquet'.")

        time.sleep(sleep_s)

# -----------------------------
# Small utils + main() wrapper
# -----------------------------
def mkdir_if_doesnt_exist(outdir: str, subdir: str) -> None:
    path = os.path.join(outdir, subdir)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def main(file: str, parquet: bool = True) -> None:
    base = f"../data/TIC_and_Gaia_Dr3_ids_sector_{file[-6:-4]}"

    if parquet:
        parent = os.path.dirname(base) or "."
        subdir = os.path.basename(base)
        mkdir_if_doesnt_exist(parent, subdir)
        out_path = base  # dir for parquet parts
        stream_map_csv_to_output(
            in_csv=file,
            out_path=out_path,
            chunk_size=25_000,
            sleep_s=1.0,
            out_format="parquet",
        )
    else:
        out_path = base + ".csv"
        stream_map_csv_to_output(
            in_csv=file,
            out_path=out_path,
            chunk_size=25_000,
            sleep_s=1.0,
            out_format="csv",
        )

if __name__ == "__main__":
    try:
        indx_num = int(sys.argv[1]) + 1
    except Exception:
        print("Usage: python dr2_to_dr3_conversion.py <array_index>", flush=True)
        sys.exit(1)

    file_numbers = list(range(1, 14)) + list(range(27, 40))
    file_name = '../data/TIC_and_Gaia_ids_sector_s' + str(file_numbers[indx_num]).zfill(2) + '.csv'
    main(file_name, parquet=True)
    gc.collect()