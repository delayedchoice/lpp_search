"""
Chunked DR2 -> DR3 (EDR3) mapping that preserves TIC IDs.

Input CSV columns required:
  - tic_id (int64)
  - dr2_source_id (int64)

Outputs:
  - dr2_to_dr3_best.csv               (tic_id, dr2_source_id, dr3_source_id, angular_distance, score)
  - [optional] dr2_to_dr3_neighbourhood.csv  (adds multiplicity diagnostics)

Notes:
  * Uses 'dr2_best_neighbour' for a curated 1:1 DR2->EDR3/DR3 mapping.
  * Set neighbourhood=True to also write a full candidates file.
"""

from astroquery.gaia import Gaia
from astropy.table import Table
from pathlib import Path
import pandas as pd
import numpy as np
import time
# --- knobs you can tune ---
CHUNK_SIZE   = 100_000         # try 100k; go 50k if still flaky
BASE_SLEEP   = 15.0            # backoff base (seconds)
BETWEEN_CHUNKS_SLEEP = 30.0    # fixed pause after each completed chunk
MAX_RETRIES  = 6               # more patience

# ... inside your chunk loop, after building Astropy Table `t` ...

def _run_astroquery(adql, table, max_retries=MAX_RETRIES, base_sleep=BASE_SLEEP):
    for attempt in range(1, max_retries+1):
        try:
            job = Gaia.launch_job_async(
                adql, upload_resource=table, upload_table_name="ids", dump_to_file=False
            )
            return job.get_results().to_pandas()
        except Exception as e:
            if attempt == max_retries:
                raise
            time.sleep(base_sleep * attempt + np.random.uniform(0, 8))


# --- Fallback TAP runner using mirrors (one chunk at a time) ---
import pyvo, time, numpy as np
from astropy.table import Table

ENDPOINTS = [
    "https://gaia.ari.uni-heidelberg.de/tap",  # ARI (often the most stable)
    "https://gaia.aip.de/tap",                 # AIP
    "https://gea.esac.esa.int/tap-server/tap", # ESAC (last resort)
]

def run_tap_upload_with_fallback(astropy_table, adql, max_retries=5, sleep=20):
    last = None
    for ep in ENDPOINTS:
        svc = pyvo.dal.TAPService(ep)
        for attempt in range(1, max_retries+1):
            try:
                job = svc.submit_job(adql, uploads={"ids": astropy_table})
                job.run()
                job.wait(phases=["COMPLETED"], timeout=36000)  # up to 10h
                return job.fetch_result().to_table()  # Astropy Table
            except Exception as e:
                last = e
                time.sleep(sleep * attempt + np.random.uniform(0, 12))
        # try the next mirror
    raise last

def xmatch_dr2_to_dr3_with_tic(
    input_csv,
    out_best_csv="dr2_to_dr3_best.csv",
    neighbourhood=False,
    out_neigh_csv="dr2_to_dr3_neighbourhood.csv",
    chunk_size=50000,
    max_retries=4,
    base_sleep=10.0,
    resume=True,
    login=True
):
    if login:
        try:
            Gaia.login()  # optional; improves persistence of jobs/results
        except Exception:
            pass

    # No hard row cap in results (we’re chunking anyway)
    Gaia.ROW_LIMIT = -1

    # Resume: DR2 ids already processed
    processed = set()
    if resume and Path(out_best_csv).exists():
        try:
            done = pd.read_csv(out_best_csv, usecols=["dr2_source_id"])
            processed = set(done["dr2_source_id"].astype("int64").tolist())
        except Exception:
            pass

    # Writers: manage headers once
    best_has_header = Path(out_best_csv).exists()
    neigh_has_header = Path(out_neigh_csv).exists()

    reader = pd.read_csv(
        input_csv,
        names=["tic_id", "dr2_source_id"],
        dtype={"tic_id": "int64", "dr2_source_id": "int64"},
        index_col = 0, header= 0, 
        chunksize=chunk_size
    )

    # ADQL for best-neighbour (1 result per DR2)
    adql_best = """
    SELECT
        u.tic_id,
        u.dr2_source_id,
        bn.edr3_source_id AS dr3_source_id,
        bn.angular_distance,
        bn.score
    FROM tap_upload.ids AS u
    JOIN gaiaedr3.dr2_best_neighbour AS bn
      ON bn.dr2_source_id = u.dr2_source_id
    """

    # ADQL for full neighbourhood (diagnostics)
    adql_neigh = """
    SELECT
        u.tic_id,
        u.dr2_source_id,
        nb.edr3_source_id AS dr3_source_id,
        nb.angular_distance,
        nb.score,
        nb.number_of_neighbours,
        nb.number_of_mates
    FROM tap_upload.ids AS u
    JOIN gaiaedr3.dr2_neighbourhood AS nb
      ON nb.dr2_source_id = u.dr2_source_id
    """

    for chunk in reader:
        # Skip rows already done (resume mode)
        todo = chunk[~chunk["dr2_source_id"].isin(processed)]
        if todo.empty:
            continue

        # Build upload table (int64 throughout)
        t = Table()
        t["tic_id"] = todo["tic_id"].astype(np.int64).values
        t["dr2_source_id"] = todo["dr2_source_id"].astype(np.int64).values

        # Helper: run a query with retries
        def _run_with_retries(adql, upload_table, max_retries, base_sleep):
            for attempt in range(1, max_retries + 1):
                try:
                    # job = Gaia.launch_job_async(
                    #     adql,
                    #     upload_resource=upload_table,
                    #     upload_table_name="ids",
                    #     dump_to_file=False,
                    # )
                    job = Gaia.launch_job_async(adql_best,
                        upload_resource=t,
                        upload_table_name="ids",
                        dump_to_file=False)

                    return job.get_results().to_pandas()
                except Exception:
                    if attempt == max_retries:
                        raise
                    time.sleep(base_sleep * attempt)

        # 1) Best-neighbour result
        # First try astroquery@ESAC; on failure, try PyVO mirrors:
        try:
            df_best = _run_astroquery(adql_best, t)
        except Exception:
            df_best = run_tap_upload_with_fallback(t, adql_best, max_retries=4, sleep=20)
        
        if not df_best.empty:
            df_best["tic_id"] = df_best["tic_id"].astype("int64")
            df_best["dr2_source_id"] = df_best["dr2_source_id"].astype("int64")
            df_best["dr3_source_id"] = df_best["dr3_source_id"].astype("int64")

        df_best.to_csv(
            out_best_csv,
            mode="a" if best_has_header else "w",
            index=False,
            header=not best_has_header,
        )
        best_has_header = True

        # Mark processed for resume
        processed.update(todo["dr2_source_id"].tolist())

        # 2) Optional full neighbourhood diagnostics
        if neighbourhood:
            df_nb = _run_with_retries(adql_neigh, t, max_retries, base_sleep)
            if not df_nb.empty:
                df_nb["tic_id"] = df_nb["tic_id"].astype("int64")
                df_nb["dr2_source_id"] = df_nb["dr2_source_id"].astype("int64")
                df_nb["dr3_source_id"] = df_nb["dr3_source_id"].astype("int64")

            df_nb.to_csv(
                out_neigh_csv,
                mode="a" if neigh_has_header else "w",
                index=False,
                header=not neigh_has_header,
            )
            neigh_has_header = True
        time.sleep(BETWEEN_CHUNKS_SLEEP + np.random.uniform(0, 15))

    return out_best_csv

def main(file: str) -> None:
    base = f"../data/TIC_and_Gaia_Dr3_ids_sector_{file[-6:-4]}"

    xmatch_dr2_to_dr3_with_tic(
        input_csv=file,
        out_best_csv=base + ".csv",
        neighbourhood=False,         # set True if you also want the full candidate list
        chunk_size=20,          # tune if you see timeouts
        max_retries=5,
        base_sleep=10.0,
        resume=True,
        login=True
    )
if __name__ == "__main__":
    try:
        indx_num = int(sys.argv[1]) + 1
    except Exception:
        print("Usage: python dr2_to_dr3_conversion.py <array_index>", flush=True)
        sys.exit(1)

    file_numbers = list(range(1, 14)) + list(range(27, 40))
    # indx_num = 0
    file_name = '../data/TIC_and_Gaia_ids_sector_s' + str(file_numbers[indx_num]).zfill(2) + '.csv'
    main(file_name)
    gc.collect()

