import pandas as pd
import pyvo
from astropy.table import Table
from pathlib import Path
import time
import itertools
import gc

# --- Two mirror endpoints (no ESA front-end involved) ---
AIP = "https://gaia.aip.de/tap"                  # Gaia@AIP TAP mirror  [1](https://gaia.aip.de/metadata/gaiaedr3/dr2_neighbourhood/)
ARI = "https://gaia.ari.uni-heidelberg.de/tap"   # ARI/GAVO Gaia TAP    [2](https://gaia.ari.uni-heidelberg.de/)

# We'll try AIP first, then ARI if AIP throws 5xx at submit:
for tap_url in (AIP, ARI):
    try:
        tap = pyvo.dal.TAPService(tap_url)
        break
    except Exception as ex:
        print(f"Failed to init TAP service {tap_url}: {ex}")
else:
    raise RuntimeError("Could not initialize any TAP service")

# --- Input (keep both columns locally; upload only dr2_source_id) ---
adql = """
SELECT n.dr2_source_id, n.dr3_source_id, n.angular_distance, n.magnitude_difference
FROM gaiadr3.dr2_neighbourhood AS n
JOIN TAP_UPLOAD.{tbl} AS u
  ON u.dr2_source_id = n.dr2_source_id
"""

outdir = Path(f"../data/dr2_to_dr3_results")
outdir.mkdir(parents=True, exist_ok=True)

def run_chunk(df, chunk_idx, tap, sec_num):
    """
    Submit one chunk, wait for completion using AsyncTAPJob.wait(),
    then fetch, reduce to one-best match, and persist.
    """
    from astropy.table import Table
    import pyvo
    import time

    # Upload as VOTable (reliable for TAP_UPLOAD)
    up = Table.from_pandas(df[["dr2_source_id"]])
    table_name = "dr2_ids_chunk"

    # We'll try primary, then fail over once if submit throws (e.g., 5xx)
    submit_attempts = [(tap, "primary")]
    AIP = "https://gaia.aip.de/tap"
    ARI = "https://gaia.ari.uni-heidelberg.de/tap"
    if tap.baseurl.rstrip("/") == AIP.rstrip("/"):
        submit_attempts.append((pyvo.dal.TAPService(ARI), "failover"))
    else:
        submit_attempts.append((pyvo.dal.TAPService(AIP), "failover"))

    last_err = None
    job = None
    for t, label in submit_attempts:
        try:
            job = t.submit_job(
                adql.format(tbl=table_name),
                uploads={table_name: up}
            )
            print(f"[chunk {chunk_idx:04d}] submitted on {label} ({t.baseurl})")
            break
        except Exception as ex:
            last_err = ex
            print(f"[chunk {chunk_idx:04d}] submit failed on {label} ({t.baseurl}): {ex}")
            time.sleep(2)

    if job is None:
        raise last_err

    # Start the job
    job.run()

    # ---- Poll using wait() (works across pyvo versions) ----
    try:
        # Wait up to ~4 minutes; adjust as you like
        job.wait(phases=["COMPLETED", "ERROR", "ABORTED"], timeout=240)
    except Exception as ex:
        # If wait() itself fails (rare), fall back to a small manual sleep and proceed
        print(f"[chunk {chunk_idx:04d}] wait() raised: {ex} -- continuing to check phase")

    # Check final phase
    phase = getattr(job, "phase", None)  # property that fetches state
    if phase not in ("COMPLETED",):
        # Try to surface server message if available
        err_txt = ""
        try:
            err_txt = job.error_summary.message.content
        except Exception:
            pass
        try:
            job.delete()
        except Exception:
            pass
        raise RuntimeError(f"Job not completed (phase={phase}). {err_txt or ''}")

    # Fetch results
    res = job.fetch_result().to_table().to_pandas()

    # One-best per DR2 (closest, then |ΔG|)
    if not res.empty:
        res.sort_values(["dr2_source_id", "angular_distance", "magnitude_difference"], inplace=True)
        res = res.groupby("dr2_source_id", as_index=False).first()

    # Persist per chunk
    res.to_parquet(outdir / f"TIC_and_Gaia_Dr3_ids_sector_{sec_num}_dr2_to_dr3_chunk_{chunk_idx:04d}.parquet", index=False)

    # Clean up server-side job
    try:
        job.delete()
    except Exception:
        pass

    return len(res)

def main(file: str) -> None:

    reader = pd.read_csv(
        file,
        names=["tic_id", "dr2_source_id"],
        dtype={"tic_id": "int64", "dr2_source_id": "int64"},
        index_col = 0, header= 0, 
        chunksize=25000)
    
    total_rows = 0
#     base = f"TIC_and_Gaia_Dr3_ids_sector_{file[-6:-4]}_"
    
    for idx, df in enumerate(reader, start=1):
        try:
            n = run_chunk(df, idx, tap, file[-6:-4])
            total_rows += n
        except Exception as ex:
            print(f"[chunk {idx:04d}] ERROR: {ex}  -- Will continue with next chunk.")
            # Optional: add backoff or a requeue mechanism here
    print(f"Done. Wrote {len(list(outdir.glob('*.parquet')))} chunk files; {total_rows} rows of matches.")

    
if __name__ == "__main__":
#     try:
#         indx_num = int(sys.argv[1]) + 1
#     except Exception:
#         print("Usage: python dr2_to_dr3_conversion.py <array_index>", flush=True)
#         sys.exit(1)

    file_numbers = list(range(1, 14)) + list(range(27, 40))
    # indx_num = 0
    
    for num in file_numbers:
        file_name = '../data/TIC_and_Gaia_ids_sector_s' + str(num).zfill(2) + '.csv'
        main(file_name)
        gc.collect()

    
# def main(file: str) -> None:
#     base = f"../data/TIC_and_Gaia_Dr3_ids_sector_{file[-6:-4]}"

#     xmatch_dr2_to_dr3_with_tic(
#         input_csv=file,
#         out_best_csv=base + ".csv",
#         neighbourhood=False,         # set True if you also want the full candidate list
#         chunk_size=20,          # tune if you see timeouts
#         max_retries=5,
#         base_sleep=10.0,
#         resume=True,
#         login=True
#     )
