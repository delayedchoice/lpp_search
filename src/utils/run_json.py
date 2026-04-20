import json

def upsert_run_json(run_path, payload_update):
    if run_path.exists():
        base = json.loads(run_path.read_text())
    else:
        base = {}
    base.update(payload_update)
    run_path.write_text(json.dumps(base, indent=2))

    # utils/run_json.py

def append_run_json_list(run_path, key, item):
    if run_path.exists():
        base = json.loads(run_path.read_text())
    else:
        base = {}
    lst = base.get(key, [])
    if not isinstance(lst, list):
        lst = []
    lst.append(item)
    base[key] = lst
    run_path.write_text(json.dumps(base, indent=2))