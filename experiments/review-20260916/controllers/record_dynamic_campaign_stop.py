"""Write the first campaign failure marker atomically; never overwrite it."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import time


def record(path, payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=".STOP-",suffix=".tmp",dir=path.parent)
    try:
        with os.fdopen(fd,"w") as out:
            json.dump(payload,out,indent=2,sort_keys=True);out.write("\n");out.flush();os.fsync(out.fileno())
        try:os.link(name,path)  # Same-directory atomic create-if-absent.
        except FileExistsError:return False
        return True
    finally:
        Path(name).unlink(missing_ok=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("path","manifest-id","index","job-id","exit-code","signal","command"):
        p.add_argument("--"+name,required=True)
    a=p.parse_args()
    payload=dict(status="campaign_stopped",manifest_id=a.manifest_id,array_index=a.index,
                 job_id=a.job_id,exit_code=int(a.exit_code),signal=a.signal,command=a.command,
                 recorded_unix=time.time(),automatic_retry=False)
    created=record(a.path,payload)
    print(json.dumps(dict(stop_marker=a.path,created=created,existing_marker_preserved=not created)))
if __name__=="__main__":main()
