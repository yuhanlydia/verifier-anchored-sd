import json
import os
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

root = Path("/root/verifier-anchored-sd")
os.chdir(root)
run = root / "results/paper_vs_subspace_2026-09-09/run_2026-09-11/attempt_04_E_balanced_offload"
run.mkdir(parents=True, exist_ok=True)
status_path = run / "status.json"
status = {
    "status": "starting",
    "started_utc": datetime.now(timezone.utc).isoformat(),
    "reason": "resume after profiling conservative offload; exact BF16 with 10/7 GiB GPU residency",
    "command": ["bash", "data/paper_vs_subspace_2026-09-11/resume_after_D.sh"],
    "scientific_parameters_changed": False,
    "residency_changed": True,
    "supervisor_pid": os.getpid(),
}

def save():
    temporary = status_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n")
    temporary.replace(status_path)

save()
env = os.environ.copy()
env.update({
    "PYTHON": str(root / ".venv/bin/python"),
    "PYTHONUNBUFFERED": "1",
    "OMP_NUM_THREADS": "8",
    "MKL_NUM_THREADS": "8",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
})
with (run / "suite.log").open("w") as log:
    process = subprocess.Popen(
        status["command"], env=env, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    status.update(status="running", runner_pid=process.pid)
    save()
    while process.poll() is None:
        status["heartbeat_utc"] = datetime.now(timezone.utc).isoformat()
        status["disk_free_gib"] = round(shutil.disk_usage(root).free / 2**30, 2)
        if status["disk_free_gib"] < 5:
            status["stop_reason"] = "disk headroom below 5 GiB"
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
            break
        save()
        time.sleep(30)
    code = process.wait()
status.update(
    status="completed" if code == 0 else "incomplete",
    exit_code=code,
    finished_utc=datetime.now(timezone.utc).isoformat(),
    scientific_results_complete=(code == 0),
)
save()
