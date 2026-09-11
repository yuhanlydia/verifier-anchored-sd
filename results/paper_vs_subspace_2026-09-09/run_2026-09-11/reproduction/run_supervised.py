import json, os, subprocess, hashlib, shutil, time, signal
from pathlib import Path
from datetime import datetime, timezone
root=Path('/root/verifier-anchored-sd'); os.chdir(root)
data=root/'data/paper_vs_subspace_2026-09-11'
results=root/'results/paper_vs_subspace_2026-09-09'
run=results/'run_2026-09-11'; run.mkdir(parents=True,exist_ok=True)
manifest=json.loads((data/'manifest.json').read_text())
for s,m in manifest['splits'].items():
    assert hashlib.sha256(Path(m['path']).read_bytes()).hexdigest()==m['sha256']
    assert sum(1 for _ in open(m['path']))==256
shutil.copy2(data/'manifest.json',run/'data_manifest.json')
env=os.environ.copy()
env.update({'CALIBRATION_TEXT':str(data/'A.jsonl'),'MAPPER_EVAL_TEXT':str(data/'B.jsonl'),'SUBSPACE_TEXT':str(data/'C.jsonl'),'EVAL_TEXT':str(data/'D.jsonl'),'E2_TEXT':str(data/'E.jsonl'),'GPU_MEMORY_GIB':'20','METHOD_BATCH_SIZE':'8','SUITE_PROFILE':'pilot','PYTHON':str(root/'.venv/bin/python'),'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8','TOKENIZERS_PARALLELISM':'false','HF_HUB_DISABLE_PROGRESS_BARS':'1'})
keys=['CALIBRATION_TEXT','MAPPER_EVAL_TEXT','SUBSPACE_TEXT','EVAL_TEXT','E2_TEXT','GPU_MEMORY_GIB','METHOD_BATCH_SIZE','SUITE_PROFILE','PYTHON','OMP_NUM_THREADS','MKL_NUM_THREADS']
status={'status':'starting','started_utc':datetime.now(timezone.utc).isoformat(),'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'command':['bash','scripts/run_paper_vs_subspace_strict.sh'],'configuration':{k:env[k] for k in keys},'gpu':'RTX 3090 24GiB','supervisor_pid':os.getpid(),'scientific_results_complete':False}
def save():
    temp=run/'status.tmp'; temp.write_text(json.dumps(status,indent=2)+'\n'); temp.replace(run/'status.json')
save()
with (run/'suite.log').open('w') as log:
    proc=subprocess.Popen(status['command'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    status.update(status='running',runner_pid=proc.pid); save()
    while proc.poll() is None:
        status['heartbeat_utc']=datetime.now(timezone.utc).isoformat()
        status['disk_free_gib']=round(shutil.disk_usage(root).free/2**30,2)
        if status['disk_free_gib']<5:
            status['stop_reason']='disk headroom below 5 GiB; stopped without altering experiment settings'
            os.killpg(proc.pid,signal.SIGTERM)
            try: proc.wait(timeout=30)
            except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL)
            break
        save(); time.sleep(30)
    code=proc.wait()
status.update(status='completed' if code==0 else 'incomplete',exit_code=code,finished_utc=datetime.now(timezone.utc).isoformat(),scientific_results_complete=(code==0))
save()
