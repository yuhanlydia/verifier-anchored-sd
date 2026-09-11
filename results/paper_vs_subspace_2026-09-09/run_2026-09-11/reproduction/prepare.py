import hashlib,json,sys
from pathlib import Path
from datetime import datetime, timezone
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer
sys.path.insert(0,str(Path.cwd()/'src'))
from verifier_anchored_sd.experiment_data import token_windows_with_sources
root=Path(__file__).parent
repo='HuggingFaceFW/fineweb-edu'
rev=HfApi().dataset_info(repo).sha
print('Dataset revision',rev,flush=True)
tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-8B',revision='b968826d9c46dd6066d109eabc6255188de91218')
ds=iter(load_dataset(repo,name='sample-10BT',split='train',revision=rev,streaming=True))
seen=set(); used_rows=set(); offset=0; manifest={'dataset':repo,'config':'sample-10BT','revision':rev,'created_utc':datetime.now(timezone.utc).isoformat(),'splits':{}}
for split in 'ABCDE':
    docs=[]; start=offset
    while len(docs)<256:
        row=next(ds); offset+=1
        txt=row['text']; digest=hashlib.sha256(txt.encode()).hexdigest()
        if digest in seen: continue
        seen.add(digest)
        if len(tok.encode(txt,add_special_tokens=False))<1025: continue
        docs.append({'text':txt,'source_id':row.get('id'),'source_offset':offset-1,'text_sha256':digest})
    p=root/f'{split}.jsonl'
    with p.open('x') as f:
        for d in docs:f.write(json.dumps(d,ensure_ascii=False)+'\n')
    # Freeze and check full-size windows used by the runner before any model capture.
    n,seq=(64,512) if split in 'CE' else (128,1024)
    windows=list(token_windows_with_sources(tok,(d['text'] for d in docs),seq_len=seq,count=n))
    assert len(windows)==n
    hashes={hashlib.sha256(json.dumps(w.token_ids.tolist()).encode()).hexdigest() for w in windows}
    assert len(hashes)==n and not hashes.intersection(used_rows),'duplicate token windows'
    used_rows.update(hashes)
    manifest['splits'][split]={'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'documents':len(docs),'source_offset_start':start,'source_offset_end_exclusive':offset,'verified_windows':n,'window_tokens':seq,'window_sha256':sorted(hashes)}
    print('Frozen',split,'docs',len(docs),'source offsets',start,offset,flush=True)
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('DATA_READY',flush=True)
