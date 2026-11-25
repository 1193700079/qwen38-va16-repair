import json,hashlib,time
from pathlib import Path
r=Path('/home/test/qwen-repair-20261125')
manifest=json.loads((r/'modelscope-manifest.json').read_text())
root=Path('/home/test/models/Qwen3.8-27B')
result=[];start=time.monotonic()
for f in manifest['Data']['Files']:
 if not f['Path'].endswith('.safetensors'):continue
 p=root/f['Path'];h=hashlib.sha256()
 with p.open('rb') as fd:
  while chunk:=fd.read(8*1024*1024):h.update(chunk)
 ok=p.stat().st_size==f['Size'] and h.hexdigest()==f['Sha256']
 row=dict(file=f['Path'],size=p.stat().st_size,sha256=h.hexdigest(),expected=f['Sha256'],passed=ok);result.append(row)
 print(f['Path'],'PASS' if ok else 'FAIL',flush=True)
 (r/'model-hash-verification.json').write_text(json.dumps(dict(elapsed_s=time.monotonic()-start,files=result),indent=2))
assert len(result)==18 and all(x['passed'] for x in result)
print('MODEL_HASHES_18_PASS',flush=True)
