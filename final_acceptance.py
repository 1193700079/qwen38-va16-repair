"""Final candidate acceptance: default greedy, request queue and unsupported-mode guard."""
import json,time,urllib.request,urllib.error,concurrent.futures
from pathlib import Path
ROOT=Path('/home/test/qwen-repair-20261125')
BASE='http://127.0.0.1:18080'
report=[]
def call(prompt,max_tokens=16,**extra):
 body=dict(model='qwen3.8-27b',messages=[dict(role='user',content=prompt)],max_tokens=max_tokens,chat_template_kwargs={'enable_thinking':False})
 body.update(extra);start=time.monotonic()
 req=urllib.request.Request(BASE+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 try:
  with urllib.request.urlopen(req,timeout=170) as r:return r.status,json.load(r),time.monotonic()-start
 except urllib.error.HTTPError as e:return e.code,json.loads(e.read()),time.monotonic()-start

def record(name,status,data,elapsed,expected=None):
 text=data.get('choices',[{}])[0].get('message',{}).get('content','')
 ok=status==200 and (expected is None or expected in text)
 row=dict(name=name,passed=ok,status=status,elapsed_s=elapsed,text=text,usage=data.get('usage'))
 report.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
 (ROOT/'final-acceptance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
 return ok

# Neither supplies temperature: verifies image default is truly greedy.
a=call('1+1等于多少？只输出数字。',8);assert record('default_greedy',*a,expected='2')
# Two simultaneous requests must serialize across the whole response lifecycle.
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 futures=[pool.submit(call,p,8) for p in ['7加5等于多少？只输出数字。','9乘9等于多少？只输出数字。']]
 for i,(f,e) in enumerate(zip(futures,['12','81'])):assert record('queued_request_'+str(i),*f.result(),expected=e)
# Unverified non-greedy modes must be rejected before reaching the model.
status,data,elapsed=call('hello',8,temperature=.7)
assert status==400 and data['error']['type']=='unsupported_compatibility_mode'
report.append(dict(name='unsupported_sampling_guard',passed=True,status=status,elapsed_s=elapsed))
(ROOT/'final-acceptance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print('FINAL_ACCEPTANCE_PASS',flush=True)
