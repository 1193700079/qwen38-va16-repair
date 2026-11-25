"""Compare incremental decode token probabilities with fresh-prefill equivalents."""
import json,time,urllib.request
from pathlib import Path
ROOT=Path('/home/test/qwen-repair-20261125')
def post(path,body):
    req=urllib.request.Request('http://127.0.0.1:18080/'+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=100) as r:return json.load(r)
prompt='1+1等于几？只输出数字。'
tok=post('tokenize',dict(model='qwen3.8-27b',messages=[dict(role='user',content=prompt)],add_generation_prompt=True,chat_template_kwargs={'enable_thinking':False}))
ids=tok['tokens']
base=dict(model='qwen3.8-27b',temperature=0,logprobs=5,return_tokens_as_token_ids=True)
start=time.monotonic()
seq=post('v1/completions',dict(base,prompt=ids,max_tokens=6))
print('SEQUENCE',json.dumps(seq,ensure_ascii=False),flush=True)
c=seq['choices'][0]
lp=c['logprobs'];tokens=lp['tokens']
report={'prompt':prompt,'input_ids':ids,'sequence':seq,'checks':[]}
(ROOT/'logit-parity.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
for i in range(min(len(tokens),4)):
    prefix=[]
    for t in tokens[:i]:
        if not t.startswith('token_id:'):raise ValueError('Token ID response required: '+t)
        prefix.append(int(t.split(':')[1]))
    fresh=post('v1/completions',dict(base,prompt=ids+prefix,max_tokens=1))
    f=fresh['choices'][0]['logprobs']
    entry=dict(step=i,incremental_token=tokens[i],incremental_top=lp['top_logprobs'][i],fresh_token=f['tokens'][0],fresh_top=f['top_logprobs'][0])
    report['checks'].append(entry)
    print('CHECK',json.dumps(entry,ensure_ascii=False),flush=True)
    (ROOT/'logit-parity.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print('DONE elapsed',time.monotonic()-start,flush=True)
