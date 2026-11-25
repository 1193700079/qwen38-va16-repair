import json,time,urllib.request
from pathlib import Path
base='http://127.0.0.1:18080/'
def post(path,body,timeout=160):
 req=urllib.request.Request(base+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=timeout) as r:return json.load(r)
text='档案校验码是4729。\n'+('这是一段用于设备测试的记录，请阅读并继续。\n'*90)+'\n问题：档案校验码是什么？仅输出数字。'
messages=[{'role':'user','content':text}]
toks=post('tokenize',dict(model='qwen3.8-27b',messages=messages,add_generation_prompt=True,chat_template_kwargs={'enable_thinking':False}))
print('PROMPT_TOKENS',len(toks['tokens']),flush=True)
assert len(toks['tokens'])>784,'Must cross actual hybrid page boundary'
t=time.monotonic();answer=post('v1/chat/completions',dict(model='qwen3.8-27b',messages=messages,max_tokens=8,temperature=0,chat_template_kwargs={'enable_thinking':False}),240)
row={'prompt_tokens':len(toks['tokens']),'elapsed_s':time.monotonic()-t,'response':answer}
Path('/home/test/qwen-repair-20261125/long-acceptance.json').write_text(json.dumps(row,ensure_ascii=False,indent=2))
print(json.dumps(row,ensure_ascii=False),flush=True)
assert answer['choices'][0]['message']['content'].strip()=='4729'
print('LONG_PAGE_CROSSING_PASS',flush=True)
