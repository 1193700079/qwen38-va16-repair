import json,time,urllib.request
from pathlib import Path
cases=[('arithmetic','甲管6小时注满，乙管4小时注满。一起开几小时注满？只输出小时数。',16),('short_code','用Python写一个函数返回两个整数之和，仅输出代码。',48),('state_isolation','7加5等于多少？只输出数字。',8)]
report=[]
for name,prompt,n in cases:
 body=dict(model='qwen3.8-27b',messages=[dict(role='user',content=prompt)],temperature=0,max_tokens=n,chat_template_kwargs={'enable_thinking':False})
 req=urllib.request.Request('http://127.0.0.1:18080/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 t=time.monotonic()
 try:
  with urllib.request.urlopen(req,timeout=150) as r:data=json.load(r)
 except Exception as e:data={'error':repr(e)}
 row=dict(case=name,elapsed=time.monotonic()-t,response=data);report.append(row)
 Path('/home/test/qwen-repair-20261125/short-acceptance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
 print(json.dumps(row,ensure_ascii=False),flush=True)
 if 'error' in data:break
