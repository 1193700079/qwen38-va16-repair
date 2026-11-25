"""Real sequential requests: first-token latency, coherent output and slot reuse."""
import json,time,urllib.request,traceback
from pathlib import Path
out=Path('/home/test/qwen-repair-20261125/acceptance-001.json')
url='http://127.0.0.1:18080/v1/chat/completions'
cases=[
 ('greeting','你好，请用两句话介绍你能做什么。',96,lambda s:len(s)>10),
 ('arithmetic','甲水管单独注满池子需6小时，乙需4小时，两管同时开几小时注满？简洁回答计算过程。',96,lambda s:'2.4' in s or '12/5' in s),
 ('code','写一个Python函数，合并重叠区间。给出代码和时间复杂度，不要其他说明。',192,lambda s:'def ' in s and 'sort' in s),
 ('long_context','阅读以下档案后回答问题。\n秘密校验号是4729。\n'+('本段记录设备巡检正常，服务保持稳定。\n'*80)+'\n问题：档案里的秘密校验号是什么？只输出数字。',16,lambda s:'4729' in s),
 ('fresh_session','这是全新问题：7加5是多少？只输出数字。',16,lambda s:'12' in s),
]
report=[]
for name,prompt,max_tokens,judge in cases:
    start=time.monotonic();ttft=None;text='';reason='';usage=None
    body=dict(model='qwen3.8-27b',messages=[dict(role='user',content=prompt)],max_tokens=max_tokens,temperature=0,stream=True,stream_options={'include_usage':True},chat_template_kwargs={'enable_thinking':False})
    try:
        req=urllib.request.Request(url,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=120) as r:
            for raw in r:
                if not raw.startswith(b'data:'):continue
                line=raw[5:].strip()
                if line==b'[DONE]':break
                d=json.loads(line)
                if d.get('usage'):usage=d['usage']
                for choice in d.get('choices',[]):
                    chunk=choice.get('delta',{}).get('content') or ''
                    if chunk and ttft is None:ttft=time.monotonic()-start
                    text+=chunk
                    if choice.get('finish_reason'):reason=choice['finish_reason']
        passed=bool(judge(text))
        row=dict(case=name,passed=passed,elapsed_s=time.monotonic()-start,ttft_s=ttft,text=text,finish_reason=reason,usage=usage)
    except Exception as e:
        row=dict(case=name,passed=False,error=repr(e),elapsed_s=time.monotonic()-start,text=text)
    report.append(row);out.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(row,ensure_ascii=False),flush=True)
    if 'error' in row:break
print('ACCEPTANCE',sum(r['passed'] for r in report),'/',len(cases),flush=True)
raise SystemExit(0 if len(report)==len(cases) and all(r['passed'] for r in report) else 1)
