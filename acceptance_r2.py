import json,time,urllib.request,threading
from pathlib import Path
B='http://127.0.0.1:18080/'
def post(p,body,timeout=300):
    r=urllib.request.Request(B+p,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(r,timeout=timeout) as x:return json.load(x)
def chat(msgs,**kw):
    return post('v1/chat/completions',dict(model='qwen3.8-27b',messages=msgs,chat_template_kwargs={'enable_thinking':False},**kw))
res={}
# 1 greedy
t=time.monotonic();g=chat([{'role':'user','content':'1+1等于几？只输出数字。'}],max_tokens=8,temperature=0)
res['greedy']={'text':g['choices'][0]['message']['content'],'s':round(time.monotonic()-t,2)}
# 2 random reproducibility (same seed)
r1=chat([{'role':'user','content':'写一句关于春天的话。'}],max_tokens=24,temperature=0.8,seed=42)
r2=chat([{'role':'user','content':'写一句关于春天的话。'}],max_tokens=24,temperature=0.8,seed=42)
res['random_repro']={'a':r1['choices'][0]['message']['content'],'b':r2['choices'][0]['message']['content'],'same':r1['choices'][0]['message']['content']==r2['choices'][0]['message']['content']}
# 3 perf: 32 tokens decode rate
t=time.monotonic();p=chat([{'role':'user','content':'从1数到40，用逗号分隔。'}],max_tokens=48,temperature=0)
dt=time.monotonic()-t;nt=p['usage']['completion_tokens']
res['perf']={'completion_tokens':nt,'wall_s':round(dt,2),'tok_per_s':round(nt/dt,3)}
# 4 concurrent staggered (mixed prefill/decode)
out={}
def worker(k,prompt,expect):
    t=time.monotonic()
    try:
        r=chat([{'role':'user','content':prompt}],max_tokens=8,temperature=0)
        out[k]={'text':r['choices'][0]['message']['content'],'ok':expect in r['choices'][0]['message']['content'],'s':round(time.monotonic()-t,2)}
    except Exception as e:
        out[k]={'err':str(e)}
th1=threading.Thread(target=worker,args=('a','计算 12*5 等于几？只输出数字。','60'));th1.start()
time.sleep(2.0)
th2=threading.Thread(target=worker,args=('b','计算 9*9 等于几？只输出数字。','81'));th2.start()
th1.join();th2.join()
res['concurrent']=out
# 5 long context >4096
text='档案校验码是4729。\n'+('这是一段用于设备测试的记录，请阅读并继续。\n'*420)+'\n问题：档案校验码是什么？仅输出数字。'
toks=post('tokenize',dict(model='qwen3.8-27b',messages=[{'role':'user','content':text}],add_generation_prompt=True,chat_template_kwargs={'enable_thinking':False}))
t=time.monotonic();L=chat([{'role':'user','content':text}],max_tokens=8,temperature=0)
res['long']={'prompt_tokens':len(toks['tokens']),'text':L['choices'][0]['message']['content'],'s':round(time.monotonic()-t,2)}
Path('/home/test/qwen-repair-20261125/acceptance-r2.json').write_text(json.dumps(res,ensure_ascii=False,indent=2))
print(json.dumps(res,ensure_ascii=False,indent=1),flush=True)
