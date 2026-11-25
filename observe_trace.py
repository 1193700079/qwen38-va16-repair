import subprocess, threading, queue, time, json, urllib.request
from pathlib import Path
import sys
name=sys.argv[1] if len(sys.argv)>1 else 'gdnrepair-trace-001'
q=queue.Queue()
p=subprocess.Popen(['docker','logs','--follow',name],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
def reader():
    for line in p.stdout: q.put(('log',line))
    q.put(('eof',''))
def request():
    payload={'model':'qwen3.8-27b','messages':[{'role':'user','content':'你好'}],'max_tokens':4,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}}
    req=urllib.request.Request('http://127.0.0.1:18080/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=50) as r: answer=r.read().decode()
    except Exception as e: answer=repr(e)
    Path('/home/test/qwen-repair-20261125/trace-response.txt').write_text(answer)
    q.put(('answer',answer))
threading.Thread(target=reader,daemon=True).start()
deadline=time.monotonic()+150
sent=False
try:
    with open('/home/test/qwen-repair-20261125/trace-001.log','w') as log:
        while time.monotonic()<deadline:
            try: kind,s=q.get(timeout=max(.01,deadline-time.monotonic()))
            except queue.Empty: break
            if kind=='answer': print('RESPONSE',s,flush=True); break
            if kind=='eof': print('LOG_EOF',flush=True); break
            log.write(s);log.flush()
            if any(k in s for k in ['REPAIR_BOUNDARY','REPAIR_CAPTURE','Application startup complete','Traceback','Error:']):print(s.rstrip(),flush=True)
            if not sent and 'Application startup complete' in s:
                sent=True; deadline=time.monotonic()+60
                threading.Thread(target=request,daemon=True).start()
finally:
    p.terminate()
    try:p.wait(timeout=5)
    except subprocess.TimeoutExpired:p.kill();p.wait()
print('OBSERVER_DONE request_sent=',sent,flush=True)
