import json, time, urllib.request, concurrent.futures as cf
B = "http://127.0.0.1:18080/"
def post(p, d, timeout=240):
    r = urllib.request.Request(B + p, data=json.dumps(d).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as x: return json.load(x)
def chat(msgs, **kw):
    return post("v1/chat/completions", dict(model="qwen3.8-27b", messages=msgs, chat_template_kwargs={"enable_thinking": False}, **kw))
out = {}
g = chat([{"role": "user", "content": "1+1=?只输出数字"}], max_tokens=4, temperature=0)
out["greedy"] = g["choices"][0]["message"]["content"].strip()
r1 = chat([{"role": "user", "content": "用一句话描写春天"}], max_tokens=40, temperature=0.7, seed=42)
r2 = chat([{"role": "user", "content": "用一句话描写春天"}], max_tokens=40, temperature=0.7, seed=42)
out["random_same"] = r1["choices"][0]["message"]["content"] == r2["choices"][0]["message"]["content"]
def one(expr, ans):
    t = time.time()
    r = chat([{"role": "user", "content": f"{expr}=?只输出数字"}], max_tokens=6, temperature=0)
    return {"text": r["choices"][0]["message"]["content"].strip(), "ok": r["choices"][0]["message"]["content"].strip() == ans, "s": round(time.time() - t, 2)}
with cf.ThreadPoolExecutor(2) as ex:
    fa = ex.submit(one, "12*5", "60"); time.sleep(2); fb = ex.submit(one, "9*9", "81")
    out["conc_a"] = fa.result(); out["conc_b"] = fb.result()
print(json.dumps(out, ensure_ascii=False))
