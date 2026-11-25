"""Validation + bounded-concurrency gate for the verified text paths.
Greedy and CPU random sampling are verified; batched decode is handled by the
compat kernels. Return explicit 400 for still-unverified paths (tools/structured/
multimodal) rather than silently mis-computing. Set GDN_MAX_CONCURRENCY=1 to
force serial execution.
"""
import asyncio
import json
import os

def validate_body(body):
    t=body.get('temperature',0)
    if t is not None and (not isinstance(t,(int,float)) or t<0):
        return 'temperature must be a non-negative number.'
    if body.get('n',1)!=1 or body.get('best_of',1)!=1:
        return 'Only one completion per request is verified.'
    if body.get('tools') or body.get('response_format') or body.get('structured_outputs'):
        return 'Structured/tool-constrained generation is not verified in this build.'
    for m in body.get('messages',[]):
        c=m.get('content')
        if isinstance(c,list) and any(x.get('type')!='text' for x in c if isinstance(x,dict)):
            return 'Only text input is verified; image/audio/video input is unavailable.'
    return None

class SerialInferenceGate:
    paths={'/v1/chat/completions','/v1/completions'}
    other_inference={'/v1/responses','/v1/messages','/inference/v1/generate','/invocations'}
    def __init__(self,app):
        self.app=app
        self.lock=asyncio.Semaphore(max(1,int(os.environ.get('GDN_MAX_CONCURRENCY','2'))))
    async def __call__(self,scope,receive,send):
        if scope.get('type')!='http' or scope.get('method')!='POST':
            return await self.app(scope,receive,send)
        path=scope.get('path')
        if path in self.other_inference:
            return await self.reject(send,'Use /v1/chat/completions or /v1/completions for this build.')
        if path not in self.paths:
            return await self.app(scope,receive,send)
        raw=b''
        while True:
            event=await receive()
            if event['type']=='http.disconnect':return
            raw+=event.get('body',b'')
            if len(raw)>4*1024*1024:return await self.reject(send,'Request exceeds 4 MiB limit.')
            if not event.get('more_body',False):break
        try:
            body=json.loads(raw)
            if not isinstance(body,dict):raise ValueError()
            error=validate_body(body)
        except (ValueError,TypeError):
            return await self.reject(send,'Expected a JSON request object.')
        if error:return await self.reject(send,error)
        sent=False
        async def replay():
            nonlocal sent
            if not sent:
                sent=True
                return {'type':'http.request','body':raw,'more_body':False}
            return await receive()
        async with self.lock:
            return await self.app(scope,replay,send)
    @staticmethod
    async def reject(send,msg):
        raw=json.dumps({'error':{'message':msg,'type':'unsupported_compatibility_mode','code':400}}).encode()
        await send({'type':'http.response.start','status':400,'headers':[(b'content-type',b'application/json')]})
        await send({'type':'http.response.body','body':raw,'more_body':False})
