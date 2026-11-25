"""Serialize verified text/greedy requests through their full streaming lifetime.
Return explicit 400 for unverified paths rather than silently mis-computing.
"""
import asyncio
import json

def validate_body(body):
    if body.get('temperature',0) not in (None,0,0.0):
        return 'This compatibility build currently validates temperature=0 only.'
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
        self.lock=asyncio.Lock()
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
