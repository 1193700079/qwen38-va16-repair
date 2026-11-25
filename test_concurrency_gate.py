import asyncio
from concurrency_gate import SerialInferenceGate

def test_inference_serialized_until_final_stream_chunk():
    async def scenario():
        active=0; maximum=0; total=0
        first_entered=asyncio.Event(); release_first=asyncio.Event()
        async def app(scope,receive,send):
            nonlocal active,maximum,total
            if scope['path']=='/health':
                await send({'type':'http.response.body','body':b'OK'});return
            active+=1; maximum=max(maximum,active);total+=1
            await send({'type':'http.response.start','status':200,'headers':[]})
            if total==1:
                first_entered.set();await release_first.wait()
            await send({'type':'http.response.body','body':b'last','more_body':False})
            active-=1
        async def receive():return {'type':'http.request','body':b'{}'}
        async def send(msg):pass
        gate=SerialInferenceGate(app)
        scope={'type':'http','method':'POST','path':'/v1/chat/completions'}
        one=asyncio.create_task(gate(scope,receive,send));await first_entered.wait()
        two=asyncio.create_task(gate(scope,receive,send))
        # Health can finish while inference is locked.
        await asyncio.wait_for(gate({'type':'http','method':'GET','path':'/health'},receive,send),1)
        await asyncio.sleep(0)
        assert total==1
        release_first.set();await asyncio.gather(one,two)
        assert maximum==1 and total==2
    asyncio.run(scenario())

def test_lock_released_after_request_error():
    async def scenario():
        calls=0
        async def app(scope,receive,send):
            nonlocal calls
            calls+=1
            if calls==1:raise ValueError('failure')
        gate=SerialInferenceGate(app)
        scope={'type':'http','method':'POST','path':'/v1/completions'}
        async def receive():return {'type':'http.request','body':b'{}'}
        async def send(msg):pass
        try:await gate(scope,receive,send)
        except ValueError:pass
        await asyncio.wait_for(gate(scope,receive,send),1)
        assert calls==2
    asyncio.run(scenario())
