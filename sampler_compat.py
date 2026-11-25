"""Correct token selection for vocabularies mis-sampled by VACC sampler_v1.
Only the small logits vector is copied to CPU; transformer computation stays VACC.
Greedy AND random (temperature/top_k/top_p) sampling are handled on CPU because the
VACC device multinomial/generator path segfaults and sampler_v1 mis-ranks large vocabs.
"""
import torch

def _cpu_random(lg, temp, tk, tp, generators):
    B, V = lg.shape
    out = torch.empty(B, dtype=torch.int64)
    for i in range(B):
        row = lg[i]
        t = float(temp[i])
        if t <= 1e-5:
            out[i] = row.argmax(); continue
        row = row / t
        k = int(tk[i])
        if 0 < k < V:
            thresh = row.topk(k).values[-1]
            row = row.masked_fill(row < thresh, float('-inf'))
        p = float(tp[i])
        if 0 < p < 1:
            sv, si = row.sort(descending=True)
            probs = sv.softmax(-1)
            cum = probs.cumsum(-1)
            sv = sv.masked_fill((cum - probs) >= p, float('-inf'))
            row = torch.full_like(row, float('-inf')).scatter(-1, si, sv)
        probs = row.softmax(-1)
        g = torch.Generator()
        gen = generators.get(i) if isinstance(generators, dict) else None
        if gen is not None:
            g.manual_seed(gen.initial_seed())
        out[i] = torch.multinomial(probs, 1, generator=g)[0]
    return out

def sample_vacc_compat(logits, top_p, top_k, temperature, all_greedy, all_random, generators):
    if all_greedy:
        # Verified against complete 248320-token vocabulary; int32 is vendor ABI.
        ids=logits.detach().to(device='cpu',dtype=torch.float32).argmax(dim=-1)
        return ids.to(device=logits.device,dtype=torch.int32), None
    lg = logits.detach().to(device='cpu', dtype=torch.float32)
    B = lg.shape[0]
    temp = temperature.detach().to('cpu', torch.float32).reshape(B)
    tk = top_k.detach().to('cpu', torch.int64).reshape(B)
    tp = top_p.detach().to('cpu', torch.float32).reshape(B)
    ids = _cpu_random(lg, temp, tk, tp, generators)
    return ids.to(device=logits.device, dtype=torch.int32), None

def gather_logprobs_compat(logprobs,num_logprobs,token_ids):
    from vllm.v1.outputs import LogprobsTensors
    lp=logprobs.detach().to(device='cpu',dtype=torch.float32)
    ids=token_ids.detach().to(device='cpu',dtype=torch.int64).reshape(-1,1)
    chosen=lp.gather(-1,ids)
    vals,idx=lp.topk(num_logprobs,dim=-1)
    ranks=(lp>chosen).sum(-1)
    return LogprobsTensors(torch.cat((ids,idx),1).to(torch.int32),torch.cat((chosen,vals),1),ranks)
