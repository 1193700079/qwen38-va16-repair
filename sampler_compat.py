"""Correct greedy token selection for vocabularies mis-sampled by VACC sampler_v1.
Only the small logits vector is copied to CPU; transformer computation stays VACC.
Random sampling remains vendor code and is NOT covered by this repair's guarantee.
"""
import torch

def sample_vacc_compat(logits, top_p, top_k, temperature, all_greedy, all_random, generators):
    if all_greedy:
        # Verified against complete 248320-token vocabulary; int32 is vendor ABI.
        ids=logits.detach().to(device='cpu',dtype=torch.float32).argmax(dim=-1)
        return ids.to(device=logits.device,dtype=torch.int32), None
    import torch_vacc
    return torch_vacc.vacc.sampler_v1(logits,top_p,top_k,temperature,all_greedy,all_random,generators)

def gather_logprobs_compat(logprobs,num_logprobs,token_ids):
    from vllm.v1.outputs import LogprobsTensors
    lp=logprobs.detach().to(device='cpu',dtype=torch.float32)
    ids=token_ids.detach().to(device='cpu',dtype=torch.int64).reshape(-1,1)
    chosen=lp.gather(-1,ids)
    vals,idx=lp.topk(num_logprobs,dim=-1)
    ranks=(lp>chosen).sum(-1)
    return LogprobsTensors(torch.cat((ids,idx),1).to(torch.int32),torch.cat((chosen,vals),1),ranks)
