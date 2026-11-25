import torch,torch_vacc
from sampler_compat import gather_logprobs_compat
torch.vacc.set_device(0)
torch.manual_seed(12)
x=torch.randn(2,248320).log_softmax(-1)
ids=torch.tensor([124160,248319],dtype=torch.int32)
r=gather_logprobs_compat(x.to('vacc'),5,ids.to('vacc'))
# LogprobsTensors is a NamedTuple; check semantic values independent of field spelling.
indices,values,ranks=r.logprob_token_ids,r.logprobs,r.selected_token_ranks
assert torch.equal(indices[:,0],ids)
torch.testing.assert_close(values[:,0],x[torch.arange(2),ids.long()])
expected=(x>x[torch.arange(2),ids.long()].unsqueeze(-1)).sum(-1)
assert torch.equal(ranks,expected)
print('PASS LOGPROBS int32 token IDs and full vocabulary',flush=True)
