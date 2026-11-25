import torch, importlib.util, sys
spec = importlib.util.spec_from_file_location('sc', sys.argv[1] if len(sys.argv)>1 else 'sampler_compat.py')
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
V = 248320
torch.manual_seed(11)
base = torch.randn(2, V)
base[0, 123456] = 40.0   # row0 peaked
base[1, 777] = 40.0      # row1 peaked
def call(temp, tk, tp, seeds=None):
    gens = {}
    if seeds:
        for i,s in enumerate(seeds):
            g=torch.Generator(); g.manual_seed(s); gens[i]=g
    return sc.sample_vacc_compat(base, torch.tensor(tp), torch.tensor(tk), torch.tensor(temp),
                                 all_greedy=False, all_random=True, generators=gens)[0]
# temp=0 -> greedy even in random branch
ids = call([0.0,0.0],[0,0],[1.0,1.0])
assert ids.tolist()==[123456,777], ids
# top_k=1 -> greedy
ids = call([1.0,1.0],[1,1],[1.0,1.0])
assert ids.tolist()==[123456,777], ids
# top_p tiny -> top token
ids = call([1.0,1.0],[0,0],[0.01,0.01])
assert ids.tolist()==[123456,777], ids
# reproducibility with same seed
a = call([1.0,1.0],[0,0],[1.0,1.0],seeds=[5,6])
b = call([1.0,1.0],[0,0],[1.0,1.0],seeds=[5,6])
assert torch.equal(a,b), (a,b)
# different seed -> (almost surely) different on flat logits
flat = torch.randn(1, 4096)
def flatcall(seed):
    g=torch.Generator(); g.manual_seed(seed)
    return sc.sample_vacc_compat(flat, torch.tensor([1.0]), torch.tensor([0]), torch.tensor([1.0]),
                                 False, True, {0:g})[0].item()
draws=[flatcall(s) for s in range(40)]
assert len(set(draws))>10, draws
# top_k restricts support
for _ in range(20):
    i = call([1.0,1.0],[5,5],[1.0,1.0],seeds=[_,_])[0].item()
    top5 = base[0].topk(5).indices.tolist()
    assert i in top5, (i,top5)
print('SAMPLER_RANDOM_CPU_PASS', flush=True)
