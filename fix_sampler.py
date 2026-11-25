from pathlib import Path
import ast,shutil
P=Path('/opt/venv/lib/python3.12/site-packages')
shutil.copyfile('/work/sampler_compat.py',P/'sampler_compat.py')
p=P/'vllm_vacc/vllm/v1/sample/sampler.py'
s=p.read_text()
old='''        sampled, _ = torch.vacc.sampler_v1('''
new='''        from sampler_compat import sample_vacc_compat
        sampled, _ = sample_vacc_compat('''
assert s.count(old)==1;s=s.replace(old,new)
old='''            logprobs_tensors = self.gather_logprobs(
                raw_logprobs, num_logprobs, token_ids=sampled
            )'''
new='''            from sampler_compat import gather_logprobs_compat
            logprobs_tensors = gather_logprobs_compat(
                raw_logprobs, num_logprobs, token_ids=sampled
            )'''
assert s.count(old)==1;s=s.replace(old,new);ast.parse(s);p.write_text(s)
print('PATCH_SAMPLER_APPLIED',flush=True)
