from pathlib import Path
import ast
s=Path('/work/sources/server_gdn_vacc_fast.py').read_text()
tree=ast.parse(s)
class Instrument(ast.NodeTransformer):
    def visit_FunctionDef(self,node):
        if node.name not in ('chunk_gated_delta_rule_fast','_chunk_core','_inv_unit_lower'):return node
        def block(body):
            out=[]
            for stmt in body:
                if isinstance(stmt,(ast.If,ast.For,ast.While)):
                    stmt.body=block(stmt.body);stmt.orelse=block(stmt.orelse)
                out.append(stmt)
                if isinstance(stmt,(ast.Assign,ast.AugAssign)):
                    label=node.name+':'+str(stmt.lineno)
                    call=ast.parse(f'_steptrace({label!r})').body[0]
                    out.append(ast.copy_location(call,stmt))
            return out
        node.body=block(node.body);return node
s=ast.unparse(ast.fix_missing_locations(Instrument().visit(tree)))
s+='''

def _steptrace(label):
    import time, os
    print("GDNSTEP before",os.getpid(),label,"default_dtype",torch.get_default_dtype(),"device",torch.vacc.current_device(),flush=True)
    torch.vacc.synchronize()
    print("GDNSTEP after",os.getpid(),label,flush=True)
'''
Path('/opt/venv/lib/python3.12/site-packages/gdn_vacc_fast.py').write_text(s)
print('Instrumented GDN local kernel per-operation',flush=True)
