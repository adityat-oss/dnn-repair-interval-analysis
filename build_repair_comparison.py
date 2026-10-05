"""Run the tutorial's pointwise repair at output bounds 0.1 and 0.02."""
import contextlib, io, json
from pathlib import Path
source=Path('tutorial.py').read_text()
results=[]
for bound in (0.1,0.02):
    namespace={'__name__':'__main__'}
    text=source.replace('-0.1 <= symbolic_outputs',f'-{bound} <= symbolic_outputs').replace('symbolic_outputs <= 0.1',f'symbolic_outputs <= {bound}')
    with contextlib.redirect_stdout(io.StringIO()) as log:
        exec(compile(text,'tutorial.py','exec'),namespace)
    Path(f'run-{bound}.log').write_text(log.getvalue())
    st=namespace['st']; points=namespace['points']
    assert namespace['feasible'], 'Toy repair infeasible'
    with st.no_grad():
        before=namespace['dnn'](points).flatten().tolist()
        after=namespace['N'](points).flatten().tolist()
    assert all(abs(v)<=bound+1e-6 for v in after)
    results.append({'bound':bound,'inputs':points.flatten().tolist(),'before':before,'after':after})
Path('repair-comparison.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results))
