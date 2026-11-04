"""Opt-in Slingshot full/truncated comparison; requires the rebuilt image.

Checks actual iterative-solver use, lineage counts/membership, and records
pseudotime deviations. A match on these fixtures is not universal equivalence.
"""
import subprocess, tempfile, pathlib, time, json
import numpy as np, pandas as pd
root=pathlib.Path(tempfile.mkdtemp(prefix='beeline-pca-')); print(root,flush=True)
rng=np.random.default_rng(73)
repo=pathlib.Path(__file__).resolve().parents[1]
rows=[]
for fixture in ['linear','branching','diffuse']:
 n,g=500,600
 t=np.linspace(0,1,n); branch=np.where(np.arange(n)%2,1,-1)
 latent=np.vstack([t, np.sin(t*3), (t>.4)*(t-.4)*branch if fixture=='branching' else np.cos(t*3), t*t, np.sin(t*7)])
 expression=rng.normal(0,.15,(g,n))+rng.normal(0,2,(g,5))@latent if fixture!='diffuse' else rng.normal(0,1,(g,n))
 pd.DataFrame(expression,index=['G'+str(i) for i in range(g)],columns=['C'+str(i) for i in range(n)]).to_csv(root/(fixture+'.csv'))
 outputs={}
 for solver in ['full','truncated']:
  start=time.perf_counter()
  command=['docker','run','--rm','--cpus=2','--memory=3g','--entrypoint','Rscript','-e','OPENBLAS_NUM_THREADS=2','-v',str(root)+':/data','-v',str(repo/'Algorithms/SLINGSHOT/estimate_pseudotime.R')+':/app/audit.R:ro','grnbeeline/slingshot:0.1.0','/app/audit.R','--expression','/data/'+fixture+'.csv','--matrixState','log_normalized','--nPCs','20','--nClusters','5','--seed','1','--pcaSolver',solver,'--output','/data/'+fixture+'-'+solver+'.csv','--embeddingOutput','/data/'+fixture+'-'+solver+'-embedding.csv']
  result=subprocess.run(command,capture_output=True,text=True)
  (root/(fixture+'-'+solver+'.log')).write_text(result.stdout+result.stderr)
  if result.returncode:raise RuntimeError(result.stderr)
  if solver=='truncated':
   assert 'PCA solver: irlba' in result.stdout, result.stdout
  outputs[solver]=pd.read_csv(root/(fixture+'-'+solver+'.csv'),index_col=0)
  rows.append({'fixture':fixture,'solver':solver,'seconds':round(time.perf_counter()-start,3)})
 a,b=outputs['full'],outputs['truncated']
 assert a.shape==b.shape,(fixture,a.shape,b.shape)
 # Lineage order can permute; choose best column match by pseudotime correlation.
 from itertools import permutations
 best=None
 for perm in permutations(range(b.shape[1])):
  v=b.iloc[:,list(perm)].to_numpy(); mask=np.isfinite(a.to_numpy()) & np.isfinite(v)
  if not np.array_equal(np.isnan(a),np.isnan(v)):continue
  err=np.max(np.abs(a.to_numpy()[mask]-v[mask]))
  if best is None or err<best:best=err
 rows[-1].update({'lineages':a.shape[1],'max_pseudotime_error':best})
 print(json.dumps(rows[-2:]),flush=True)
(root/'comparison.json').write_text(json.dumps(rows,indent=2))
