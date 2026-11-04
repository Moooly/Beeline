"""Opt-in exact-output comparisons and a small compaction benchmark.

Requires Docker images and numpy/pandas. Retains artifacts under a printed temp
folder. Timings describe these fixtures only, not production-wide speedups.
"""
import sys, io, json, subprocess, tempfile, time
from pathlib import Path
import numpy as np,pandas as pd
repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo))
from BLRun.edges import reduce_edges
root=Path(tempfile.mkdtemp(prefix='beeline-compare-'));print(root,flush=True)
rng=np.random.default_rng(18)
rows=[('source'+str(s),'target'+str(t),float(rng.normal())) for t in range(1000) for s in range(100)]
source=root/'edges.tsv';source.write_text(''.join(f'{s}\t{t}\t{w}\n' for s,t,w in rows))
old=root/'old.awk';old.write_bytes(subprocess.check_output(['git','-C',str(repo),'show','52851d6b:Algorithms/compactEdgeList.awk']))
report=[]
for label,path in [('old',old),('heap',repo/'Algorithms/compactEdgeList.awk')]:
 times=[]
 for _ in range(3):
  start=time.perf_counter()
  out=subprocess.check_output(['awk','-v','top_k=20','-v','source_col=1','-v','target_col=2','-v','score_col=3','-v','output_source=Gene1','-v','output_target=Gene2','-v','output_score=EdgeWeight','-f',str(path),str(source)],text=True)
  times.append(time.perf_counter()-start)
 actual=pd.read_csv(io.StringIO(out),sep='\t')
 expected=reduce_edges([rows],20)
 pd.testing.assert_frame_equal(reduce_edges([actual.itertuples(index=False,name=None)],20),expected,check_exact=False,atol=1e-14)
 report.append({'operation':'AWK100kK20','implementation':label,'seconds_median':float(np.median(times))})
print(json.dumps(report),flush=True)
# Actual Arboreto full/bounded inference comparisons, fixed seeds and identical inputs.
genes=['G'+str(i) for i in range(100)]
pd.DataFrame(rng.lognormal(1,.5,(180,len(genes))),columns=genes).to_csv(root/'expression.tsv',sep='\t')
for algo in ['GENIE3','GRNBoost2']:
 for cap in [0,5]:
  start=time.perf_counter()
  args=['docker','run','--rm','--cpus=2','--memory=3g','-e','OPENBLAS_NUM_THREADS=1','-v',str(root)+':/data','-v',str(repo/'Algorithms/ARBORETO/runArboreto.py')+':/runArboreto.py:ro','grnbeeline/arboreto:base','python','/runArboreto.py','--algo='+algo,'--inFile=/data/expression.tsv','--outFile=/data/'+algo+str(cap)+'.tsv','--nEstimators=30','--nWorkers=2','--seed=17','--topK='+str(cap)]
  result=subprocess.run(args,capture_output=True,text=True)
  (root/(algo+str(cap)+'.log')).write_text(result.stdout+result.stderr)
  if result.returncode:raise RuntimeError(result.stderr)
  report.append({'operation':algo,'top_k':cap,'seconds':time.perf_counter()-start})
 full=pd.read_csv(root/(algo+'0.tsv'),sep='\t').rename(columns={'TF':'Gene1','target':'Gene2','importance':'EdgeWeight'})
 capped=pd.read_csv(root/(algo+'5.tsv'),sep='\t').rename(columns={'TF':'Gene1','target':'Gene2','importance':'EdgeWeight'})
 pd.testing.assert_frame_equal(reduce_edges([full.itertuples(index=False,name=None)],5),reduce_edges([capped.itertuples(index=False,name=None)],5),check_exact=True)
 print(algo+' EXACT',flush=True)
(root/'report.json').write_text(json.dumps(report,indent=2))
