"""Opt-in Docker validation on a sampled copy of the bundled GSD dataset.

Original inputs remain untouched. Checks both trajectories and compares full
versus bounded output for deterministic methods. Temporary artifacts retained.
"""
import sys,tempfile,json,subprocess,time
from pathlib import Path
import numpy as np,pandas as pd,yaml
repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo))
from BLRunner import RUNNERS, execute_runner
from BLRun.edges import reduce_edges
root=Path(tempfile.mkdtemp(prefix='beeline-gsd-'));print(root,flush=True)
source=root/'inputs'/'fixture'/'run';source.mkdir(parents=True)
x=pd.read_csv(repo/'inputs/example/GSD/ex1/ExpressionData.csv',index_col=0)
t=pd.read_csv(repo/'inputs/example/GSD/ex1/PseudoTime.csv',index_col=0)
# Balanced coverage along each bundled GSD lineage, without changing the source.
keep=set()
for col in t:
 order=t[col].dropna().sort_values().index
 keep.update(order[np.linspace(0,len(order)-1,100).astype(int)])
cells=[c for c in x.columns if c in keep]
x.loc[:,cells].to_csv(source/'ExpressionData.csv');t.loc[cells].to_csv(source/'PseudoTime.csv')
entries={e['algorithm_id']:e for e in yaml.safe_load((repo/'config-files/config.yaml').read_text())['input_settings']['algorithms']}
report=[]
for name in sys.argv[1:] or ['PPCOR','LEAP','SINCERITIES','SCODE','GRISLI','SCRIBE','SINGE','PIDC']:
 frames={}
 for k in ([0,3] if name in ['PPCOR','LEAP','SINCERITIES','PIDC'] else [3]):
  e=entries[name];params={key:val[0] if isinstance(val,list) else val for key,val in e.get('params',{}).items()}
  params.update({'SCODE':{'z':2,'nIter':10,'nRep':2},'GRISLI':{'L':2,'R':10},'SCRIBE':{'delay':'1','method':'RDI'},'SINGE':{'num_replicates':1}}.get(name,{}))
  params['maxRegulatorsPerTarget']=k
  config={'input':{'input_dir':str(root/'inputs')},'dataset':{'dataset_id':'fixture','run_id':'run'},'output_settings':{'output_dir':str(root/('out'+str(k)))},'algo_name':name,'image':e['image'],'params':params,'resources':{'cpu_budget':2,'memory_budget_mb':5120,'trajectory_workers':2}}
  start=time.perf_counter()
  try:
   r=RUNNERS[name](root,config);execute_runner(r);frames[k]=pd.read_csv(r.output_dir/'rankedEdges.csv',sep='\t')
   assert not frames[k].duplicated(['Gene1','Gene2']).any()
   assert not (frames[k].Gene1==frames[k].Gene2).any()
   if k and not frames[k].empty: assert frames[k].groupby('Gene2').size().max()<=k
   result={'algorithm':name,'cap':k,'edges':len(frames[k]),'status':'passed','seconds':round(time.perf_counter()-start,3)}
  except Exception as exc:result={'algorithm':name,'cap':k,'status':'failed','error':repr(exc)}
  report.append(result);print(json.dumps(result),flush=True)
 if 0 in frames and 3 in frames:
  expected=reduce_edges([frames[0].itertuples(index=False,name=None)],3)
  pd.testing.assert_frame_equal(frames[3],expected,check_exact=False,atol=1e-12,rtol=1e-12)
  print(name+' FULL/CAPPED MATCH',flush=True)
 (root/'report.json').write_text(json.dumps(report,indent=2))

raise SystemExit(int(any(item['status']=='failed' for item in report)))
