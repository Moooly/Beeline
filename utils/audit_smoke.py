"""Opt-in Docker smoke checks using isolated synthetic inputs, never user projects.

Run from the checkout: python utils/audit_smoke.py [algorithm ...]
Logs and raw outputs are retained in the printed temporary directory.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from BLRunner import RUNNERS, execute_runner


def main():
    repo = Path(__file__).resolve().parents[1]
    root = Path(tempfile.mkdtemp(prefix='beeline audit '))
    print('ARTIFACTS ' + str(root), flush=True)
    source = root/'input'/'fixture'/'run'
    source.mkdir(parents=True)
    rng = np.random.default_rng(42)
    cells = ['cell'+str(i) for i in range(80)]
    genes = ['G'+str(i) for i in range(12)]
    t = np.linspace(0, 1, len(cells))
    expression = rng.lognormal(1, .4, (len(genes),len(cells)))
    for i in range(len(genes)):
        expression[i] += 3 * np.sin((i % 4 + 1)*t + i)**2
    pd.DataFrame(expression,index=genes,columns=cells).to_csv(source/'ExpressionData.csv')
    pd.DataFrame({'PseudoTime1':t},index=cells).to_csv(source/'PseudoTime.csv')
    entries={entry['algorithm_id']:entry for entry in yaml.safe_load((repo/'config-files/config.yaml').read_text())['input_settings']['algorithms']}
    entries.update({'JUMP3': {'image':'jump3:base'}, 'SCSGL': {'image':'scsgl:base'}})
    overrides={
        'SCODE':{'z':2,'nIter':10,'nRep':2},
        'GENIE3':{'nEstimators':20,'seed':42},
        'GRNBOOST2':{'nEstimators':30,'seed':42},
        'SCSGL':{'pos_density':.2,'neg_density':.2,'assoc':'correlation'},
        'PPCOR':{'pVal':1}, 'GRISLI':{'L':2,'R':10},
        'SINCERITIES':{'nBins':10},
        'SCRIBE':{'method':'RDI','delay':'1','ignorePT':False},
        'SINGE':{'num_replicates':1,'dT':2,'num_lags':2},
    }
    algorithms=sys.argv[1:] or ['PEARSON','GENIE3','GRNBOOST2','PPCOR','LEAP','SCODE','SINCERITIES','GRISLI','GRNVBEM','SCRIBE','PIDC','SINGE']
    results=[]
    for name in algorithms:
        entry=entries[name]
        params={key:value[0] if isinstance(value,list) and len(value)==1 else value for key,value in entry.get('params',{}).items()}
        params.update(overrides.get(name,{})); params['maxRegulatorsPerTarget']=3
        config={'input':{'input_dir':str(root/'input')},'dataset':{'dataset_id':'fixture','run_id':'run'},
                'output_settings':{'output_dir':str(root/'output')},'algo_name':name,'image':entry['image'],
                'params':params,'resources':{'cpu_budget':2,'memory_budget_mb':5120,'trajectory_workers':1}}
        start=time.perf_counter()
        try:
            runner=RUNNERS[name](root,config)
            execute_runner(runner)
            output=pd.read_csv(runner.output_dir/'rankedEdges.csv',sep='\t')
            assert not output.empty, 'unexpected empty fixture network'
            assert not (output.Gene1==output.Gene2).any()
            assert not output.duplicated(['Gene1','Gene2']).any()
            assert output.groupby('Gene2').size().max()<=3
            assert output.EdgeWeight.abs().is_monotonic_decreasing
            result={'algorithm':name,'status':'passed','edges':len(output)}
        except Exception as exc:
            result={'algorithm':name,'status':'failed','error':str(exc)}
        result['seconds']=round(time.perf_counter()-start,3)
        results.append(result); print(json.dumps(result),flush=True)
        (root/'results.json').write_text(json.dumps(results,indent=2))
    return int(any(result['status']=='failed' for result in results))

if __name__=='__main__':
    raise SystemExit(main())
