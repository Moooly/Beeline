"""Independent scientific fixtures for the November 2026 audit corrections."""
import io
import json
import random
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from BLRun.edges import reduce_edges, reduce_frame, top_indices
from BLRun.runner import Runner
from BLRun.scodeRunner import SCODERunner
from BLRun.grisliRunner import GRISLIRunner
from BLRun.scribeRunner import SCRIBERunner
from BLRun.singeRunner import SINGERunner
from BLRun.sinceritiesRunner import SINCERITIESRunner
from BLRun.pearsonRunner import PearsonRunner
from BLRunner import enabled, execute_config, execute_runner, get_working_dirs


def oracle(rows, k):
    # Deliberately use a full dataframe sort/groupby, independent of streaming heaps.
    frame = pd.DataFrame(rows, columns=['Gene1', 'Gene2', 'EdgeWeight'])
    frame = frame[(frame.Gene1 != frame.Gene2) & np.isfinite(frame.EdgeWeight)]
    frame['magnitude'] = frame.EdgeWeight.abs()
    frame = frame.sort_values(['magnitude', 'Gene1', 'Gene2', 'EdgeWeight'],
                              ascending=[False, True, True, False])
    frame = frame.drop_duplicates(['Gene1', 'Gene2'])
    if k is not None:
        frame = frame.groupby('Gene2', sort=False).head(k)
    return frame[['Gene1', 'Gene2', 'EdgeWeight']].reset_index(drop=True)


def runner_at(tmp_path, cls, params=None):
    runner = cls.__new__(cls)
    runner.params = params or {}
    runner.output_dir = tmp_path
    runner.working_dir = tmp_path / 'working_dir'
    runner.working_dir.mkdir(exist_ok=True)
    runner.input_dir = tmp_path
    runner.exprData, runner.pseudoTimeData = 'expression.csv', 'time.csv'
    runner.image = 'local'
    runner.cpu_budget=1; runner.trajectory_workers=1; runner.memory_budget_mb=None
    pd.DataFrame([[1,2,4,8,16,32], [8,2,1,3,5,0], [1,3,2,4,2,3]],
                 index=['a','b','c'], columns=list('uvwxyz')).to_csv(tmp_path / runner.exprData)
    pd.DataFrame({'pt': range(6)}, index=list('uvwxyz')).to_csv(tmp_path / runner.pseudoTimeData)
    return runner


@pytest.mark.parametrize('k', [1, 2, 7, None])
def test_streaming_reducer_matches_independent_full_oracle(k):
    rng = random.Random(19)
    names = ['1', '2', '10', 'a', 'b', 'z', 'µ']
    rows = [(rng.choice(names), rng.choice(names), float(rng.randrange(-4,5))) for _ in range(2000)]
    expected = oracle(rows, k)
    for _ in range(4):
        rng.shuffle(rows)
        actual = reduce_edges([rows[:1000], rows[1000:]], k)
        pd.testing.assert_frame_equal(actual, expected)
        pd.testing.assert_frame_equal(reduce_frame(pd.DataFrame(rows, columns=['Gene1','Gene2','EdgeWeight']), k), expected)


def test_awk_matches_oracle_with_duplicates_sign_ties_and_self_links():
    rng = random.Random(2)
    rows = [(rng.choice(['1','2','10','a','b']), rng.choice(['1','2','10','a','b']),
             float(rng.randrange(-9,10))) for _ in range(1000)]
    command = ['awk', '-v', 'top_k=3', '-v', 'source_col=1', '-v', 'target_col=2',
               '-v', 'score_col=3', '-v', 'output_source=Gene1', '-v', 'output_target=Gene2',
               '-v', 'output_score=EdgeWeight', '-f', str(Path(__file__).parents[1] / 'Algorithms/compactEdgeList.awk')]
    output = subprocess.check_output(command, input=''.join(f'{s}\t{t}\t{w}\n' for s,t,w in rows), text=True)
    actual = pd.read_csv(io.StringIO(output), sep='\t', dtype={'Gene1': str, 'Gene2': str})
    pd.testing.assert_frame_equal(oracle(actual.itertuples(index=False, name=None), 3), oracle(rows,3))


@pytest.mark.parametrize('k', [None, 1, 2])
def test_scode_derivative_matrix_rows_are_targets(tmp_path, k):
    runner = runner_at(tmp_path, SCODERunner, {'maxRegulatorsPerTarget': k})
    (runner.working_dir / '0').mkdir()
    # dx_a/dt = 99*a - 7*b + 2*c, etc. Self dynamics must not consume K.
    matrix = np.array([[99.,-7.,2.],[3.,99.,5.],[-6.,1.,99.]])
    pd.DataFrame(matrix).to_csv(runner.working_dir / '0/meanA.txt', sep='\t', header=False,index=False)
    runner.parseOutput()
    actual = pd.read_csv(tmp_path / 'rankedEdges.csv', sep='\t')
    expected = oracle([('b','a',-7.),('c','a',2.),('a','b',3.),('c','b',5.),('a','c',-6.),('b','c',1.)], k)
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize('cls', [SCRIBERunner, SINGERunner])
def test_duplicate_trajectories_do_not_consume_slots(tmp_path, cls):
    runner = runner_at(tmp_path, cls, {'maxRegulatorsPerTarget': 2})
    pd.DataFrame({'pt1':range(6),'pt2':range(6)},index=list('uvwxyz')).to_csv(tmp_path / runner.pseudoTimeData)
    for index in range(2):
        if cls is SCRIBERunner:
            path = runner.working_dir / f'outFile{index}.csv'
            path.write_text('a x 10\nb x 9\nc x 8\nx x 100\n')
        else:
            path = runner.working_dir / str(index)
            path.mkdir()
            (path / 'SINGE_Ranked_Edge_List.txt').write_text('Gene1\tGene2\tEdgeWeight\na\tx\t10\nb\tx\t9\nc\tx\t8\nx\tx\t100\n')
    runner.parseOutput()
    result = pd.read_csv(tmp_path / 'rankedEdges.csv', sep='\t')
    assert list(result.Gene1) == ['a','b']


@pytest.mark.parametrize('k', [None, 1])
def test_grisli_uses_retained_gene_order_and_size(tmp_path,k):
    runner = runner_at(tmp_path, GRISLIRunner, {'maxRegulatorsPerTarget':k})
    (runner.working_dir / 'GeneList.txt').write_text('c\na\n')
    (runner.working_dir / '0').mkdir()
    (runner.working_dir / '0/outFile.txt').write_text('4,1\n2,3\n')
    runner.parseOutput()
    result = pd.read_csv(tmp_path / 'rankedEdges.csv',sep='\t')
    assert list(result.itertuples(index=False,name=None)) == [('c','a',3.),('a','c',2.)]


def test_failed_rerun_cannot_reuse_old_result(tmp_path):
    runner = runner_at(tmp_path, PearsonRunner)
    (tmp_path/'rankedEdges.csv').write_text('Gene1\tGene2\tEdgeWeight\na\tb\t1\n')
    runner.run = lambda: None
    with pytest.raises(FileNotFoundError):
        execute_runner(runner)
    assert not (tmp_path/'rankedEdges.csv').exists()
    assert json.loads((runner.working_dir/'phase_timings.json').read_text())['status']=='Failed'


def test_header_only_result_is_valid(tmp_path):
    runner = runner_at(tmp_path, PearsonRunner)
    runner.run = lambda: (runner.working_dir/'outFile.txt').write_text('Gene1\tGene2\tEdgeWeight\n')
    assert execute_runner(runner)['status']=='Completed'


def test_parallel_memory_caps_and_failure_logs(tmp_path):
    runner = runner_at(tmp_path, PearsonRunner)
    runner.cpu_budget=8; runner.trajectory_workers=8; runner.memory_budget_mb=1024
    observed=[]
    def execute(command, **kwargs):
        observed.append(kwargs)
        kwargs['output_path'].write_text(command+'\n')
        if command=='failed': raise RuntimeError('fixture failure')
    runner._run_docker=execute
    with pytest.raises(RuntimeError): runner._run_docker_batch(['first','failed','last'])
    assert runner.execution_worker_count==2
    assert all(item['memory_budget_mb']==512 for item in observed)
    assert (tmp_path/'output.txt').read_text()=='first\nfailed\nlast\n'
    assert (tmp_path/'trajectory-1.log').read_text()=='failed\n'


@pytest.mark.parametrize('values,bins', [([0]*6,5),([0,0,0,1,1,1],10),(list(range(6)),2)])
def test_sincerities_rejects_collapsed_bins(tmp_path,values,bins):
    runner=runner_at(tmp_path,SINCERITIESRunner,{'nBins':bins})
    pd.DataFrame({'pt':values},index=list('uvwxyz')).to_csv(tmp_path/runner.pseudoTimeData)
    with pytest.raises(ValueError,match='at least 5'): runner.generateInputs()


def test_pearson_matches_pairwise_finite_correlation(tmp_path):
    runner=runner_at(tmp_path,PearsonRunner,{'maxRegulatorsPerTarget':1})
    expression=pd.read_csv(tmp_path/runner.exprData,index_col=0)
    expected=expression.T.corr()
    runner.run(); runner.parseOutput()
    for source,target,weight in pd.read_csv(tmp_path/'rankedEdges.csv',sep='\t').itertuples(index=False,name=None):
        assert weight==pytest.approx(expected.loc[source,target],abs=1e-14)
    runner._expression_view_cache=None
    expression.iloc[0,0]=np.nan; expression.to_csv(tmp_path/runner.exprData)
    with pytest.raises(ValueError,match='finite'):runner.run()


def test_deterministic_partial_selection_excludes_self():
    assert top_indices(np.array([99.,-2.,2.,np.nan]),['x','z','a','q'],1,exclude=0)==[2]


def test_boolean_config_normalization():
    for value in (True,[True]): assert enabled(value)
    for value in (False,[False]): assert not enabled(value)
    with pytest.raises(ValueError):enabled('false')


def test_config_collision_rejected_before_cleanup(tmp_path):
    config={'input_settings':{'input_dir':str(tmp_path),'datasets':[{'dataset_id':'d','runs':[{'run_id':'r'}]}],
                              'algorithms':[{'algorithm_id':'PEARSON','image':'local','should_run':True}]*2},
            'output_settings':{'output_dir':str(tmp_path/'output')}}
    with pytest.raises(ValueError,match='Duplicate'):get_working_dirs(config)
    config['input_settings']['datasets'][0]['dataset_id']='../outside'
    with pytest.raises(ValueError,match='path segment'):get_working_dirs(config)


def test_worker_retry_never_reads_input(tmp_path):
    with patch('BLRunner.load_config', return_value={}), patch('BLRunner.get_working_dirs',return_value=[tmp_path]), patch('builtins.input',side_effect=AssertionError('must not prompt')):
        (tmp_path/'prior').write_text('prior')
        with pytest.raises(RuntimeError,match='overwrite'):execute_config('unused',interactive=False)


def test_path_stats_does_not_depend_on_file_row_order():
    import networkx as nx
    from BLEval.PathStats import _compute_path_stats
    graph=nx.DiGraph([('a','x'),('b','y')])
    rows=pd.DataFrame([('a','x',.9),('c','x',.7),('b','y',.8),('d','y',.6)],
                      columns=['Gene1','Gene2','EdgeWeight'])
    assert _compute_path_stats(rows,graph)['numPred']==2
    assert _compute_path_stats(rows,graph)==_compute_path_stats(rows.iloc[[2,0,3,1]],graph)


def test_celloracle_caps_emitted_signed_score_after_self_filter():
    # Load only this pure function, without importing the optional CellOracle stack.
    import ast
    from types import SimpleNamespace
    path=Path(__file__).parents[1]/'Algorithms/CELLORACLE/runCellOracle.py'
    tree=ast.parse(path.read_text())
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='extract_edges')
    module=ast.Module(body=[function],type_ignores=[])
    namespace={'pd':pd,'np':np}
    exec(compile(module,str(path),'exec'),namespace)
    links=SimpleNamespace(links_dict={'unit':pd.DataFrame({
        'source':['x','a','b','c'],'target':['x']*4,
        'p':[0]*4,'coef_mean':[100,.1,-.8,.7],'coef_abs':[100,.9,.8,.7]})})
    result=namespace['extract_edges'](links,1,2)
    assert list(result.Gene1)==['b','c']
    assert list(result.EdgeWeight)==[-.8,.7]


def test_jump3_accepts_canonical_single_lineage_and_sorts_time(tmp_path):
    from BLRun.jump3Runner import JUMP3Runner
    runner=runner_at(tmp_path,JUMP3Runner)
    pd.DataFrame({'PseudoTime1':[5,2,1,4,3,6]},index=list('uvwxyz')).to_csv(tmp_path/runner.pseudoTimeData)
    runner.generateInputs()
    result=pd.read_csv(runner.working_dir/'ExpressionData.csv')
    assert list(result.Time)==[0,1,2,3,4,5]
    assert list(result.a)==[4,2,16,8,1,32]


def test_scsgl_generation_does_not_need_ground_truth(tmp_path):
    from BLRun.scsglRunner import SCSGLRunner
    runner=runner_at(tmp_path,SCSGLRunner)
    runner.generateInputs()
    assert (runner.working_dir/'ExpressionData.csv').is_file()


def test_resource_initialization_failure_is_recorded(tmp_path):
    from contextlib import contextmanager
    runner=runner_at(tmp_path,PearsonRunner)
    @contextmanager
    def broken_environment():
        raise RuntimeError('resource initialization failed')
        yield
    runner.resource_environment=broken_environment
    with pytest.raises(RuntimeError,match='resource initialization'):
        execute_runner(runner)
    saved=json.loads((runner.working_dir/'phase_timings.json').read_text())
    assert saved['status']=='Failed'
