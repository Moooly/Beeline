"""Second-audit regression fixtures."""
import json
import subprocess
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import pytest
from BLRun import runner as cache
from BLRun.celloracleRunner import CellOracleRunner
from BLRun.grnvbemRunner import GRNVBEMRunner
from BLRun.pearsonRunner import PearsonRunner
from BLRun.scribeRunner import SCRIBERunner
from BLRunner import execute_runner
from test_audit_regressions import runner_at


def test_literal_identifiers_survive_alignment_and_inference(tmp_path):
    r = runner_at(tmp_path, PearsonRunner)
    (tmp_path / r.exprData).write_text(',001,002,003\n0001,1,2,3\nNA,1,3,2\n')
    (tmp_path / r.pseudoTimeData).write_text(',pt\n003,2\n001,0\n002,1\n')
    assert r.read_expression_data().index.tolist() == ['0001', 'NA']
    assert r.read_pseudotime_data().index.tolist() == ['001', '002', '003']
    r.run(); r.parseOutput()
    out = pd.read_csv(r.output_dir / 'rankedEdges.csv', sep='\t', keep_default_na=False, dtype=str)
    assert set(out.Gene1) == {'0001', 'NA'}


@pytest.mark.parametrize('header', ['cell,cell', 'cell,'])
def test_invalid_headers(tmp_path, header):
    r = runner_at(tmp_path, PearsonRunner)
    (tmp_path / r.exprData).write_text(',' + header + '\na,1,2\nb,3,4\n')
    with pytest.raises(ValueError, match='identifiers'):
        r.read_expression_data()


def test_cache_budget_and_mutation_isolation(tmp_path):
    r = runner_at(tmp_path, PearsonRunner)
    cache._DATAFRAME_CACHE.clear()
    with patch.object(cache, '_DATAFRAME_CACHE_BYTES', 1):
        r.read_expression_data()
        assert not cache._DATAFRAME_CACHE
    r.release_input_views()
    a = r.read_expression_data(); a.iloc[0, 0] = -999
    r.release_input_views()
    assert r.read_expression_data().iloc[0, 0] == 1
    cache._DATAFRAME_CACHE.clear()


@pytest.mark.parametrize('contents', ['', 'Parent\tChild\tScore\na\tx\n',
    'Parent\tChild\tScore\na\tx\tbad\n', 'Parent\tChild\tScore\na\tx\tinf\n'])
def test_grnvbem_failure_not_published(tmp_path, contents):
    r = runner_at(tmp_path, GRNVBEMRunner, {'maxRegulatorsPerTarget': 1})
    r.generateInputs = lambda: None
    r.run = lambda: (r.working_dir / 'outFile0.txt').write_text(contents)
    with pytest.raises(ValueError):
        execute_runner(r)
    assert not (r.output_dir / 'rankedEdges.csv').exists()
    assert r._expression_view_cache is None
    assert json.loads((r.working_dir / 'phase_timings.json').read_text())['status'] == 'Failed'


def test_grnvbem_valid_empty_network(tmp_path):
    r = runner_at(tmp_path, GRNVBEMRunner)
    (r.working_dir / 'outFile0.txt').write_text('Parent\tChild\tScore\n')
    r.parseOutput()
    assert (r.output_dir / 'rankedEdges.csv').read_text() == 'Gene1\tGene2\tEdgeWeight\n'


@pytest.mark.parametrize('invalid', ['', 'bad', 'NaN', 'inf', '1e999', '-1', '10'])
def test_grisli_rejects_invalid_rank(tmp_path, invalid):
    (tmp_path / 'genes').write_text('a\nb\nc\n')
    script = Path(__file__).resolve().parents[1] / 'Algorithms/compactRankMatrix.awk'
    result = subprocess.run(['awk', '-v', 'top_k=1', '-v', f'gene_file={tmp_path / "genes"}', '-f', str(script)],
        input=f'9,{invalid},8\n1,9,6\n5,4,9\n', text=True, capture_output=True)
    assert result.returncode != 0
    assert not result.stdout


@pytest.mark.parametrize('contents', ['', 'wrong header\n', 'Parent - Child Weight Probability Score\na\t->\tx\t1\t.5\tbad\n'])
def test_grnvbem_compactor_preserves_failure(contents):
    script = Path(__file__).resolve().parents[1] / 'Algorithms/compactEdgeList.awk'
    result = subprocess.run(['awk', '-F', '\t', '-v', 'top_k=1', '-v', 'strict_grnvbem=1', '-v', 'input_header=1',
        '-v', 'source_col=1', '-v', 'target_col=3', '-v', 'score_col=6', '-f', str(script)],
        input=contents, text=True, capture_output=True)
    assert result.returncode != 0
    assert not result.stdout


def test_celloracle_filters_before_capping(tmp_path):
    r = runner_at(tmp_path, CellOracleRunner, {'maxRegulatorsPerTarget': 2})
    (r.working_dir / 'outFile.txt').write_text(
        'Gene1\tGene2\tEdgeWeight\nx\tx\t99\nb\tx\t3\nb\tx\t3\na\tx\t-3\nc\tx\t3\nz\tx\tinf\n')
    r.parseOutput()
    actual = pd.read_csv(r.output_dir / 'rankedEdges.csv', sep='\t')
    assert actual.Gene1.tolist() == ['a', 'b']
    assert actual.EdgeWeight.tolist() == [-3, 3]


def test_scribe_budget_scales_with_data(tmp_path):
    r = runner_at(tmp_path, SCRIBERunner, {'delay': 5})
    r.cpu_budget = 4; r.memory_budget_mb = 3072
    assert r._resolve_pair_workers(1, gene_count=300, cell_count=200) == 4
    assert r._resolve_pair_workers(1, gene_count=4000, cell_count=200) < 4
    with pytest.raises(RuntimeError, match='Gene filter'):
        r._resolve_pair_workers(1, gene_count=8000, cell_count=200)


def test_singe_wrapper_propagates_failed_replicate(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'Algorithms/SINGE/SINGE.sh'
    (tmp_path / 'SINGE.sh').write_text(script.read_text())
    (tmp_path / 'run_SINGE_GLG_Test.sh').write_text('#!/bin/bash\nexit 7\n')
    (tmp_path / 'params').write_text('--replicate 1\n')
    result = subprocess.run(['bash', str(tmp_path / 'SINGE.sh'), 'runtime', 'GLG', 'data', 'genes', 'output', str(tmp_path / 'params'), '1'],capture_output=True)
    assert result.returncode == 7


@pytest.mark.parametrize('cap', [None, 1])
def test_grisli_raw_parser_also_rejects_missing_rank(tmp_path, cap):
    from BLRun.grisliRunner import GRISLIRunner
    r = runner_at(tmp_path, GRISLIRunner, {'maxRegulatorsPerTarget': cap})
    (r.working_dir / '0').mkdir()
    (r.working_dir / '0/outFile.txt').write_text('9,,8\n1,9,6\n5,4,9\n')
    with pytest.raises(ValueError, match='GRISLI rank matrix'):
        r.parseOutput()


def test_strict_grnvbem_accepts_native_header_and_signed_rows():
    script = Path(__file__).resolve().parents[1] / 'Algorithms/compactEdgeList.awk'
    result = subprocess.run(['awk', '-F', '\t', '-v', 'top_k=1', '-v', 'strict_grnvbem=1', '-v', 'input_header=1',
        '-v', 'source_col=1', '-v', 'target_col=3', '-v', 'score_col=6', '-f', str(script)],
        input='Parent - Child Weight Probability Score\na\t-\tx\t-.8\t.5\t-.4\nb\t+\tx\t.2\t.4\t.1\n', text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert 'a\tx\t-.4' in result.stdout


def test_singe_reuses_slots_without_losing_replicates_or_concurrency(tmp_path):
    from BLRun.singeRunner import SINGERunner
    r = runner_at(tmp_path, SINGERunner, {'num_replicates': 3})
    r.cpu_budget = 2; r.trajectory_workers = 2
    batches = []
    r._run_docker_batch = lambda commands, **kwargs: batches.append((commands, kwargs))
    r.run()
    commands, options = batches[1]
    assert len(commands) == 2
    assert options['max_workers'] == 2
    assert sum(command.count(' GLG ') for command in commands) == 3
    for replicate in [1, 2, 3]:
        assert sum(f'hyperparameters.txt {replicate}' in command for command in commands) == 1
    assert ' && ' in commands[0]
    assert all('SINGE.sh:ro' in command for command in commands)
