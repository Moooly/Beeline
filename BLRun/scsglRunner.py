import shlex
from pathlib import Path
import os
import pandas as pd

from BLRun.runner import Runner


class SCSGLRunner(Runner):
    """Concrete runner for the scSGL GRN inference algorithm."""

    def generateInputs(self):
        '''
        Function to generate desired inputs for scSGL.
        If the folder/files under self.input_dir exist,
        this function will not do anything.
        '''

        # Create ExpressionData.csv file in the created input directory
        SCSGL_EXPRESSION_FILE = self.working_dir / "ExpressionData.csv"
        if not SCSGL_EXPRESSION_FILE.exists():
            # input data
            ExpressionData = self.read_expression_data()

            # Write gene expression data in SCSGL folder
            ExpressionData.to_csv(SCSGL_EXPRESSION_FILE,
                                 sep = ',', header  = True)

    def run(self):
        '''
        Function to run SCSGL algorithm
        '''

        pos_density = str(self.params['pos_density'])
        neg_density = str(self.params['neg_density'])
        assoc = str(self.params['assoc'])

        script_path = Path(__file__).resolve().parents[1] / 'Algorithms/SCSGL/run_scSGL.py'
        cmdToRun = ' '.join(['docker run --rm',
                            f"-v {shlex.quote(str(self.working_dir))}:/usr/working_dir",
                            f'-v {shlex.quote(str(script_path))}:/run_scSGL.py:ro',
                            '--expose=41269',
                            f'{self.image} /bin/sh -c \"time -v -o',
                            "/usr/working_dir/time.txt", 'python /run_scSGL.py',
                            '--expression_file=/usr/working_dir/ExpressionData.csv',
                            '--out_file=/usr/working_dir/outFile.txt',
                            '--pos_density='+pos_density, '--neg_density='+neg_density, '--assoc='+assoc,
                            '\"'])

        self._run_docker(cmdToRun)

    def parseOutput(self):
        '''
        Function to parse outputs from SCSGL.
        '''
        workDir = self.working_dir
        outFile = workDir / 'outFile.txt'

        # Quit if output file does not exist
        if not outFile.exists():
            raise FileNotFoundError(outFile)

        # Read output file
        OutDF = pd.read_csv(outFile, sep = '\t', header = 0, converters={0: str, 1: str})

        OutDF.sort_values(by="EdgeWeight", ascending=False, inplace=True)

        self._write_ranked_edges(OutDF[['Gene1', 'Gene2', 'EdgeWeight']])
