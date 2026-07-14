import os
import pandas as pd

from BLRun.runner import Runner


class GENIE3Runner(Runner):
    """Concrete runner for the GENIE3 GRN inference algorithm."""

    def generateInputs(self):
        '''
        Function to generate desired inputs for GENIE3.
        If the folder/files under self.input_dir exist,
        this function will not do anything.
        '''

        # Create ExpressionData.csv file in the created input directory
        GENIE3_EXPRESSION_FILE = self.working_dir / "ExpressionData.csv"
        if not GENIE3_EXPRESSION_FILE.exists():
            # input data
            ExpressionData = pd.read_csv(self.input_dir / self.exprData,
                                         header = 0, index_col = 0)

            # Write .csv file — arboreto expects cells as rows, genes as columns
            ExpressionData.T.to_csv(GENIE3_EXPRESSION_FILE,
                                 sep = '\t', header  = True, index = True)

    def run(self):
        '''
        Function to run GENIE3 algorithm
        '''

        # Forest hyperparameters. Fall back to the entrypoint's own defaults
        # (n_estimators=400, max_features='sqrt') when unset, so standalone
        # BEELINE runs without a params map behave exactly as before.
        n_estimators = str(self.params.get('nEstimators', 400))
        max_features = str(self.params.get('maxFeatures', 'sqrt'))

        cmdToRun = ' '.join(['docker run --rm',
                            f"-v {self.working_dir}:/usr/working_dir",
                            '--expose=41269',
                            f'{self.image} /bin/sh -c \"time -v -o',
                            "/usr/working_dir/time.txt",
                            'python runArboreto.py --algo=GENIE3',
                            '--inFile=/usr/working_dir/ExpressionData.csv',
                            '--outFile=/usr/working_dir/outFile.txt',
                            f'--nEstimators={n_estimators}',
                            f'--maxFeatures={max_features}', '\"'])

        self._run_docker(cmdToRun)

    def parseOutput(self):
        '''
        Function to parse outputs from GENIE3.
        '''
        workDir = self.working_dir
        outFile = workDir / 'outFile.txt'

        # Quit if output file does not exist
        if not outFile.exists():
            print(str(outFile) + ' does not exist, skipping...')
            return

        # Read output
        OutDF = pd.read_csv(outFile, sep = '\t', header = 0)

        self._write_ranked_edges(OutDF.rename(columns={
            'TF': 'Gene1', 'target': 'Gene2', 'importance': 'EdgeWeight'
        })[['Gene1', 'Gene2', 'EdgeWeight']])
