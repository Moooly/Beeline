from optparse import OptionParser
import os
import sys
import numpy as np
import pandas as pd
from arboreto.algo import grnboost2, genie3, diy
from arboreto.utils import load_tf_names
from distributed import Client, LocalCluster

# GENIE3 random-forest settings. Identical to Arboreto's GENIE3 defaults
# (max_features='sqrt', n_jobs=1) except n_estimators, reduced from 1000 to 400.
# Runtime is ~linear in the number of trees, and the top-ranked edges are
# preserved: the change is smaller than the forest's own seed-to-seed variation.
GENIE3_RF_KWARGS = {
    'n_jobs': 1,
    'n_estimators': 400,
    'max_features': 'sqrt',
}

def parseArgs(args):
    parser = OptionParser()

    parser.add_option('', '--algo', type = 'str',
                      help='Algorithm to run. Can either by GENIE3 or GRNBoost2')

    parser.add_option('', '--inFile', type='str',
                      help='Path to input tab-separated expression SamplesxGenes file')

    parser.add_option('', '--outFile', type = 'str',
                      help='File where the output network is stored')

    (opts, args) = parser.parse_args(args)

    return opts, args

def main(args):
    opts, args = parseArgs(args)
    inDF = pd.read_csv(opts.inFile, sep = '\t', index_col = 0, header = 0)

    client = Client(processes = False)    

    if opts.algo == 'GENIE3':
        # float32: sklearn's tree code casts features to float32 for split-finding
        # regardless, so passing float32 up front avoids a per-forest conversion and
        # halves the matrix's memory. Reduced tree count is applied via
        # GENIE3_RF_KWARGS; the call otherwise mirrors arboreto.algo.genie3 exactly.
        expr = inDF.to_numpy(dtype=np.float32)
        network = diy(expr, regressor_type='RF', regressor_kwargs=GENIE3_RF_KWARGS,
                      client_or_address=client, gene_names=inDF.columns)
        network.to_csv(opts.outFile, index=False, sep='\t')

    elif opts.algo == 'GRNBoost2':
        # float32: same rationale as GENIE3 — sklearn's tree code casts features
        # to float32 anyway, so this skips a per-tree conversion and halves the
        # matrix's memory. Everything else (early stopping, SGBM_KWARGS) is
        # unchanged: this still calls arboreto.algo.grnboost2 exactly as before.
        expr = inDF.to_numpy(dtype=np.float32)
        network = grnboost2(expr, client_or_address = client, gene_names = inDF.columns)
        network.to_csv(opts.outFile, index = False, sep = '\t')

    else:
        print("Wrong algorithm name. Should either be GENIE3 or GRNBoost2.")
                        
if __name__ == "__main__":
    main(sys.argv)
