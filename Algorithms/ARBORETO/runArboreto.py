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

# GRNBoost2 stochastic gradient-boosting settings. These mirror Arboreto's
# SGBM_KWARGS defaults, so with no overrides diy(...) reproduces grnboost2(...)
# exactly. n_estimators is an upper bound; early stopping usually halts sooner.
GRNBOOST2_GBM_KWARGS = {
    'learning_rate': 0.01,
    'n_estimators': 5000,
    'max_features': 0.1,
    'subsample': 0.9,
}
# Arboreto's default early-stopping window for grnboost2().
GRNBOOST2_EARLY_STOP_WINDOW_LENGTH = 25

# GENIE3 accepts 'all' as a friendly alias for "consider every gene at a split",
# which scikit-learn spells as max_features=None.
GENIE3_MAX_FEATURES_ALIASES = {'all': None}

def parseArgs(args):
    parser = OptionParser()

    parser.add_option('', '--algo', type = 'str',
                      help='Algorithm to run. Can either by GENIE3 or GRNBoost2')

    parser.add_option('', '--inFile', type='str',
                      help='Path to input tab-separated expression SamplesxGenes file')

    parser.add_option('', '--outFile', type = 'str',
                      help='File where the output network is stored')

    parser.add_option('', '--nEstimators', type='int', default=None,
                      help='Number of trees / boosting rounds. GENIE3: forest '
                           'size. GRNBoost2: upper bound on boosting rounds.')

    parser.add_option('', '--maxFeatures', type='str', default=None,
                      help="Features considered per split. GENIE3: one of "
                           "'sqrt', 'log2' or 'all'. GRNBoost2: a fraction in "
                           "(0, 1].")

    parser.add_option('', '--learningRate', type='float', default=None,
                      help='GRNBoost2 only: boosting shrinkage / learning rate.')

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
        rf_kwargs = dict(GENIE3_RF_KWARGS)
        if opts.nEstimators is not None:
            rf_kwargs['n_estimators'] = opts.nEstimators
        if opts.maxFeatures is not None:
            rf_kwargs['max_features'] = GENIE3_MAX_FEATURES_ALIASES.get(
                opts.maxFeatures, opts.maxFeatures)

        expr = inDF.to_numpy(dtype=np.float32)
        network = diy(expr, regressor_type='RF', regressor_kwargs=rf_kwargs,
                      client_or_address=client, gene_names=inDF.columns)
        network.to_csv(opts.outFile, index=False, sep='\t')

    elif opts.algo == 'GRNBoost2':
        # float32: same rationale as GENIE3 — sklearn's tree code casts features
        # to float32 anyway, so this skips a per-tree conversion and halves the
        # matrix's memory. With no overrides, diy(regressor_type='GBM', ...) plus
        # the early-stopping window reproduces arboreto.algo.grnboost2 exactly.
        gbm_kwargs = dict(GRNBOOST2_GBM_KWARGS)
        if opts.learningRate is not None:
            gbm_kwargs['learning_rate'] = opts.learningRate
        if opts.nEstimators is not None:
            gbm_kwargs['n_estimators'] = opts.nEstimators
        if opts.maxFeatures is not None:
            gbm_kwargs['max_features'] = float(opts.maxFeatures)

        expr = inDF.to_numpy(dtype=np.float32)
        network = diy(expr, regressor_type='GBM', regressor_kwargs=gbm_kwargs,
                      client_or_address=client, gene_names=inDF.columns,
                      early_stop_window_length=GRNBOOST2_EARLY_STOP_WINDOW_LENGTH)
        network.to_csv(opts.outFile, index = False, sep = '\t')

    else:
        print("Wrong algorithm name. Should either be GENIE3 or GRNBoost2.")

if __name__ == "__main__":
    main(sys.argv)
