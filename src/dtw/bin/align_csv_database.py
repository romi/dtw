#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       File author(s):
#           Jonathan Legrand <jonathan.legrand@ens-lyon.fr>
#
#       File contributor(s):
#           Fabrice Besnard <fabrice.besnard@ens-lyon.fr>
#           Christophe Godin <christophe.godin@inria.fr>
#           Jonathan Legrand <jonathan.legrand@ens-lyon.fr>
#
#       File maintainer(s):
#           Christophe Godin <christophe.godin@inria.fr>
#           Jonathan Legrand <jonathan.legrand@ens-lyon.fr>
#
#       Mosaic Inria team, RDP Lab, Lyon
# ------------------------------------------------------------------------------

import argparse
import logging
from os.path import join
from os.path import split

import numpy as np
import pandas as pd

from dtw.tasks.compare_sequences import CONSTRAINTS
from dtw.tasks.compare_sequences import DEF_BEAMSIZE
from dtw.tasks.compare_sequences import DEF_CONSTRAINT
from dtw.tasks.compare_sequences import DEF_DELINS_COST
from dtw.tasks.compare_sequences import DEF_FREE_ENDS
from dtw.tasks.compare_sequences import DEF_MAX_STRETCH
from dtw.tasks.compare_sequences import sequence_comparison
from dtw.tasks.logger import BIN_LOG_FMT
from dtw.tasks.logger import DEFAULT_LOG_LEVEL
from dtw.tasks.logger import get_logger

DESCRIPTION = """Compare the angles and inter-nodes sequences from two CSV files.

See the CSV files in `data-analysis/DB_eval_v1` for examples of expected file structure.

example:
$ cd data-analysis/DB_eval_v1
$ align_csv_database.py groundtruth.csv predicted_v0.4.csv eval_v0.4 --free_ends 0.4
"""


def parsing():
    parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('ref_csv', type=str,
                        help="Path to the CSV containing the reference sequences.")
    parser.add_argument('test_csv', type=str,
                        help="Path to the CSV containing the modified sequences.")
    parser.add_argument('xp_id', type=str,
                        help="Name of the experiment,used as prefix for the output CSV.")

    dtw_opt = parser.add_argument_group('DTW algorithm arguments')
    dtw_opt.add_argument('--constraint', type=str, default=DEF_CONSTRAINT, choices=CONSTRAINTS,
                         help=f"Type of constraint to use, '{DEF_CONSTRAINT}' by default.")
    dtw_opt.add_argument('--free_ends', type=float, nargs='+', default=DEF_FREE_ENDS,
                         help=f"Free ends values to use, specify the relaxation bounds, '{DEF_FREE_ENDS}' by default.")
    dtw_opt.add_argument('--beam_size', type=int, default=DEF_BEAMSIZE,
                         help=f"Maximum amount of distortion allowed for signal warping, '{DEF_BEAMSIZE}' by default.")
    dtw_opt.add_argument('--delins_cost', type=float, nargs=2, default=DEF_DELINS_COST,
                         help=f"Cost of deletion and insertion to use, '{DEF_DELINS_COST}' by default.")
    dtw_opt.add_argument('--max_stretch', type=int, default=DEF_MAX_STRETCH,
                         help=f"Maximum amount of stretching allowed for signal warping, '{DEF_MAX_STRETCH}' by default.")

    log_opt = parser.add_argument_group('logging arguments')
    log_opt.add_argument('--log_level', type=str, default=DEFAULT_LOG_LEVEL.lower(),
                         choices=['info', 'warning', 'error', 'critical', 'debug'],
                         help=f"Logging level to use, '{DEFAULT_LOG_LEVEL.lower()}' by default.")

    return parser


def main(args):
    logger_name = "align_csv_database"
    ref_path, _ = split(args.ref_csv)

    # Get logging level from input & convert it to corresponding numeric level value
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}")
    logger = get_logger(logger_name, join(ref_path, f"{args.xp_id}_{logger_name}.log"), numeric_level, BIN_LOG_FMT)

    logger.info("Loading CSV files...")
    ref_df = pd.read_csv(args.ref_csv)
    test_df = pd.read_csv(args.test_csv)

    ref_pids = ref_df["PlantID"].unique()  # list the available "PlantID" in reference CSV
    logger.info(f"Found {len(ref_pids)} PlantID in the reference CSV file.")
    test_pids = test_df["PlantID"].unique()  # list the available "PlantID" in test CSV
    logger.info(f"Found {len(test_pids)} PlantID in the test CSV file.")
    common_pids = list(set(ref_pids) & set(test_pids))
    logger.info(f"Found {len(common_pids)} common PlantID in the reference & test CSV file.")

    flag_kwargs = {'cum_dist_flag': False, 'bp_flag': False, 'ld_flag': False,
                   'free_ends_flag': False, 'optimal_path_flag': True,
                   'graphic_optimal_path_flag': False, 'graphic_seq_alignment': False, 'verbose': False}

    if len(args.free_ends) == 1:
        args.free_ends = args.free_ends[0]

    df = pd.DataFrame()  # returned pandas DataFrame will all alignment results
    for plant_id in common_pids:
        logger.name = logger_name
        logger.info(f"Performing sequence comparison for '{plant_id}'...")
        # Create ground-truth & predicted angles and inter-nodes arrays
        seq_ref = np.array([ref_df[ref_df["PlantID"] == plant_id]["angles"],
                            ref_df[ref_df["PlantID"] == plant_id]["Internodes"]]).T
        seq_test = np.array([test_df[test_df["PlantID"] == plant_id]["angles"],
                             test_df[test_df["PlantID"] == plant_id]["Internodes"]]).T

        # Get the max value for inter-nodes, used by `mixed_spread`
        max_ref = np.max(seq_ref[:, 1])
        max_test = np.max(seq_test[:, 1])
        # Update the keyword arguments to use with this type of distance
        mixed_kwargs = {'mixed_type': [True, False],
                        'mixed_weight': [0.5, 0.5],
                        'mixed_spread': [1, max(max_ref, max_test)]}

        df_result = sequence_comparison(seq_test, seq_ref, constraint=args.constraint, dist_type="mixed",
                                        free_ends=args.free_ends, delins_cost=args.delins_cost,
                                        max_stretch=args.max_stretch, beam_size=args.beam_size, logger=logger,
                                        **mixed_kwargs, **flag_kwargs)

        # Add a column containing name:
        df_result.insert(loc=0, column='PlantID', value=[plant_id] * len(df_result.index))
        # Append this result to the returned dataframe:
        if df.empty:
            df = df_result
        else:
            df = pd.concat([df, df_result])

    logger.name = logger_name
    out_csv = join(ref_path, f'{args.xp_id}_result.csv')
    logger.info(f"Exporting result CSV to '{out_csv}'")
    df.to_csv(out_csv, index=False)


if __name__ == '__main__':
    parser = parsing()
    args = parser.parse_args()
    main(args)
