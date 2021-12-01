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
from os import remove
from os.path import exists
from os.path import join

import numpy as np
from plantdb.fsdb import FSDB
from plantdb.io import read_json

from dtw.tasks.compare_sequences import CONSTRAINTS
from dtw.tasks.compare_sequences import DEF_BEAMSIZE
from dtw.tasks.compare_sequences import DEF_CONSTRAINT
from dtw.tasks.compare_sequences import DEF_DELINS_COST
from dtw.tasks.compare_sequences import DEF_DIST_TYPE
from dtw.tasks.compare_sequences import DEF_FREE_ENDS
from dtw.tasks.compare_sequences import DEF_MAX_STRETCH
from dtw.tasks.compare_sequences import DIST_TYPES
from dtw.tasks.compare_sequences import sequence_comparison
from dtw.tasks.logger import BIN_LOG_FMT
from dtw.tasks.logger import DEFAULT_LOG_LEVEL
from dtw.tasks.logger import get_logger

DESCRIPTION = """Compare the sequence obtained with the ROMI reconstruction pipeline to a manual measurement using the DTW algorithm.

The manual measures are assumed to be the ground-truth.
"""


def parsing():
    parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('db_path', type=str,
                        help="Path to a ROMI local database managed by `FSDB`.")
    parser.add_argument('scan', type=str,
                        help="Name of the plant or scan to analyse.")

    dtw_opt = parser.add_argument_group('DTW algorithm arguments')
    dtw_opt.add_argument('--constraint', type=str, default=DEF_CONSTRAINT, choices=CONSTRAINTS,
                         help=f"Type of constraint to use, '{DEF_CONSTRAINT}' by default.")
    dtw_opt.add_argument('--dist_type', type=str, default=DEF_DIST_TYPE, choices=DIST_TYPES,
                         help=f"Type of distance to use, '{DEF_DIST_TYPE}' by default.")
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
    logger_name = "romi_compare_to_manual_gt"
    # Get logging level from input & convert it to corresponding numeric level value
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}")
    logger = get_logger(logger_name, join(args.db_path, args.scan, "compare_to_manual_gt.log"),
                        numeric_level, BIN_LOG_FMT)

    lock_file = join(args.db_path, 'lock')
    if exists(lock_file):
        remove(lock_file)

    # TODO: Add SSHFSDB if given `args.db_path` is an URL ??
    db = FSDB(args.db_path)
    db.connect()
    scan = db.get_scan(args.scan)

    # Loop the fileset ids to find the one containing the required data:
    fs = None
    for sfs in scan.filesets:
        if sfs.id.startswith("AnglesAndInternodes"):
            fs = scan.get_fileset(sfs.id)
    if fs is None:
        logger.critical(f"Could not find `AnglesAndInternodes` dataset for scan id `{args.scan}`!")
        exit(1)

    if len(args.free_ends) == 1:
        args.free_ends = args.free_ends[0]

    logger.info(f"Performing sequence comparison to ground-truth for '{args.scan}'...")
    # Load the JSON file produced by task `AnglesAndInternodes`:
    json_dict = read_json(fs.files[0])
    # Get the predicted angles and inter-nodes sequences:
    pred_angles = json_dict['angles']
    pred_internodes = json_dict['internodes']
    # Get the ground-truth angles and inter-nodes sequences:
    gt_angles = scan.get_measures('angles')
    gt_internodes = scan.get_measures('internodes')
    # Create ground-truth & predicted angles and inter-nodes arrays
    seq_gt = np.array([pred_angles, pred_internodes]).T
    seq_pred = np.array([gt_angles, gt_internodes]).T

    mixed_kwargs = {}
    if args.dist_type == 'mixed':
        # Get the max value for inter-nodes, used by `mixed_spread`
        max_gt = np.max(seq_gt[:, 1])
        max_pred = np.max(seq_pred[:, 1])
        # Update the keyword arguments to use with this type of distance
        mixed_kwargs = {'mixed_type': [True, False],
                        'mixed_weight': [0.5, 0.5],
                        'mixed_spread': [1, max(max_gt, max_pred)]}

    flag_kwargs = {'cum_dist_flag': False, 'bp_flag': False, 'ld_flag': False,
                   'free_ends_flag': False, 'optimal_path_flag': True,
                   'graphic_optimal_path_flag': False, 'graphic_seq_alignment': False}

    dtwcomputer = sequence_comparison(seq_pred, seq_gt, constraint=args.constraint, dist_type=args.dist_type,
                             free_ends=args.free_ends, delins_cost=args.delins_cost, max_stretch=args.max_stretch,
                             beam_size=args.beamsize, verbose=True, **mixed_kwargs)
    df = dtwcomputer.print_results(**flag_kwargs)

    logger.name = logger_name
    out_csv = join(args.db_path, args.scan, 'dtw_result.csv')
    logger.info(f"Exporting result CSV to '{out_csv}'")
    df.to_csv(out_csv, index=False)


if __name__ == '__main__':
    parser = parsing()
    args = parser.parse_args()
    main(args)
