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
from math import degrees
from pathlib import Path

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

DEFAULT_FIG_FMT = 'png'


def parsing():
    parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('db_path', type=str,
                        help="path to a ROMI local database managed by `FSDB`.")
    parser.add_argument('scan', type=str,
                        help="name of the plant or scan to analyse.")

    dtw_opt = parser.add_argument_group('DTW algorithm arguments')
    dtw_opt.add_argument('--constraint', type=str, default=DEF_CONSTRAINT, choices=CONSTRAINTS,
                         help=f"type of constraint to use, '{DEF_CONSTRAINT}' by default.")
    dtw_opt.add_argument('--dist_type', type=str, default=DEF_DIST_TYPE, choices=DIST_TYPES,
                         help=f"type of distance to use, '{DEF_DIST_TYPE}' by default.")
    dtw_opt.add_argument('--free_ends', type=float, nargs='+', default=DEF_FREE_ENDS,
                         help=f"free ends values to use, specify the relaxation bounds, '{DEF_FREE_ENDS}' by default.")
    dtw_opt.add_argument('--beam_size', type=int, default=DEF_BEAMSIZE,
                         help=f"maximum amount of distortion allowed for signal warping, '{DEF_BEAMSIZE}' by default.")
    dtw_opt.add_argument('--delins_cost', type=float, nargs=2, default=DEF_DELINS_COST,
                         help=f"cost of deletion and insertion to use, '{DEF_DELINS_COST}' by default.")
    dtw_opt.add_argument('--max_stretch', type=int, default=DEF_MAX_STRETCH,
                         help=f"maximum amount of stretching allowed for signal warping, '{DEF_MAX_STRETCH}' by default.")

    fig_opt = parser.add_argument_group('figure arguments')
    fig_opt.add_argument('--to_degrees', action="store_true",
                         help="use it to convert angles in degrees.")
    fig_opt.add_argument('--figure_format', type=str, default=DEFAULT_FIG_FMT, choices=['png', 'jpeg', "svg", 'eps'],
                         help=f"set the file format of the alignment figure, '{DEFAULT_FIG_FMT}' by default.")

    log_opt = parser.add_argument_group('logging arguments')
    log_opt.add_argument('--log_level', type=str, default=DEFAULT_LOG_LEVEL.lower(),
                         choices=['info', 'warning', 'error', 'critical', 'debug'],
                         help=f"logging level to use, '{DEFAULT_LOG_LEVEL.lower()}' by default.")

    return parser


def main(args):
    logger_name = "romi_compare_to_manual_gt"
    args.db_path = Path(args.db_path)
    logger = get_logger(logger_name, args.db_path / args.scan / "compare_to_manual_gt.log",
                        args.log_level.upper(), BIN_LOG_FMT)

    lock_file = args.db_path / 'lock'
    lock_file.unlink(missing_ok=True)

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
    # Get the predicted angles and internodes sequences:
    pred_internodes = json_dict['internodes']
    pred_angles = json_dict['angles']
    if args.to_degrees:
        pred_angles = [degrees(angle) for angle in pred_angles]

    # Get the ground-truth angles and internodes sequences:
    gt_internodes = scan.get_measures('internodes')
    gt_angles = scan.get_measures('angles')
    if args.to_degrees:
        gt_angles = [degrees(angle) for angle in gt_angles]

    # Create ground-truth & predicted angles and internodes arrays
    seq_pred = np.array([pred_angles, pred_internodes]).T
    seq_gt = np.array([gt_angles, gt_internodes]).T

    mixed_kwargs = {}
    if args.dist_type == 'mixed':
        # Get the max value for internodes, used by `mixed_spread`
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
                                      free_ends=args.free_ends, delins_cost=args.delins_cost,
                                      max_stretch=args.max_stretch, beam_size=args.beam_size, verbose=True,
                                      names=['angles', 'internodes'], **mixed_kwargs)
    df = dtwcomputer.print_results(**flag_kwargs)

    logger.name = logger_name
    out_csv = args.db_path / args.scan / 'dtw_result.csv'
    logger.info(f"Exporting result CSV to '{out_csv}'")
    df.to_csv(out_csv, index=False)
    dtwcomputer.plot_results(figname=args.db_path / args.scan / f"SM-DTW_alignment.{args.figure_format}")


if __name__ == '__main__':
    parser = parsing()
    args = parser.parse_args()
    main(args)
