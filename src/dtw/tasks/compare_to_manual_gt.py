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
from dtw.dtw import CONSTRAINTS
from dtw.dtw import DEF_BEAMSIZE
from dtw.dtw import DEF_CONSTRAINT
from dtw.dtw import DEF_DELINS_COST
from dtw.dtw import DEF_DIST_TYPE
from dtw.dtw import DEF_FREE_ENDS
from dtw.dtw import DEF_MAX_STRETCH
from dtw.dtw import DIST_TYPES
from dtw.dtw import sequence_comparison
from dtw.tasks.logger import DEFAULT_LOG_LEVEL
from dtw.tasks.logger import get_logger
from plantdb.fsdb import FSDB
from plantdb.io import read_json

DESCRIPTION = """Compare the sequence obtained with the ROMI reconstruction pipeline to a manual measurement using the DTW algorithm.

The manual measures are assumed to be the ground-truth.

Parameters
----------
constraint: str
    Type of constraint to use.
dist_type: str
    Type of distance to use.
free_ends: float or tuple of int
    A float corresponds to a percentage of sequence length for max exploration of `free_ends`,
    and in that case ``free_ends <= 0.4``.
    A tuple of 2 integers ``(k,l)`` that specifies relaxation bounds on
    the alignment of sequences endpoints: relaxed by ``k`` at the sequence beginning
    and relaxed by ``l`` at the sequence ending.
beamsize : int
    maximum amount of distortion allowed for signal warping.
delins_cost : tuple of float
    Deletion and insertion costs.
max_stretch : bool
    ???.

convert_dtw_results

"""


def parsing():
    parser = argparse.ArgumentParser(description=DESCRIPTION)

    parser.add_argument('db_path', type=str,
                        help="path to a ROMI local database managed by `FSDB`.")
    parser.add_argument('scan', type=str,
                        help="name of the plant/scan to analyse.")

    dtw_opt = parser.add_argument_group('DTW algorithm arguments')
    dtw_opt.add_argument('-c', '--constraint', type=str, default=DEF_CONSTRAINT,
                         choices=CONSTRAINTS,
                         help=f"type of constraint to use, '{DEF_CONSTRAINT}' by default.")
    dtw_opt.add_argument('-d', '--dist_type', type=str, default=DEF_DIST_TYPE,
                         choices=DIST_TYPES,
                         help=f"type of distance to use, '{DEF_DIST_TYPE}' by default.")
    dtw_opt.add_argument('-f', '--free_ends', type=float, nargs='+', default=DEF_FREE_ENDS,
                         help=f"Free ends values to use, '{DEF_FREE_ENDS}' by default.")
    dtw_opt.add_argument('-b', '--beamsize', type=int, default=DEF_BEAMSIZE,
                         help=f"Beam size values to use, '{DEF_BEAMSIZE}' by default.")
    dtw_opt.add_argument('-i', '--delins_cost', type=int, nargs=2, default=DEF_DELINS_COST,
                         help=f"Beam size values to use, '{DEF_DELINS_COST}' by default.")
    dtw_opt.add_argument('-s', '--max_stretch', type=int, default=DEF_MAX_STRETCH,
                         help=f"Beam size values to use, '{DEF_MAX_STRETCH}' by default.")

    log_opt = parser.add_argument_group('logging arguments')
    log_opt.add_argument('--log_level', type=str, default=DEFAULT_LOG_LEVEL.lower(),
                         choices=['info', 'warning', 'error', 'critical', 'debug'],
                         help=f"logging level to use, '{DEFAULT_LOG_LEVEL.lower()}' by default.")

    return parser


def main(args):
    # Get logging level from input & convert it to corresponding numeric level value
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}")
    logger = get_logger('compare_to_manual_gt', join(args.db_path, args.scan, "compare_to_manual_gt.log"), numeric_level)

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

    df = sequence_comparison(seq_pred, seq_gt, constraint=args.constraint, dist_type=args.dist_type, free_ends=args.free_ends, beam_size=args.beamsize,
                             delins_cost=args.delins_cost, max_stretch=args.max_stretch, verbose=True, **mixed_kwargs, **flag_kwargs)
    df.to_csv(join(args.db_path, args.scan, 'dtw_result.csv'))


if __name__ == '__main__':
    parser = parsing()
    args = parser.parse_args()
    main(args)
