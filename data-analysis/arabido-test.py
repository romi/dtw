#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       File author(s):
#           Christophe Godin <christophe.godin@inria.fr>
#
#       File contributor(s):
#           Fabrice Besnard <fabrice.besnard@ens-lyon.fr>
#           Jonathan Legrand <jonathan.legrand@ens-lyon.fr>
#
#       File maintainer(s):
#           Christophe Godin <christophe.godin@inria.fr>
#
#       Mosaic Inria team, RDP Lab, Lyon
# ------------------------------------------------------------------------------

"""
Data provided by F. Besnard, INRAe, RDP, ENS de Lyon - UMR 5667, April 2020.
"""

import pandas as pd
from dtw.dtw import sequence_comparison

from os.path import abspath
from os.path import dirname
from os.path import join


def plant_dataframe(df, plant_name):
    """Returns a sub-dataframe corresponding to the selected plant name.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe in which to extract the data associated to the `plant_name`.
    plant_name : str
        Name of the plant considered.

    Notes
    -----
    This requires the CSV to have a column "PlantID" where the plant names are defined.

    Returns
    -------
    pandas.DataFrame
        A sub-dataframe corresponding to the selected `plant_id`.
    """
    return df[df['PlantID'] == plant_name]


"""
# Example of call for the comparison of two sequence with all the flags set:

dtw.sequence_comparison(seq_test, seq_ref,
                        constraint_type='merge_split', dist_type='mixed',
                        mixed_type=[True,False], mixed_spread=[1, max(max1,max2)], mixed_weight=[0.5,0.5],
                        free_ends=(2,2), beam_size=-1, delins_cost=(1.,1.),
                        cumdistflag=False, bpflag=False, ldflag=False, freeendsflag=False,
                        optimalpathflag=True, graphicoptimalpathflag=True)

"""


def main(db_directory, testname='v0.4'):
    # Construction of the main dataframe from a csv file
    groundtruth_df = pd.read_csv(db_directory + '/' + 'groundtruth.csv')
    # Construction of the main dataframe from a csv file
    # predicted_df = pd.read_csv('DB_eval_v1/predicted_v0.4.csv')
    predicted_df = pd.read_csv(db_directory + '/' + 'predicted_v0.4.csv')

    # Labels of the found columns
    # col_labs = testdf.columns

    # plant_id = 'Col0_12_10_2018_A'    # --> Ok interpreted as a merge last angles
    # plant_id = 'Col0_12_10_2018_C'    # sur seg au debut
    # plant_id = 'Col0_26_10_2018_B'    # facile
    # plant_id = 'Col0_26_10_2018_C'    # facile

    # Analyze
    # 1. a list of plants given explicitly
    # plant_ids = ['Col0_26_10_2018_B'] #,'Col0_12_10_2018_C' ]
    # plant_ids = ['Col0_12_10_2018_C']

    # 2. a whole set of plants:
    # Analyze the whole set of plants in the given databases
    plant_ids = groundtruth_df.PlantID.unique()

    VERBOSE = True

    df = pd.DataFrame()

    for plant_id in plant_ids:
        # Get ground-truth and predicted DataFrames for current `plant_id`
        gtruth = plant_dataframe(groundtruth_df, plant_id)
        predicted = plant_dataframe(predicted_df, plant_id)
        # Get 'angle' and 'Internode' columns from DataFrames as numpy array
        vecseq_gt = gtruth[['angles', 'Internodes']].values
        vecseq_pred = predicted[['angles', 'Internodes']].values
        max_gt = max(vecseq_gt[:, 1])
        max_pred = max(vecseq_pred[:, 1])

        if VERBOSE:
            filestg = f"PROCESSING FILE : {plant_id}"
            dashline = '#' * 80
            print()
            print(dashline)
            print(f"{filestg:*^80}")

        df_result = sequence_comparison(vecseq_pred, vecseq_gt, constraint='merge_split', dist_type='mixed', free_ends=0.3, free_ends_eps=1e-3, beamsize=-1,
                                        delins_cost=(1., 1.), mixed_type=[True, False], mixed_spread=[1, max(max_gt, max_pred)], mixed_weight=[0.5, 0.5],
                                        cumdistflag=False, bpflag=False, ldflag=False, freeendsflag=False, optimalpathflag=True, graphicoptimalpathflag=False,
                                        graphicseqalignment=False, verbose=VERBOSE)

        # Add a column containing name
        # df_result['PlantID'] = [plant_id] * len(df_result.index)
        df_result.insert(loc=0, column='PlantID', value=[plant_id] * len(df_result.index))

        if df.empty:
            df = df_result
        else:
            df = pd.concat([df, df_result])

    # print(df_result)
    df.to_csv(db_directory + testname + '_result.csv')


if __name__ == '__main__':
    base_dir = abspath(dirname(__file__))
    db_directory = join(base_dir, 'DB_eval_v1')
    main(db_directory)
