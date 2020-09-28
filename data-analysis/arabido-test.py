"""
Author: C. Godin Inria, RDP, April 2020

Data: F. Besnard, Inra, RDP, April 2020

"""

import pandas as pd
from dtw import dtw

# Construction of the main dataframe from a csv file
groundtruth_df = pd.read_csv('DB_eval_v1/groundtruth.csv')
# Construction of the main dataframe from a csv file
#predicted_df = pd.read_csv('DB_eval_v1/predicted_v0.4.csv')
predicted_df = pd.read_csv('DB_eval_v1/predicted_v20200727.csv')

# Labels of the found olumns
# col_labs = testdf.columns

#############################
# Extraction of angles only
#############################


def angles_array(df, plantname_col, plantname):
  """
  Function that returns a sub dataframe corresponding to the selected plant name
  Parameters
  ----------
  df: dataframe
  plantname_col: name of the column containing the plant names
  plantname: name of the plant considered

  Returns: a subdataframe corresponding to the selected plantname
  -------
  """

  return df[df[plantname_col] == plantname]



"""
# Example of call for the comparisson of two sequence with all the flags set:

dtw.runCompare(vecseq_test,vecseq_ref,
               constraint_type = 'MERGE_SPLIT', dist_type = 'MIXED',
               mixed_type = [True,False], mixed_spread = [1,max(max1,max2)], mixed_weight = [0.5,0.5],
               freeends = (2,2), beamsize = -1, delinscost = (1.,1.),
               cumdistflag=False, bpflag=False, ldflag=False, freeendsflag=False,
               optimalpathflag=True, graphicoptimalpathflag=True)

"""

#plantname = 'Col0_12_10_2018_A'    # --> Ok interpreted as a merge last angles
#plantname = 'Col0_12_10_2018_C'    # sur seg au debut
plantname = 'Col0_26_10_2018_B'    # facile
#plantname = 'Col0_26_10_2018_C'    # facile

gtruth = angles_array(groundtruth_df, 'Plant ID', plantname)
predicted = angles_array(predicted_df, 'Plant ID', plantname)

# extract angle arrays from plant grround truth and predicted dataframes
seq_ref  = gtruth['angles'].values
seq_test = predicted['angles'].values

# Run comparison of simple 1-D sequences
#dtw.runCompare(seq_test,seq_ref,dist_type = 'ANGULAR',freeends=(2,2),bpflag = False, ldflag=False)


#####################################
# Extraction of angles and internodes
#####################################

# selection angle and internode columns
vecref_df = gtruth[['angles','Internodes']]
vectest_df = predicted[['angles','Internodes']]
vecseq_ref = vecref_df.values
vecseq_test= vectest_df.values

max1 = max(vecseq_ref[:,1])
max2 = max(vecseq_test[:,1])


dtw.runCompare(vecseq_test,vecseq_ref,
               constraint_type = 'MERGE_SPLIT', dist_type = 'MIXED',
               mixed_type = [True,False], mixed_spread = [1,max(max1,max2)], mixed_weight = [0.5,0.5],
               freeends = (2,2), beamsize = -1, delinscost = (1.,1.),
               cumdistflag=False, bpflag=False, ldflag=False, freeendsflag=False,
               optimalpathflag=True, graphicoptimalpathflag=True)
