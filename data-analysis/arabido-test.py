"""
Author: C. Godin Inria, RDP, April 2020

Data: F. Besnard, Inra, RDP, April 2020

"""

import pandas as pd
from dtw import dtw

Databasis_dir = 'DB_eval_v1/'
# Construction of the main dataframe from a csv file
groundtruth_df = pd.read_csv(Databasis_dir + 'groundtruth.csv')
# Construction of the main dataframe from a csv file
#predicted_df = pd.read_csv('DB_eval_v1/predicted_v0.4.csv')
predicted_df = pd.read_csv(Databasis_dir + 'predicted_v20200727.csv')

testname = 'v20200727'

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
# Example of call for the comparison of two sequence with all the flags set:

dtw.runCompare(vecseq_test,vecseq_ref,
               constraint_type = 'MERGE_SPLIT', dist_type = 'MIXED',
               mixed_type = [True,False], mixed_spread = [1,max(max1,max2)], mixed_weight = [0.5,0.5],
               freeends = (2,2), beamsize = -1, delinscost = (1.,1.),
               cumdistflag=False, bpflag=False, ldflag=False, freeendsflag=False,
               optimalpathflag=True, graphicoptimalpathflag=True)

"""

#plantname = 'Col0_12_10_2018_A'    # --> Ok interpreted as a merge last angles
#plantname = 'Col0_12_10_2018_C'    # sur seg au debut
#plantname = 'Col0_26_10_2018_B'    # facile
#plantname = 'Col0_26_10_2018_C'    # facile

# Analyze a list of plants given explicitly
#plantnames = ['Col0_26_10_2018_B'] #,'Col0_12_10_2018_C' ]
plantnames = ['Col0_12_10_2018_C']

# Analyze the whole set of plants in the given databases
# plantnames = groundtruth_df.PlantID.unique()

VERBOSE = True

df = pd.DataFrame()

for plantname in plantnames:

    gtruth = angles_array(groundtruth_df, 'PlantID', plantname)
    predicted = angles_array(predicted_df, 'PlantID', plantname)

    # extract angle arrays from plant grround truth and predicted dataframes
    seq_ref  = gtruth['angles'].values
    seq_test = predicted['angles'].values

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

    if VERBOSE:
        filestg = '############## PROCESSING FILE : ' + plantname + ' ##############'
        dashline = '#' * len(filestg)
        print()
        print(dashline)
        print(filestg)

    df_result = dtw.runCompare(vecseq_test,vecseq_ref,
                   constraint_type = 'MERGE_SPLIT', dist_type = 'MIXED',
                   mixed_type = [True,False], mixed_spread = [1,max(max1,max2)], mixed_weight = [0.5,0.5],
#                   freeends = (2,1),
                   freeends = 0.3,
                   freeendseps = 1e-3,
                   beamsize = -1, delinscost = (1.,1.),
                   cumdistflag=False, bpflag=False, ldflag=False, freeendsflag=False,
                   optimalpathflag=True, graphicoptimalpathflag=False,
                   graphicseqalignment = False,
                   Verbose = VERBOSE)

    #Add a column containing name

    # df_result['PlantID'] = [plantname] * len(df_result.index)
    df_result.insert(loc = 0, column = 'PlantID', value = [plantname] * len(df_result.index) )

    if df.empty:
        df = df_result
    else:
        df = pd.concat([df, df_result])

#print(df_result)
df.to_csv(Databasis_dir + testname + '_result.csv')
