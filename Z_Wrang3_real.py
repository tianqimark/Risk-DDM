#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri September 2021

@author: hutianqi
"""

# =============================================================================
# Import data from the server
# =============================================================================

import numpy as np
import pandas as pd

### Read through concatenation
batch1 = pd.read_csv('/Users/hutianqi/Desktop/Project Risk and Intelligence/01 Dataset/00 RawData/Batch1_trials.csv')
batch2 = pd.read_csv('/Users/hutianqi/Desktop/Project Risk and Intelligence/01 Dataset/00 RawData/Batch2_trials.csv')
batch3 = pd.read_csv('/Users/hutianqi/Desktop/Project Risk and Intelligence/01 Dataset/00 RawData/Batch3_trials.csv')
batch4 = pd.read_csv('/Users/hutianqi/Desktop/Project Risk and Intelligence/01 Dataset/00 RawData/Batch4_trials.csv')
batch5 = pd.read_csv('/Users/hutianqi/Desktop/Project Risk and Intelligence/01 Dataset/00 RawData/Batch5_trials.csv')

frames = [batch1, batch2, batch3, batch4, batch5]
datar = pd.concat(frames, ignore_index=True)

### Read an individual dataset
# datar = pd.read_csv('/Users/hutianqi/Desktop/Project Risk and Intelligence/01 Dataset/00 RawData/Batch5_trials.csv')

###
datar.columns[-5:]
datar = datar[datar['even_gamble.200.player.pay_pound'].notna()]

# =============================================================================
# calculate payment and creat a table for making reward payment
# =============================================================================

pay_columns = datar[['ECI_survey.1.player.prolific_code', 'even_gamble.200.player.pay_pound']]

# pay_columns.to_csv("/Users/hutianqi/Desktop/Project Risk and Intelligence/01 Dataset/00 RawData/ZZ Reward Payment/Batch5_reward.csv", index=False)

pay_reward = datar['even_gamble.200.player.pay_pound'].sum()
pay_reward_real = datar['even_gamble.200.player.pay_pound'].sum() * 1.4

pay_total = pay_reward_real + 3 * 1.4 * datar.shape[0]

#### check if certain IDs are selected to count
# datar['ECI_survey.1.player.prolific_code'].isin(['614b22e6970210e4d70b4178'])
# datar['ECI_survey.1.player.prolific_code'].isin(['614caafc501c550ac719df6f'])

# where = np.where(datar['ECI_survey.1.player.prolific_code'] == '614b22e6970210e4d70b4178')

# =============================================================================
# Adding demographic information to the datar
# =============================================================================

### Read through concatenation
demo1 = pd.read_csv('/Users/hutianqi/Desktop/Project Risk and Intelligence/01 Dataset/00 RawData/Batch1_demo.csv')
demo2 = pd.read_csv('/Users/hutianqi/Desktop/Project Risk and Intelligence/01 Dataset/00 RawData/Batch2_demo.csv')
demo3 = pd.read_csv('/Users/hutianqi/Desktop/Project Risk and Intelligence/01 Dataset/00 RawData/Batch3_demo.csv')
demo4 = pd.read_csv('/Users/hutianqi/Desktop/Project Risk and Intelligence/01 Dataset/00 RawData/Batch4_demo.csv')
demo5 = pd.read_csv('/Users/hutianqi/Desktop/Project Risk and Intelligence/01 Dataset/00 RawData/Batch5_demo.csv')

frames = [demo1, demo2, demo3, demo4, demo5]
demo = pd.concat(frames, ignore_index=True)

### Read an individual dataset
# demo = pd.read_csv('/Users/hutianqi/Desktop/Project Risk and Intelligence/01 Dataset/00 RawData/Batch5_demo.csv')

###
demo.index = demo['participant_id']
datar.index = datar['ECI_survey.1.player.prolific_code']

datar['gender'] = demo['Sex']
datar['age'] = demo['age']

# =============================================================================
# preprosing data to create workable tables
# =============================================================================

import numpy as np
import pandas as pd

# # Now we can directly read the sorted raw dataset
# datar = pd.read_csv('/Users/hutianqi/Desktop/Project Risk and Intelligence/Z1 Preprocessing/Self test local.csv')

# Assign each subject a numerical code
datar.reset_index(drop=True, inplace = True)
datar['subject'] = datar.index + 1

# # export a reference table for the numerial code to the participant.code assigned by the system
# subject_ref = datar[['participant.code', 'subject']].copy()
# subject_ref.to_csv('/Users/hutianqi/Desktop/Project Risk and Intelligence/Z1 Preprocessing/test_sub_code.csv', index=False)


""" create values of gambles (change here for the real exp data) """
level = 10 # No. of levels of gambles. 5 in the pilot and 10 in the real exp.

trials_number = level * level * 2 # 50 in the pilot, 200 in the real

# build a trial table
column_names = ['subject', 'trial',
                'display', 'gain', 'loss', 'stake', 'gain_adv',
                'RT', 'accept']

# index length = trial number * number of subjects
index_length = trials_number * datar.shape[0]

trials = pd.DataFrame(index = range(index_length), columns = column_names)

# assign subject and trial numbers
for i in range(trials.shape[0]):
    counter1 = i // trials_number
    trials['subject'][i] = counter1 + 1

    counter2 = i % trials_number
    trials['trial'][i] = counter2 + 1


# extract info from the raw dataset
datar.index = datar['subject']

for i in range(trials.shape[0]):
    subject = trials['subject'][i]
    trial = trials['trial'][i]

    gain = 'even_gamble.' + str(trial) + '.player.gain'
    trials['gain'][i] = datar.loc[subject, gain]

    loss = 'even_gamble.' + str(trial) + '.player.loss'
    trials['loss'][i] = datar.loc[subject, loss]

    display = 'even_gamble.' + str(trial) + '.player.display'
    trials['display'][i] = datar.loc[subject, display]

    RT = 'even_gamble.' + str(trial) + '.player.jsdectime'
    trials['RT'][i] = datar.loc[subject, RT]

    accept = 'even_gamble.' + str(trial) + '.player.accept'
    trials['accept'][i] = datar.loc[subject, accept]


# calculate the stake of the gamble and gain advantage in every trial
trials['stake'] = (trials['gain'] + trials['loss']) / 2
trials['gain_adv'] = trials['gain'] - trials['loss']

# =============================================================================
# Sort out the order of trials
# =============================================================================

values = list(range(1, level + 1))

for i in range(len(values)):
    values[i] *= 10

# create two lists for assigning values in loops
values_list1 = []
values_list2 = []
len_list = len(values) * level

for i in range(len_list):
    counter1 = i // level
    values_list1.append(values[counter1])

    counter2 = i % level
    values_list2.append(values[counter2])

# create a table of ordered trials
ordered = pd.DataFrame(index = range(index_length), columns = column_names)

for i in range(ordered.shape[0]):
    # subject
    counter1 = i // trials_number
    ordered['subject'][i] = counter1 + 1

    # display
    display_number = trials_number / 2
    counter2 = i // display_number
    if counter2 % 2 == 0:
        ordered['display'][i] = 1
    else:
        ordered['display'][i] = 0

    # gain and loss
    counter3 = int(i % display_number)
    ordered['gain'][i] = values_list1[counter3]
    ordered['loss'][i] = values_list2[counter3]


### find values of RT, accept and trial

for i in range(ordered.shape[0]):
    where = np.where((trials['subject'] == ordered['subject'][i]) &
                     (trials['display'] == ordered['display'][i]) &
                     (trials['gain'] == ordered['gain'][i]) &
                     (trials['loss'] == ordered['loss'][i]))

    where = int(where[0])

    ordered['RT'][i] = trials.iloc[where]['RT']
    ordered['accept'][i] = trials.iloc[where]['accept']
    ordered['trial'][i] = trials.iloc[where]['trial']
    ordered['stake'][i] = trials.iloc[where]['stake']
    ordered['gain_adv'][i] = trials.iloc[where]['gain_adv']


# check if there is any missing valus:
ordered.isnull().values.any()


# =============================================================================
# create a table of individual differences (Psychometrics)
# =============================================================================

# build a table
column_names_ids = ['subject', 'gender', 'age',
                    'RT_mean', 'accept_rate', 'valid_trials',
                    'RS', 'IS', 'PSA', 'NGA',
                    'RTP_E', 'RTP_F', 'RTP_FI', 'RTP_FG',
                    'RTP_HS', 'RTP_R', 'RTP_S', 'RTP_average']


# index length = trial number * number of subjects
index_length_ids = datar.shape[0]

ids = pd.DataFrame(index = range(index_length_ids), columns = column_names_ids)

ids['subject'] = ids.index + 1

ids.index = ids['subject']
datar.index = datar['subject']

for subject in range(1, ids.shape[0] + 1):

    ids.loc[subject, 'gender'] = 1 if datar.loc[subject, 'gender'] == 'Male' else 0
    ids.loc[subject, 'age'] = datar.loc[subject, 'age']

    ids.loc[subject, 'PSA'] = datar.loc[subject, 'ECI_survey.1.player.PA_score']
    ids.loc[subject, 'NGA'] = datar.loc[subject, 'ECI_survey.1.player.NA_score']
    ids.loc[subject, 'RS'] = datar.loc[subject, 'ECI_survey.1.player.reflectiveness_score']
    ids.loc[subject, 'IS'] = datar.loc[subject, 'ECI_survey.1.player.intuitiveness_score']

    ids.loc[subject, 'RTP_E'] = datar.loc[subject, 'ECI_survey.1.player.RTP_E']
    ids.loc[subject, 'RTP_F'] = datar.loc[subject, 'ECI_survey.1.player.RTP_F']
    ids.loc[subject, 'RTP_FI'] = datar.loc[subject, 'ECI_survey.1.player.RTP_FI']
    ids.loc[subject, 'RTP_FG'] = datar.loc[subject, 'ECI_survey.1.player.RTP_FG']
    ids.loc[subject, 'RTP_HS'] = datar.loc[subject, 'ECI_survey.1.player.RTP_HS']
    ids.loc[subject, 'RTP_R'] = datar.loc[subject, 'ECI_survey.1.player.RTP_R']
    ids.loc[subject, 'RTP_S'] = datar.loc[subject, 'ECI_survey.1.player.RTP_S']
    ids.loc[subject, 'RTP_average'] = datar.loc[subject, 'ECI_survey.1.player.RTP_average']


ids.reset_index(drop = True, inplace = True)
datar.reset_index(drop = True, inplace = True)

# =============================================================================
# filling in ids with the info from the trials
# =============================================================================

trials_copy = trials.copy()

""" excluding trials based on the RT conditions: """
disrupt = 10
floor = 0.3
# floor = 0

trials_copy = trials_copy[(trials_copy['RT'] > floor) & (trials_copy['RT'] < disrupt)]

### extract information

ids.index = ids['subject']

for subject in range(1, ids.shape[0] + 1):
    subject_data = trials_copy[trials_copy['subject'] == subject]

    ids.loc[subject, 'RT_mean'] = subject_data['RT'].mean()
    ids.loc[subject, 'accept_rate'] = subject_data['accept'].sum() / subject_data.shape[0]
    ids.loc[subject, 'valid_trials'] = subject_data.shape[0]


# =============================================================================
# Adding psychometrics data to the ordered trials df
# =============================================================================

ids.index = ids['subject']
ordered.index = ordered['subject']

ordered['gender'] = ids['gender']
ordered['age'] = ids['age']

ordered['valid_trials'] = ids['valid_trials']

ordered['RS'] = ids['RS']
ordered['IS'] = ids['IS']
ordered['PSA'] = ids['PSA']
ordered['NGA'] = ids['NGA']

ordered['RTP_E'] = ids['RTP_E']
ordered['RTP_F'] = ids['RTP_F']
ordered['RTP_FI'] = ids['RTP_FI']
ordered['RTP_FG'] = ids['RTP_FG']
ordered['RTP_HS'] = ids['RTP_HS']
ordered['RTP_R'] = ids['RTP_R']
ordered['RTP_S'] = ids['RTP_S']
ordered['RTP_average'] = ids['RTP_average']

# reset index is not necessary but a good habit...
ids.reset_index(drop = True, inplace = True)
ordered.reset_index(drop = True, inplace = True)

# =============================================================================
# Export data
# =============================================================================

# ordered.to_csv("/Users/hutianqi/Desktop/Project Risk and Intelligence/01 Dataset/Trials.csv", index=False)
# ids.to_csv("/Users/hutianqi/Desktop/Project Risk and Intelligence/01 Dataset/IDT.csv", index=False)





# # =============================================================================
# # Change column names
# # =============================================================================
# import numpy as np
# import pandas as pd

# data_address = '/Users/hutianqi/Desktop/Project Risk and Intelligence/01 Dataset/Trials.csv'
# ids_address = '/Users/hutianqi/Desktop/Project Risk and Intelligence/01 Dataset/IDT.csv'

# data = pd.read_csv(data_address)
# ids = pd.read_csv(ids_address)

# data.columns
# ids.columns

# data.rename(columns={"PA": "PSA", "NA": "NGA"}, inplace=True)
# ids.rename(columns={"PA": "PSA", "NA": "NGA"}, inplace=True)

# # Export datasets with updated info
# data.to_csv(data_address, index=False)
# ids.to_csv(ids_address, index=False)
