#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 14:45:31 2022

@author: hutianqi
"""

import time
start_time = time.time()

import numpy as np
import pandas as pd
# np.random.seed(123)

import pyddm as ddm
from pyddm import Model, Fittable, Sample
from pyddm.models import NoiseConstant, BoundConstant, ICPoint, OverlayChain, OverlayNonDecision, OverlayUniformMixture
from pyddm.functions import fit_adjust_model, display_model

# set up the parallel pool
from pyddm import set_N_cpus
set_N_cpus(4)

# =============================================================================
# Load data and preprocessing
# =============================================================================

# load the full dataset with 72 subjects
data = pd.read_csv('/Users/hutianqi/Desktop/Project Cognitive Imprecision/Z9 oTree/ZZ Pilot Testing 2206/Z1 Data/Trials_pilot.csv')
# data = pd.read_csv('/home/bs/bsth4/DDM/Trials_22S1.csv')

# Two additional steps to ensure the data quality, following previous literature.
# (1) exclude trials in which DTs are above 10 seconds (This is the time limitation in Khaw's study).
# (2) from smith 2018: trials with very short (􏰃300 ms) RTs were removed from analysis

disrupt = 10
floor = 0.3
data = data[(data['RT'] > floor) & (data['RT'] < disrupt)]

# drop rows where a correct response is non-existant due to WTP = certainty
data.dropna(inplace = True)

# reset index
data.reset_index(drop = True, inplace = True)

# Only take useful info
# data = data[['subject', 'value_diff', 'RT', 'correct']]
data = data[['subject', 'value_diff', 'RT', 'correct', 'lot_correct']]
# data = data[['subject', 'decmode', 'WTP', 'certainty', 'RT', 'correct', 'lot_correct']]

# =============================================================================
# DDM fitting
# =============================================================================

""" The Simple_Drift and Biased_Start features need to be commented off in case they are imported from a seperated file"""

# Formulate the drift rate
class Simple_Drift(ddm.models.Drift):
    name = "Drift depends on the value defference"
    required_parameters = ["d"]
    required_conditions = ["value_diff"]

    # We must always define the get_drift function, which is used to compute the instantaneous value of drift.
    def get_drift(self, conditions, **kwargs):
        return self.d * conditions['value_diff']


# the initial condition as a proportion of the total distance between the bounds
class Biased_Start(ICPoint):
    name = "A biased starting point."
    required_parameters = ["x0"]
    required_conditions = ["lot_correct"]

    def get_IC(self, x, dx, conditions):
        x0 = self.x0/2 + .5 #rescale to between 0 and 1
        # Bias > .5 for high reward (lottery), bias < .5 for low reward (monetary certainty).
        # On original scale, positive bias for high reward conditions, negative for low reward
        if not conditions['lot_correct']:
            x0 = 1-x0
        shift_i = int((len(x)-1)*x0)
        assert shift_i >= 0 and shift_i < len(x), "Invalid initial conditions"
        pdf = np.zeros(len(x))
        pdf[shift_i] = 1. # Initial condition at x=x0*2*B.
        return pdf



model_fDDM = Model(name='fDDMn001BU_10s',
                    drift=Simple_Drift(d = Fittable(minval = 0, maxval = 0.4)),
                    IC=Biased_Start(x0 = Fittable(minval = -0.7, maxval = 0.7)),
                    noise = NoiseConstant(noise=1),
                    bound = BoundConstant(B=Fittable(minval=0.3, maxval=2.5)),
                    overlay=OverlayChain(overlays = 
                                        [OverlayNonDecision(nondectime = Fittable(minval=0, maxval=2.2)), 
                                          OverlayUniformMixture(umixturecoef=.05)]),
                    dx=0.001, dt=0.001, T_dur=10)
                    # dx=0.01, dt=0.01, T_dur=10)


# model_sDDM = Model(name='sDDM for baseline modelling',
#                    drift=Simple_Drift(d = Fittable(minval = 0, maxval = 0.35)),
#                    noise = NoiseConstant(noise=1),
#                    bound = BoundConstant(B=Fittable(minval=0.2, maxval=2.5)),
#                    overlay=OverlayChain(overlays = 
#                                         [OverlayNonDecision(nondectime = Fittable(minval=0, maxval=2.5)), 
#                                          OverlayUniformMixture(umixturecoef=.05)]),
#                     # dx=0.001, dt=0.001, T_dur=10)
#                     dx=0.01, dt=0.01, T_dur=10)


# Check if an analytical solution exists.
model_fDDM.has_analytical_solution()
model_fDDM.get_model_parameter_names()
model_fDDM.get_model_parameters()

# model_sDDM.has_analytical_solution()
# model_sDDM.get_model_parameter_names()
# model_sDDM.get_model_parameters()


# =============================================================================
# Create a df to record parameter values -- sDDM
# =============================================================================

column_names = ['subject',
                'd', 'B', 'x0', 'nondectime',
                'loss_func_value', 'criterion', 'sample_size']

# sDDM_fitting = pd.DataFrame(columns = column_names)
fDDM_fitting = pd.DataFrame(columns = column_names)


# =============================================================================
# Decide which subjects to fit
# =============================================================================

"""if fitting a group of subjects with consecutive ID"""
# start_ID = 2
# end_ID = 3
# fitting_pool = list(range(start_ID, end_ID + 1))

"""if fitting a single subject or multiple subjects with non-consecutive ID"""
fitting_pool = [3]

# =============================================================================
# Saving route
# =============================================================================

# create the file name for the df 'sDDM_fitting'
if len(fitting_pool) != 1 and (fitting_pool[-1] - fitting_pool[0]) == (len(fitting_pool) -1):
    # If fitting a consecutive group of subjects
    df_name = 'new_fDDM_2206P_' + str(fitting_pool[0]) + 'to' + str(fitting_pool[-1])

else:
    # If fitting a group of dispersed subjects
    df_name = 'new_fDDM_2206P_'
    for i in fitting_pool:
        df_name = df_name + str(i) + '_'

# df_name = df_name + '_Test_b1' + '.csv'
df_name = df_name + '_b6' + '.csv'

df_folder = '/Users/hutianqi/Desktop/Project Cognitive Imprecision/Z9 oTree/ZZ Pilot Testing 2206/ZZ Output/'
# df_folder = '/home/bs/bsth4/DDM/Z1_DDM_Output/'

df_saving_address = df_folder + df_name

# =============================================================================
# Fit the model to each subject in the fitting pool 
# =============================================================================

for subject in data['subject'].unique():
    if subject in fitting_pool:

        dummy = data[data['subject'] == subject]

        sample_size = dummy.count()['subject']

        sample_dummy = Sample.from_pandas_dataframe(dummy, rt_column_name="RT", correct_column_name="correct")

        fit_sample = fit_adjust_model(sample = sample_dummy, model = model_fDDM, fitting_method = 'differential_evolution')

        result_list = [[subject] + fit_sample.get_model_parameters() + \
                        [fit_sample.fitresult.value(), fit_sample.fitresult.loss, sample_size]]

        result_df = pd.DataFrame(result_list, columns = column_names)

        fDDM_fitting = fDDM_fitting.append(result_df, ignore_index=True)


# To save the df 'xDDM_fitting'
fDDM_fitting.to_csv(df_saving_address, index=False)


# =============================================================================
# Run Time
# =============================================================================

run_time = time.time() - start_time
print("--- %s minutes %.2s seconds ---" % (int(run_time // 60), run_time % 60))

