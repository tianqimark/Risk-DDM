#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15 Dec 2021

Fitting program for the risk DDM at group level (CRT)

Apply to own data collected in Sep 2021

@author: hutianqi
"""

import time
start_time = time.time()

import numpy as np
import pandas as pd
# np.random.seed(123)

import ddm.models
from ddm import ICPoint, Model, Fittable, Sample
from ddm.models import NoiseConstant, BoundConstant, OverlayChain, OverlayNonDecision, OverlayPoissonMixture, OverlayUniformMixture
from ddm.functions import fit_adjust_model, display_model
from ddm.models.loss import LossRobustLikelihood

# set up a parallel pool
from ddm import set_N_cpus
set_N_cpus(4)

# =============================================================================
# Load data and preprocessing
# =============================================================================

# load the full dataset and fitting reference table
data = pd.read_csv('/Users/hutianqi/Desktop/Project Risk and Intelligence/01 Dataset/Trials.csv')
fit_ref = pd.read_csv('/Users/hutianqi/Desktop/Project Risk and Intelligence/01 Dataset/Z_Fit_Ref2.csv')

# # for HPC fitting
# data = pd.read_csv('/home/bs/bsth4/DDM/Trials.csv')
# fit_ref = pd.read_csv('/home/bs/bsth4/DDM/Z_Fit_Ref2.csv')

# drop trials where RTs are either too long or too short
disrupt = 10
floor = 0.3
data = data[(data['RT'] > floor) & (data['RT'] < disrupt)]

# Only take useful info
data = data[['subject', 'gain', 'loss', 'accept', 'RT', 'domain_group', 'RS_level']]


# =============================================================================
# DDM fitting
# =============================================================================

# The Simple_Drift and Biased_Start features need to be commented off in case the yare imported from a seperated file

# the initial point as a proportion of the total distance between the bounds
class Biased_Start(ICPoint):
    name = "A biased starting point."
    required_parameters = ["x0"]
    
    """ why should I indclude **kwargs here? (without *args it also works)"""
    def get_IC(self, x, dx, *args, **kwargs): 
    # def get_IC(self, x, dx, **kwargs): 
        x0 = self.x0/2 + .5 #rescale to between 0 and 1
        # Bias > .5 gain/accept, bias < .5 for loss/reject.
        shift_i = int((len(x)-1)*x0)
        assert shift_i >= 0 and shift_i < len(x), "Invalid initial conditions"
        pdf = np.zeros(len(x))
        pdf[shift_i] = 1. # Initial condition at x=x0*2*B.
        return pdf

# Formulate the drift rate
class Risk_Drift(ddm.models.Drift):
    name = "This specification consists of vg, vl, and fixed (utility)"
    required_parameters = ["vg", "vl", "fixed"]
    required_conditions = ["gain", "loss"]

    def get_drift(self, conditions, **kwargs):
        valuation = self.vg * conditions['gain'] - self.vl * conditions['loss']
        return valuation + self.fixed



model_rDDM = Model(name='Risk_DDM',
                   drift=Risk_Drift(vg = Fittable(minval = 0.001, maxval = 0.06),
                                    vl = Fittable(minval = 0.001, maxval = 0.06),
                                    fixed = Fittable(minval = -1.6, maxval = 0.8)),
                   IC=Biased_Start(x0 = Fittable(minval = -0.55, maxval = 0.55)),
                   noise = NoiseConstant(noise=1),
                   bound = BoundConstant(B=Fittable(minval=0.5, maxval=2)),
                   
                    # Uniform Mixture Model
                   overlay=OverlayChain(overlays = 
                                        [OverlayNonDecision(nondectime = Fittable(minval=0, maxval=1.1)), 
                                         OverlayUniformMixture(umixturecoef=.05)]),
               
                   dx=0.001, dt=0.001, T_dur=10)
                   # dx=0.01, dt=0.01, T_dur=10)



# Check if an analytical solution exists.
model_rDDM.has_analytical_solution()
model_rDDM.get_model_parameter_names()
model_rDDM.get_model_parameters()

# ### check numerical stability?
# model_rDDM.can_solve_cn(conditions={'gain': data['gain'], 'loss': data['loss']})
# model_rDDM.can_solve_explicit(conditions={'gain': data['gain'], 'loss': data['loss']})
# model_rDDM.can_solve_explicit(conditions={'gain': [10, 20, 30], 'loss': [10, 20, 30]})

# =============================================================================
# functions for fitting the model to sub-groups in the datasets
# =============================================================================
""" Try coding elegantly this time... """

### fit to all subjects in a certain group based on the RS level
def Fit_DDM_Group_All_ib(domain = 'whole'):
    
    # create a df to record the output data
    column_names = ['domain', 'RS_level', 'sub_omit'] + model_rDDM.get_model_parameter_names() + \
    ['loss_func_value', 'criterion', 'sample_size', 'on_bound']
    rDDM_fitting_all = pd.DataFrame(columns = column_names)
    
    # decide whether to fit rDDM on the whole dataset, 
    # the advantageous trials or disadvantageous trials
    if domain == 'adv':
        data_for_fit = data[data['domain_group'] == 1]
    elif domain == 'dis':
        data_for_fit = data[data['domain_group'] == 2]
    else:
        data_for_fit = data
        
    # fit on groups of different RS levels
    for group in np.sort(data_for_fit['RS_level'].unique()):
        
        # find the current fitting parameters in the fit_ref
        where = np.where((fit_ref['domain'] == domain) & 
                         (fit_ref['RS_level'] == group) & 
                         (fit_ref['sub_omit'] == 'None'))
    
        where = int(where[0])
        
        # allow for extra ranges from the best fitting results
        extra = 0.05
        extra_v_dis = 0.0005
        extra_v_els = 0.005
        
        if domain == 'dis':
            vg_max = fit_ref.iloc[where]['vg_best'] + extra_v_dis
            vg_min = fit_ref.iloc[where]['vg_best'] - extra_v_dis
            vl_max = fit_ref.iloc[where]['vl_best'] + extra_v_dis
            vl_min = fit_ref.iloc[where]['vl_best'] - extra_v_dis
        else:
            vg_max = fit_ref.iloc[where]['vg_best'] + extra_v_els
            vg_min = fit_ref.iloc[where]['vg_best'] - extra_v_els
            vl_max = fit_ref.iloc[where]['vl_best'] + extra_v_els
            vl_min = fit_ref.iloc[where]['vl_best'] - extra_v_els
        
        fixed_max = fit_ref.iloc[where]['fixed_best'] + extra
        fixed_min = fit_ref.iloc[where]['fixed_best'] - extra
        B_max = fit_ref.iloc[where]['B_best'] + extra
        B_min = fit_ref.iloc[where]['B_best'] - extra
        x0_max = fit_ref.iloc[where]['x0_best'] + extra
        x0_min = fit_ref.iloc[where]['x0_best'] - extra
        nondectime_max = fit_ref.iloc[where]['nondectime_best'] + extra
        nondectime_min = fit_ref.iloc[where]['nondectime_best'] - extra

        # create a DDM modell based on the current fitting range
        model_rDDM_ib = Model(name='Risk_DDM_individual_range_fit',
            drift=Risk_Drift(vg = Fittable(minval = vg_min, maxval = vg_max),
                             vl = Fittable(minval = vl_min, maxval = vl_max),
                             fixed = Fittable(minval = fixed_min, maxval = fixed_max)),
            IC=Biased_Start(x0 = Fittable(minval = x0_min, maxval = x0_max)),
            noise = NoiseConstant(noise=1),
            bound = BoundConstant(B=Fittable(minval= B_min, maxval= B_max)),
            
            # Uniform Mixture Model
            overlay=OverlayChain(overlays = 
                                 [OverlayNonDecision(nondectime = Fittable(minval=nondectime_min, maxval=nondectime_max)), 
                                  OverlayUniformMixture(umixturecoef=.05)]),
        
            dx=0.001, dt=0.001, T_dur=10)
            # dx=0.01, dt=0.01, T_dur=10)

        # extract the subgroup for the fitting (based on the RS level)
        data_subgroup = data_for_fit[data_for_fit['RS_level'] == group]

        sample_size = data_subgroup.shape[0]
        
        # create a sample and start fitting
        data_subgroup_sample = Sample.from_pandas_dataframe(data_subgroup, rt_column_name="RT", correct_column_name="accept")
    
        fit_sample = fit_adjust_model(sample = data_subgroup_sample, model = model_rDDM_ib, fitting_method = 'differential_evolution')
    
        # sort out results
        result_list = [[domain, group, 'None'] + fit_sample.get_model_parameters() + \
                       [fit_sample.fitresult.value(), fit_sample.fitresult.loss, sample_size, 0]]
        # 0 in the end is the place holder for the counts in the on_buond column
    
        result_df = pd.DataFrame(result_list, columns = column_names)
        
        # check whether the estimated results are on the limits of the fitting ranges
        if result_df['vg'][0] == vg_min or result_df['vg'][0] == vg_max:
            result_df['on_bound'][0] += 1

        if result_df['vl'][0] == vl_min or result_df['vl'][0] == vl_max:
            result_df['on_bound'][0] += 1

        if result_df['fixed'][0] == fixed_min or result_df['fixed'][0] == fixed_max:
            result_df['on_bound'][0] += 1

        if result_df['B'][0] == B_min or result_df['B'][0] == B_max:
            result_df['on_bound'][0] += 1

        if result_df['x0'][0] == x0_min or result_df['x0'][0] == x0_max:
            result_df['on_bound'][0] += 1

        if result_df['nondectime'][0] == nondectime_min or result_df['nondectime'][0] == nondectime_max:
            result_df['on_bound'][0] += 1
    
        # append the results
        rDDM_fitting_all = rDDM_fitting_all.append(result_df, ignore_index=True)
 
           
    return rDDM_fitting_all
      

### fit to leave-one-out subsets in a certain group based on the CRT score, JK method
def Fit_DDM_Group_JK_ib(domain = 'whole'):
    
    # create a df to record the output data
    column_names = ['domain', 'RS_level', 'sub_omit'] + model_rDDM.get_model_parameter_names() + \
    ['loss_func_value', 'criterion', 'sample_size', 'on_bound']
    rDDM_fitting_jk = pd.DataFrame(columns = column_names)
    
    # decide whether to fit rDDM on the whole dataset, 
    # the advantageous trials or disadvantageous trials
    if domain == 'adv':
        data_for_fit = data[data['domain_group'] == 1]
    elif domain == 'dis':
        data_for_fit = data[data['domain_group'] == 2]
    else:
        data_for_fit = data

    # fit on groups of different RS levels
    for group in np.sort(data_for_fit['RS_level'].unique()):
        data_subgroup = data_for_fit[data_for_fit['RS_level'] == group]
        
        # omit subjects one at a time
        for sub_omit in np.sort(data_subgroup['subject'].unique()):

            # find the current fitting parameters in the fit_ref
            where = np.where((fit_ref['domain'] == domain) & 
                             (fit_ref['RS_level'] == group) & 
                             (fit_ref['sub_omit'] == str(sub_omit)))
        
            where = int(where[0])
                
            # allow for extra ranges from the best fitting results
            extra = 0.05
            extra_v_dis = 0.0005
            extra_v_els = 0.005
            
            if domain == 'dis':
                vg_max = fit_ref.iloc[where]['vg_best'] + extra_v_dis
                vg_min = fit_ref.iloc[where]['vg_best'] - extra_v_dis
                vl_max = fit_ref.iloc[where]['vl_best'] + extra_v_dis
                vl_min = fit_ref.iloc[where]['vl_best'] - extra_v_dis
            else:
                vg_max = fit_ref.iloc[where]['vg_best'] + extra_v_els
                vg_min = fit_ref.iloc[where]['vg_best'] - extra_v_els
                vl_max = fit_ref.iloc[where]['vl_best'] + extra_v_els
                vl_min = fit_ref.iloc[where]['vl_best'] - extra_v_els
            
            fixed_max = fit_ref.iloc[where]['fixed_best'] + extra
            fixed_min = fit_ref.iloc[where]['fixed_best'] - extra
            B_max = fit_ref.iloc[where]['B_best'] + extra
            B_min = fit_ref.iloc[where]['B_best'] - extra
            x0_max = fit_ref.iloc[where]['x0_best'] + extra
            x0_min = fit_ref.iloc[where]['x0_best'] - extra
            nondectime_max = fit_ref.iloc[where]['nondectime_best'] + extra
            nondectime_min = fit_ref.iloc[where]['nondectime_best'] - extra
    
            # create a DDM modell based on the current fitting range
            model_rDDM_ib = Model(name='Risk_DDM_individual_range_fit',
                drift=Risk_Drift(vg = Fittable(minval = vg_min, maxval = vg_max),
                                 vl = Fittable(minval = vl_min, maxval = vl_max),
                                 fixed = Fittable(minval = fixed_min, maxval = fixed_max)),
                IC=Biased_Start(x0 = Fittable(minval = x0_min, maxval = x0_max)),
                noise = NoiseConstant(noise=1),
                bound = BoundConstant(B=Fittable(minval= B_min, maxval= B_max)),
                
                # Uniform Mixture Model
                overlay=OverlayChain(overlays = 
                                     [OverlayNonDecision(nondectime = Fittable(minval=nondectime_min, maxval=nondectime_max)), 
                                      OverlayUniformMixture(umixturecoef=.05)]),
            
                dx=0.001, dt=0.001, T_dur=10)
                # dx=0.01, dt=0.01, T_dur=10)
    
            data_jk = data_subgroup[data_subgroup['subject'] != sub_omit]

            sample_size = data_jk.shape[0]
            
            # create a sample and start fitting
            data_jk_sample = Sample.from_pandas_dataframe(data_jk, rt_column_name="RT", correct_column_name="accept")
        
            fit_sample = fit_adjust_model(sample = data_jk_sample, model = model_rDDM_ib, fitting_method = 'differential_evolution')
        
            # sort out results
            result_list = [[domain, group, sub_omit] + fit_sample.get_model_parameters() + \
                           [fit_sample.fitresult.value(), fit_sample.fitresult.loss, sample_size, 0]]
            # 0 in the end is the place holder for the counts in the on_buond column
        
            result_df = pd.DataFrame(result_list, columns = column_names)
            
            # check whether the estimated results are on the limits of the fitting ranges
            if result_df['vg'][0] == vg_min or result_df['vg'][0] == vg_max:
                result_df['on_bound'][0] += 1
    
            if result_df['vl'][0] == vl_min or result_df['vl'][0] == vl_max:
                result_df['on_bound'][0] += 1
    
            if result_df['fixed'][0] == fixed_min or result_df['fixed'][0] == fixed_max:
                result_df['on_bound'][0] += 1
    
            if result_df['B'][0] == B_min or result_df['B'][0] == B_max:
                result_df['on_bound'][0] += 1
    
            if result_df['x0'][0] == x0_min or result_df['x0'][0] == x0_max:
                result_df['on_bound'][0] += 1
    
            if result_df['nondectime'][0] == nondectime_min or result_df['nondectime'][0] == nondectime_max:
                result_df['on_bound'][0] += 1
        
            # append the results
            rDDM_fitting_jk = rDDM_fitting_jk.append(result_df, ignore_index=True)
     
           
    return rDDM_fitting_jk





# =============================================================================
# Fitting for the whole groups (without JK method)
# =============================================================================

### decide which domain to fit.
# domain_all = 'whole'
domain_all = 'adv'
# domain_all = 'dis'

fitting_all_ib = Fit_DDM_Group_All_ib(domain = domain_all)

### Save the results
""" Is_test """
# it is convenient to distinguish whether a run is for a test which uses larger dx and dt
Is_test = True
# Is_test = False

# create a file name for the df "fitting_result"
if not Is_test:
    df_name = 'DDM_GrpRL_All_' + str(domain_all) + '_ib201' + ".csv"
else:
    df_name = 'Test_DDM_GrpRL_All_' + str(domain_all) + '_ib201' + ".csv"
    
# assign the saving folder
save_folder = '/Users/hutianqi/Desktop/Project Risk and Intelligence/02 Fitting Programs/Z1_Output/'
# save_folder = '/home/bs/bsth4/DDM/Z1_DDM_Output/'

df_saving_address = save_folder + df_name

fitting_all_ib.to_csv(df_saving_address, index=False)


# =============================================================================
# Fitting for the JK subsets
# =============================================================================

### decide which domain to fit.
# domain_jk = 'whole'
# domain_jk = 'adv'
domain_jk = 'dis'

fitting_jk_ib = Fit_DDM_Group_JK_ib(domain = domain_jk)

### Save the results
""" Is_test """
# it is convenient to distinguish whether a run is for a test which uses larger dx and dt
Is_test = True
# Is_test = False

# create a file name for the df "fitting_result"
if not Is_test:
    df_name = 'DDM_GrpRL_JK_' + str(domain_jk) + '_ib201' + ".csv"
else:
    df_name = 'Test_DDM_GrpRL_JK_' + str(domain_jk) + '_ib201' + ".csv"
    
# assign the saving folder
save_folder = '/Users/hutianqi/Desktop/Project Risk and Intelligence/02 Fitting Programs/Z1_Output/'
# save_folder = '/home/bs/bsth4/DDM/Z1_DDM_Output/'

df_saving_address = save_folder + df_name

fitting_jk_ib.to_csv(df_saving_address, index=False)


# =============================================================================
# Run Time
# =============================================================================

run_time = time.time() - start_time
print("--- %s minutes %.2s seconds ---" % (int(run_time // 60), run_time % 60))

# =============================================================================
# 
# =============================================================================


# where = np.where((fit_ref['domain'] == 'adv') & 
#                   (fit_ref['RS_level'] == 3) & 
#                   (fit_ref['sub_omit'] == 'None'))
    

# where = int(where[0])



# where = np.where((fit_ref['domain'] == 'adv') & 
#                   (fit_ref['RS_level'] == 3) & 
#                   (fit_ref['sub_omit'] == 2))
    
# where = np.where((fit_ref['domain'] == 'adv') & 
#                   (fit_ref['RS_level'] == 3) & 
#                   (fit_ref['sub_omit'] == '2'))

# where = int(where[0])









