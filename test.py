# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 16:43:50 2022

@author: Lenovo
"""

import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from pymdp import utils
from pymdp.agent import Agent

verbose = False

context_names = ['D1', 'D2', 'D3']
choice_names = ['Start', 'ChD1', 'ChD2', 'ChD3']

""" Define `num_states` and `num_factors` below """
num_states = [len(context_names), len(choice_names)]
num_factors = len(num_states)

context_action_names = ['Do-nothing']
choice_action_names = ['Start', 'ChD1', 'ChD2', 'ChD3']

""" Define `num_controls` below """
num_controls = [len(context_action_names), len(choice_action_names)]

behavior_obs_names = ['Null', 'Win', 'Lose']
choice_obs_names = ['Start', 'ChD1', 'ChD2', 'ChD3']

""" Define `num_obs` and `num_modalities` below """
num_obs = [len(behavior_obs_names), len(choice_obs_names)]
num_modalities = len(num_obs)


########
##A

A = utils.obj_array( num_modalities )

prob_win = [0, 0, 1] # what is the probability of being good (element 0) and being bad (element 1)
p_D1 = 0.55
p_D2 = 0.55
p_D3 = 0.55


A_behavior = np.zeros((len(behavior_obs_names), len(context_names), len(choice_names)))

A_choice = np.zeros((len(choice_obs_names), len(context_names), len(choice_names)))

for choice_id, choice_name in enumerate(choice_names):
    
    if choice_name == 'Start':

        A_behavior[0,:,choice_id] = 1.0
    
    elif choice_name == 'ChD1':
        
        A_behavior[1:,:,choice_id] = np.array([ [p_D1, 0 , 0], [1-p_D1, 0, 0]])
    
    elif choice_name == 'ChD2':
        
        A_behavior[1:,:,choice_id] = np.array([ [0, p_D2, 0], [0, 1-p_D2, 0]])
    
    elif choice_name == 'ChD3':
        
        A_behavior[1:,:,choice_id] = np.array([ [0, 0, p_D3], [0, 0, 1-p_D3]])
        
  
A[0] = A_behavior

A_choice = np.zeros((len(choice_obs_names), len(context_names), len(choice_names)))

for choice_id in range(len(choice_names)):

  A_choice[choice_id, :, choice_id] = 1.0

A[1] = A_choice

##B
B = utils.obj_array(num_factors)

B_context = np.zeros( (len(context_names), len(context_names), len(context_action_names)) )

B_context[:,:,0] = np.eye(len(context_names))

B[0] = B_context

B_choice = np.zeros( (len(choice_names), len(choice_names), len(choice_action_names)) )

for choice_i in range(len(choice_names)):
  
  B_choice[choice_i, :, choice_i] = 1.0 # you can observe your actions without ambiguity

B[1] = B_choice