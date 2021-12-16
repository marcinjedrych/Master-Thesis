# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 19:39:29 2021

@author: Marcin Jedrych
"""
import pymdp
import numpy as np
from pymdp import utils

# first forced choice task, then free choice task
# chose 1,2 or 3

#forced choice task
#free choice task

#specify dimensionalities of hidden state factors, control factors and observation modalities
context_names = ['123', '132','213','231','312','321'] #123 = 1 highest reward, 2 lower and 3 lowest reward
choice_names = ['Start','Deck 1','Deck 2','Deck 3']

""" Define `num_states` and `num_factors` below """
num_states = [len(context_names), len(choice_names)]
num_factors = len(num_states)

context_action_names = ['Do-nothing'] #?
choice_action_names = ['Move-start', 'Chose 1', 'Chose 2', 'Chose 3']

""" Define `num_controls` below """
num_controls = [len(context_action_names), len(choice_action_names)]

reward_obs_names = ['Null','Small reward', 'Medium reward', 'Big reward']
choice_obs_names = ['Start', 'Deck 1', 'Deck 2', 'Deck 3']

""" Define `num_obs` and `num_modalities` below """
num_obs = [len(reward_obs_names), len(choice_obs_names)]
num_modalities = len(num_obs)

""" Generate the A array """
#small, medium or big reward
A = utils.obj_array( num_modalities )

#reward modality
pr = 0.6
x = 1-pr

A_reward = np.zeros((len(reward_obs_names),len(context_names), len(choice_names)))
for choice_id, choice_name in enumerate(choice_names):
    if choice_name == 'Start':
        A_reward[:3,:,choice_id] = 0
        
    elif choice_name == 'Deck 1':
        A_reward[:3,:,choice_id] = np.array([[x/2, x/2, x/2, pr, x/2, pr],
                                              [x/2, x/2, pr, x/2, pr, x/2],
                                                [pr, pr, x/2, x/2, x/2, x/2]])
        
    elif choice_name == 'Deck 2':
        A_reward[1:,:,choice_id] = np.array([[x/2, pr, x/2, x/2, pr, x/2],
                                             [pr, x/2, x/2, x/2, x/2, pr],
                                             [x/2, x/2, pr, pr, x/2, x/2]])
    elif choice_name == 'Deck 3':
        A_reward[1:, :, choice_id] = np.array([[pr, x/2, pr, x/2, x/2, x/2],
                                               [x/2, pr, x/2, pr, x/2, x/2],
                                               [x/2, x/2, x/2, x/2, pr, pr]])
A[0] = A_reward

#choice observation modality
A_choice = np.zeros((len(choice_obs_names), len(context_names), len(choice_names)))

for choice_id in range(len(choice_names)):
  A_choice[choice_id, :, choice_id] = 1.0
A[1] = A_choice
print(A)

"""Generate the B array"""
B = utils.obj_array(num_factors)

#context state
B_context = np.zeros( (len(context_names), len(context_names), len(context_action_names)) )
B_context[:,:,0] = np.eye(len(context_names))
B[0] = B_context

B_choice = np.zeros( (len(choice_names), len(choice_names), len(choice_action_names)) )

for choice_i in range(len(choice_names)):
  B_choice[choice_i, :, choice_i] = 1.0
B[1] = B_choice

"""Generate the C array""" #prior preferences
C = utils.obj_array_zeros(num_obs)

from pymdp.maths import softmax #for the plot

C_reward = np.zeros(len(reward_obs_names))
C_reward[1] = -4.0 
C_reward[2] = 2.0 
C[1] = C_reward

""" Generate the D vectors"""
D = utils.obj_array(num_factors)

D_context = np.array([0.33,0.33,0.33])

D[0] = D_context

D_choice = np.zeros(len(choice_names))

D_choice[choice_names.index("Start")] = 1.0

D[1] = D_choice

print('-------------A')
print(A)
print('-------------B')
print(B)
print('-------------C')
print(C)
print('-------------D')
print(D)
print('-------------')
print(f'Beliefs about which arm is better: {D[0]}')
print(f'Beliefs about starting location: {D[1]}')


from pymdp.agent import Agent

my_agent = Agent(A = A, B = B, C = C, D = D)

# class TwoArmedBandit(object):

#   def __init__(self, context = None, p_hint = 1.0, p_reward = 0.8):

#     self.context_names = ['123', '132','213','231','312','321']

#     if context == None:
#       self.context = self.context_names[utils.sample(np.array([0.33,0.33,0.33]))] # randomly sample which bandit arm is better (Left or Right)
#     else:
#       self.context = context

#     self.p_reward = pr

#     self.reward_obs_names = ['Small reward', 'Medium reward', 'Big reward']


#   def step(self, action): 

#     if action == "Move-start":
#       observed_reward = "Null"
#       observed_choice = "Start"
#     elif action == "Chose 1":
#       if self.context == "Left-Better":
#         observed_hint = self.hint_obs_names[utils.sample(np.array([0.0, self.p_hint, 1.0 - self.p_hint]))]
#       elif self.context == "Right-Better":
#         observed_hint = self.hint_obs_names[utils.sample(np.array([0.0, 1.0 - self.p_hint, self.p_hint]))]
#       observed_reward = "Null"
#       observed_choice = "Hint"
#     elif action == "Chose 2":
#       observed_choice = "Left Arm"
#       if self.context == "Left-Better":
#         observed_reward = self.reward_obs_names[utils.sample(np.array([0.0, 1.0 - self.p_reward, self.p_reward]))]
#       elif self.context == "Right-Better":
#         observed_reward = self.reward_obs_names[utils.sample(np.array([0.0, self.p_reward, 1.0 - self.p_reward]))]
#     elif action == "Chose 3":
#       observed_hint = "Null"
#       observed_choice = "Right Arm"
#       if self.context == "Right-Better":
#         observed_reward = self.reward_obs_names[utils.sample(np.array([0.0, 1.0 - self.p_reward, self.p_reward]))]
#       elif self.context == "Left-Better":
#         observed_reward = self.reward_obs_names[utils.sample(np.array([0.0, self.p_reward, 1.0 - self.p_reward]))]
    
#     obs = [observed_hint, observed_reward, observed_choice]

#     return obs
  

