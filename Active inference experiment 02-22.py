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

prob_win = [0.5, 0.5, 0.7] # what is the probability of win

p_D1 = 0.55
p_D2 = 0.55
p_D3 = 0.55

# hier ben ik niet zeker over maar anders krijg ik de error dat A-matrix niet genormaliseerd is
k = (1- p_D1)/2
l = p_D1/2

A_behavior = np.zeros((len(behavior_obs_names), len(context_names), len(choice_names)))

A_choice = np.zeros((len(choice_obs_names), len(context_names), len(choice_names)))

for choice_id, choice_name in enumerate(choice_names):
    
    if choice_name == 'Start':

        A_behavior[0,:,choice_id] = 1.0
    
    #dit klopt nog niet
    elif choice_name == 'ChD1':
        
        A_behavior[1:,:,choice_id] = np.array([ [p_D1, k , k], [1-p_D1, l, l]])
    
    elif choice_name == 'ChD2':
        
        A_behavior[1:,:,choice_id] = np.array([ [k, p_D2, k], [l, 1-p_D2, l]])
    
    elif choice_name == 'ChD3':
        
        A_behavior[1:,:,choice_id] = np.array([ [k, k, p_D3], [l, l, 1-p_D3]])
        
  
A[0] = utils.norm_dist(A_behavior)

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

##C

from pymdp.maths import softmax

C = utils.obj_array_zeros([3, 4])
C[0][1] = 0.5


##D
D = utils.obj_array(num_factors)
D_context = np.array([1/3,1/3,1/3])

D[0] = D_context

D_choice = np.zeros(len(choice_names))

D_choice[choice_names.index("Start")] = 1.0

D[1] = D_choice

############################################

my_agent = Agent(A = A, B = B, C = C, D = D)

class Knowthyself(object):

  def __init__(self, context = None, p_consist = 0.8):

    self.context_names = ['D1', 'D2', 'D3']

    if context == None:
      self.context = self.context_names[utils.sample(np.array(prob_win))] # randomly sample
    else:
      self.context = context

    self.p_consist = p_consist

    self.behavior_obs_names = ['Null', 'Win', 'Lose']


  def step(self, action):

    if action == "Start":
      observed_behavior = "Null"
      observed_choice   = "Start"
	
    #dit klopt nog niet
    elif action == "ChD1":
      observed_choice = "ChD1"
      if self.context == "D1":
        observed_behavior = self.behavior_obs_names[utils.sample(np.array([0.0, self.p_consist, 1 - self.p_consist]))]
      elif self.context in ("D2","D3"):
        observed_behavior = self.behavior_obs_names[utils.sample(np.array([0.0, 1 - self.p_consist, self.p_consist]))]
    
    elif action == "ChD2":
      observed_choice = "ChD2"
      if self.context == "D2":
        observed_behavior = self.behavior_obs_names[utils.sample(np.array([0.0, self.p_consist, 1 - self.p_consist]))]
      elif self.context in ("D1","D3"):
        observed_behavior = self.behavior_obs_names[utils.sample(np.array([0.0, 1 - self.p_consist, self.p_consist]))]
        
    elif action == "ChD3":
      observed_choice = "ChD3"
      if self.context == "D3":
        observed_behavior = self.behavior_obs_names[utils.sample(np.array([0.0, self.p_consist, 1 - self.p_consist]))]
      elif self.context in ("D1","D2"):
        observed_behavior = self.behavior_obs_names[utils.sample(np.array([0.0, 1 - self.p_consist, self.p_consist]))]
    
    obs = [observed_behavior, observed_choice]

    return obs

def run_active_inference_loop(my_agent, my_env, T = 5):

  """ Initialize the first observation """
  obs_label = ["Null", "Start"]  # agent observes a `Null` behavior, and seeing itself in the `Start` location
  obs = [behavior_obs_names.index(obs_label[0]), choice_obs_names.index(obs_label[1])]
  ent = np.zeros((T, num_factors))
  
  for t in range(T):
    qs = my_agent.infer_states(obs)
    print("***", qs[0], qs[1])
    #print("QS", qs)

    q_pi, efe = my_agent.infer_policies()
    
    ##forced choice trials
    if t < 6:
        choice_action = 'ChD3'
    else:
        chosen_action_id = my_agent.sample_action()
        print('chosen action id',chosen_action_id )
    
        movement_id = int(chosen_action_id[1])
        print("movement id", movement_id , '\n')
        
        choice_action = choice_action_names[movement_id]
        print("choice action", choice_action, '\n')

    obs_label = my_env.step(choice_action)
    print("obslabel", obs_label, '\n')

    obs = [behavior_obs_names.index(obs_label[0]), choice_obs_names.index(obs_label[1])]
    print("obs", obs, '\n')

    print(f'Action at time {t}: {choice_action}')
    print(f'Behavior at time {t}: {obs_label[0]}')
    
  print("Q = ", qs[0]) 
  return 'test'

p_consist = 0.9 # This is how consistent behavior is with actual character
env = Knowthyself(p_consist = p_consist)

T = 24

my_agent = Agent(A = A, B = B, C = C, D = D) # redefine the agent with the new preferences

entr = run_active_inference_loop(my_agent, env, T = T)


if verbose:
	print('----------------A')
	print(A)
	print('----------------B')
	print(B)
	print('----------------C')
	print(C)
	print('----------------D')
	print(D)

my_agent = Agent(A = A, B = B, C = C, D = D) # redefine the agent with the new preferences