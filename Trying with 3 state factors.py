# -*- coding: utf-8 -*-
"""
A simulation of the experiment with 3 state factors. (only with  decks)

Created on Tue Feb 17 12:11:06 2022
@author: Lenovo
"""

import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from pymdp import utils
from pymdp.agent import Agent

verbose = False

D1_names = ['High','Low']
D2_names = ['High','Low']
choice_names = ['Start', 'ChD1', 'ChD2']

""" Define `num_states` and `num_factors` below """
num_states = [len(D1_names), len(D2_names), len(choice_names)]
num_factors = len(num_states)

context_action_names = ['Do-nothing']
choice_action_names = ['Start', 'ChD1', 'ChD2']

""" Define `num_controls` below """
num_controls = [len(context_action_names), len(choice_action_names)]

behavior_obs_names = ['Null', 'High', 'Low']
choice_obs_names = ['Start', 'ChD1', 'ChD2']

""" Define `num_obs` and `num_modalities` below """
num_obs = [len(behavior_obs_names), len(choice_obs_names)]
num_modalities = len(num_obs)


########
##A

A = utils.obj_array( num_modalities )

prob_win = [0.5,0.5] # what is the probability of win for each deck

pH1 = 0.55
pH2 = 0.6

# 3x2x2x3 = 36 cells
A_behavior = np.zeros((len(behavior_obs_names), len(D1_names), len(D2_names), len(choice_names)))

# 3x2x2x3 = 36 cells
A_choice = np.zeros((len(choice_obs_names), len(D1_names), len(D2_names), len(choice_names)))

for choice_id, choice_name in enumerate(choice_names):
    
    if choice_name == 'Start':

        A_behavior[0,:,:,choice_id] = 1.0

    elif choice_name == 'ChD1':
        
        A_behavior[choice_id,:,:] = np.array([[0,pH1,1-pH1],[0,1-pH1,pH1]])
    
    elif choice_name == 'ChD2':
        
        A_behavior[choice_id,:,:] = np.array([[0,pH2,1-pH2],[0,1-pH2,pH2]])
        
#dit is waarschijnlijk verkeerd, nog bekijken.^

  
A[0] = utils.norm_dist(A_behavior)

A_choice = np.zeros((len(choice_obs_names), len(D1_names), len(D2_names), len(choice_names)))

for choice_id in range(len(choice_names)):

  A_choice[choice_id, :,:, choice_id] = 1.0

A[1] = A_choice

##B
B = utils.obj_array(num_factors)

B_context1 = np.zeros( (len(D1_names), len(D1_names), len(context_action_names))) 

B_context1[:,:,0] = np.eye( len(D1_names))

B[0] = B_context1


B_context2 = np.zeros( (len(D2_names), len(D2_names), len(context_action_names))) 

B_context2[:,:,0] = np.eye( len(D1_names))

B[1] = B_context2


B_choice = np.zeros( (len(choice_names), len(choice_names), len(choice_action_names)) )

for choice_i in range(len(choice_names)):
  
  B_choice[choice_i, :, choice_i] = 1.0 # you can observe your actions without ambiguity

B[2] = B_choice

##C

from pymdp.maths import softmax

C = utils.obj_array_zeros([3, 3])
C[0][1] = 0.8 #higher preference for high reward
C[0][2] = 0.4


##D
D = utils.obj_array(num_factors)
D_context1 = np.array([0.5,0.5])
D_context2 = np.array([0.5,0.5])

D[0] = D_context1
D[1] = D_context2

D_choice = np.zeros(len(choice_names))

D_choice[choice_names.index("Start")] = 1.0

D[2] = D_choice

############################################

my_agent = Agent(A = A, B = B, C = C, D = D)

class omgeving(object):

  def __init__(self, context = None, p_consist = 0.8):

    self.context_names = ['High','Low']

    if context == None:
      self.context = self.context_names[utils.sample(np.array(prob_win))] # randomly sample
    else:
      self.context = context

    self.p_consist = p_consist

    self.behavior_obs_names = ['Null', 'High', 'Low']


  def step(self, action):

    if action == "Start":
      observed_behavior = "Null"
      observed_choice   = "Start"

    elif action == "ChD1":
      observed_choice = "ChD1"
      if self.context == "High":
        observed_behavior = self.behavior_obs_names[utils.sample(np.array([0.0, self.p_consist, 1 - self.p_consist]))]
      else:
        observed_behavior = self.behavior_obs_names[utils.sample(np.array([0.0, 1 - self.p_consist, self.p_consist]))]
    
    elif action == "ChD2":
      observed_choice = "ChD2"
      if self.context == "High":
        observed_behavior = self.behavior_obs_names[utils.sample(np.array([0.0, self.p_consist, 1 - self.p_consist]))]
      else:
        observed_behavior = self.behavior_obs_names[utils.sample(np.array([0.0, 1 - self.p_consist, self.p_consist]))]
        
    # elif action == "ChD3":
    #   observed_choice = "ChD3"
    #   if self.context == "D3":
    #     observed_behavior = self.behavior_obs_names[utils.sample(np.array([0.0, self.p_consist, 1 - self.p_consist]))]
    #   elif self.context in ("D1","D2"):
    #     observed_behavior = self.behavior_obs_names[utils.sample(np.array([0.0, 1 - self.p_consist, self.p_consist]))]
    
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
        choice_action = 'ChD1'
    else:
        chosen_action_id = my_agent.sample_action()
        print('chosen action id',chosen_action_id )
    
        movement_id = int(chosen_action_id[2])
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

p_consist = 0.8 # This is how consistent behavior is with actual character
env = omgeving(p_consist = p_consist)
T = 12
entr = run_active_inference_loop(my_agent, env, T = T)

