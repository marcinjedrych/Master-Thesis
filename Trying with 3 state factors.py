# -*- coding: utf-8 -*-
"""
A simulation of the experiment with 3 state factors. (only with 2 decks)

Created on Tue Feb 26 13:18:06 2022
@author: Marcin Jedrych
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

reward_obs_names = ['Null', 'High', 'Low']
choice_obs_names = ['Start', 'ChD1', 'ChD2']

""" Define `num_obs` and `num_modalities` below """
num_obs = [len(reward_obs_names), len(choice_obs_names)]
num_modalities = len(num_obs)


########
##A

A = utils.obj_array( num_modalities )

prob_win1 = [0.5,0.5] # what is the probability of high and low reward for deck1
prob_win2 = [0.6,0.4] # what is the probability of high and low reward for deck2

#probabilities according to the generative model
pH1_G = 0.7 #chance to see high reward if deck 1 is good
pH1_B = 0.4 #chance to see high reward if deck 1 is bad
pH2_G = 0.7  #chance to see high reward if deck 2 is good
pH2_B = 0.4 #chance to see high reward if deck 2 is bad

# 3x2x2x3 = 36 cells
A_reward = np.zeros((len(reward_obs_names), len(D1_names), len(D2_names), len(choice_names)))

# 3x2x2x3 = 36 cells
A_choice = np.zeros((len(choice_obs_names), len(D1_names), len(D2_names), len(choice_names)))

for choice_id, choice_name in enumerate(choice_names):
    
    if choice_name == 'Start':

        A_reward[0,:,:,choice_id] = 1.0

    elif choice_name == 'ChD1':
        
        for loop in range(len(D1_names)):
            A_reward[1:,:,loop, choice_id] = np.array([[pH1_G,pH1_B],[1-pH1_G,1-pH1_B]])
    
    elif choice_name == 'ChD2':
        
        for loop in range(len(D2_names)):
            A_reward[1:,loop,:, choice_id] = np.array([[pH2_G,pH2_B],[1-pH2_G,1-pH2_B]])

A[0] = A_reward

A_choice = np.zeros((len(choice_obs_names), len(D1_names), len(D2_names), len(choice_names)))

for choice_id in range(len(choice_names)):

  A_choice[choice_id, :,:, choice_id] = 1.0

A[1] = A_choice

##B (3 arrays because 3 state factors)
B = utils.obj_array(num_factors)

B_context1 = np.zeros( (len(D1_names), len(D1_names), len(context_action_names))) 

B_context1[:,:,0] = np.eye(len(D1_names))

B[0] = B_context1


B_context2 = np.zeros((len(D2_names), len(D2_names), len(context_action_names))) 

B_context2[:,:,0] = np.eye( len(D1_names))

B[1] = B_context2


B_choice = np.zeros((len(choice_names), len(choice_names), len(choice_action_names)))

for choice_i in range(len(choice_names)):
  
  B_choice[choice_i, :, choice_i] = 1.0 ##

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
#high and low reward equaly likely for both decks in start.^

D[0] = D_context1
D[1] = D_context2

D_choice = np.zeros(len(choice_names))

D_choice[choice_names.index("Start")] = 1.0 #agent in start

D[2] = D_choice

############################################

my_agent = Agent(A = A, B = B, C = C, D = D)  # = generative model

class omgeving(object):      # = generative process

  def __init__(self, context = None, p_consist = 0.8):

    self.context_names = ['High','Low']
    self.context = context
    self.p_consist = p_consist
    self.reward_obs_names = ['Null', 'High', 'Low']

  def step(self, action):

    if action == "Start":
        observed_reward = "Null"
        observed_choice   = "Start"

    elif action == "ChD1":
        self.context = self.context_names[utils.sample(np.array(prob_win1))]
        observed_choice = "ChD1"
        if self.context == "High":
            observed_reward = self.reward_obs_names[utils.sample(np.array([0.0, self.p_consist, 1 - self.p_consist]))]
        else:
            observed_reward = self.reward_obs_names[utils.sample(np.array([0.0, 1 - self.p_consist, self.p_consist]))]
    
    elif action == "ChD2":
        self.context = self.context_names[utils.sample(np.array(prob_win2))]
        observed_choice = "ChD2"
        if self.context == "High":
            observed_reward = self.reward_obs_names[utils.sample(np.array([0.0, self.p_consist, 1 - self.p_consist]))]
        else:
            observed_reward = self.reward_obs_names[utils.sample(np.array([0.0, 1 - self.p_consist, self.p_consist]))]
    
    obs = [observed_reward, observed_choice]

    return obs

def run_active_inference_loop(my_agent, my_env, T = 5):

  """ Initialize the first observation """
  obs_label = ["Null", "Start"]  # agent observes a `Null` reward, and seeing itself in the `Start` location
  obs = [reward_obs_names.index(obs_label[0]), choice_obs_names.index(obs_label[1])]
  print('Initial observation:',obs)
  chosendecks = ["Start"]
  High_or_Low = [obs_label[0]] #will make a list containing whether it is high or low reward on each timepoint (for plotting)
  
  for t in range(T):
    qs = my_agent.infer_states(obs)  # agent changes beliefs about states based on observation
    print("Beliefs about the decks reward: D1 =", qs[0],"D2 =", qs[1])

    q_pi, efe = my_agent.infer_policies() #based on beliefs agent gives value to actions
    print('EFE for each action:', efe)  #[Start, ChD1, ChD2]
    
    ##forced choice trials
    if t < 6:
        choice_action = 'ChD1'
    else:
        chosen_action_id = my_agent.sample_action()   #agent choses action with less negative expected free energy
        #print('chosen action id',chosen_action_id)
    
        movement_id = int(chosen_action_id[2])
        #print("movement id", movement_id)
         
        choice_action = choice_action_names[movement_id]
        print("Chosen action:", choice_action)
    
    chosendecks.append(choice_action)
    obs_label = my_env.step(choice_action)
    #print("obslabel", obs_label)

    obs = [reward_obs_names.index(obs_label[0]), choice_obs_names.index(obs_label[1])]

    print(f'Action at time {t}: {choice_action}')      
    print(f'Reward at time {t}: {obs_label[0]}')
    High_or_Low.append(obs_label[0][0])
    print(f'New observation:',choice_action,'&', reward_obs_names[obs[0]] + ' reward', '\n')
    
  return chosendecks, High_or_Low

p_consist = 0.8 # This is how consistent reward is with actual
env = omgeving(p_consist = p_consist)
T = 50

timepoints = [0]
for t in range(T):
    timepoints.append(t)
    
choices, rewards = run_active_inference_loop(my_agent, env, T = T)

#This shows which deck the agent is chosing over time and which reward the agent gets at each timepoint
plt.scatter(timepoints, choices)
for i, txt in enumerate(rewards):
    plt.annotate(txt, (timepoints[i], choices[i]))    
plt.plot(timepoints, choices)
plt.title('Behavior of the model')
plt.ylabel('Which deck?')
plt.xlabel('Time')
plt.text(x = -1.25,y = 2.13,s = 'H = High reward' + '\n' + 'L = Low Reward')
plt.show()

#First agent is in 'Start', next there are 6 forced choices followed by free choice trials.