# -*- coding: utf-8 -*-
"""
effect of horizon effect and plots of behavior
12/10/22
@author: Lenovo
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pymdp import utils
from pymdp.agent import Agent
import statistics
verbose = False
verbose2 = True
  
D1_names = ['High','Low']
D2_names = ['High','Low']
D3_names = ['High', 'Low']
choice_names = ['Start', 'ChD1', 'ChD2','ChD3']

""" Define `num_states` and `num_factors` below """
num_states = [len(D1_names), len(D2_names),len(D3_names), len(choice_names)]
num_factors = len(num_states)

context_action_names = ['Do-nothing']
choice_action_names = ['Start', 'ChD1', 'ChD2','ChD3']

""" Define `num_controls` below """
num_controls = [len(context_action_names), len(choice_action_names)]

reward_obs_names = ['Null', 'High', 'Low']
choice_obs_names = ['Start', 'ChD1', 'ChD2','ChD3']

""" Define `num_obs` and `num_modalities` below """
num_obs = [len(reward_obs_names), len(choice_obs_names)]
num_modalities = len(num_obs)

## A matrix
   
A = utils.obj_array( num_modalities )

# reality low reward context (good or bad)
Plow_GB1 = [0.3,0.7] 
Plow_GB2 = [0.3,0.7]
Plow_GB3 = [0,1] 
# reality high reward context 
Phigh_GB1 = [0.3,0.7]
Phigh_GB2 = [1,0]     
Phigh_GB3 = [1,0] 

# for step function
prob_win_good = 0.7 # what is the probability of high reward for a good deck
prob_win_bad  =  0.3  # what is the probability of high reward for a bad deck

# probabilities according to the generative model (for A-matrix)
pH1_G = prob_win_good #chance to see high reward if deck 1 is good
pH1_B = prob_win_bad #chance to see high reward if deck 1 is bad
pH2_G = prob_win_good  #chance to see high reward if deck 2 is good
pH2_B = prob_win_bad #chance to see high reward if deck 2 is bad
pH3_G = prob_win_good #chance to see high reward is deck 3 is good #??
pH3_B = prob_win_bad #chance to see high reward if deck 3 is bad  #??
  
# 3x2x2x2x4 = 96 cells
A_reward = np.zeros((len(reward_obs_names), len(D1_names), len(D2_names),len(D3_names), len(choice_names)))

# 4x2x2x2x4 = 128 cells
A_choice = np.zeros((len(choice_obs_names), len(D1_names), len(D2_names),len(D3_names), len(choice_names)))

# with probabilities in each cell
for choice_id, choice_name in enumerate(choice_names):
    
    if choice_name == 'Start':

        A_reward[0,:,:,:,choice_id] = 1.0

    elif choice_name == 'ChD1':
        
        for i in range(len(D2_names)):
            for loop in range(len(D3_names)):
                A_reward[1:,:,i,loop,choice_id] = np.array([[pH1_G,pH1_B],[1-pH1_G,1-pH1_B]])
        
    elif choice_name == 'ChD2':
        
        for i in range(len(D1_names)):
            for loop in range(len(D2_names)):
                A_reward[1:,i,loop,:,choice_id] = np.array([[pH2_G,pH2_B],[1-pH2_G,1-pH2_B]])

    elif choice_name == 'ChD3':
        
        for i in range(len(D3_names)):
            for loop in range(len(D1_names)):
                A_reward[1:,loop,:,i,choice_id] = np.array([[pH3_G,pH3_B],[1-pH3_G,1-pH3_B]])

A[0] = A_reward

A_choice = np.zeros((len(choice_obs_names), len(D1_names), len(D2_names),len(D3_names), len(choice_names)))

for choice_id in range(len(choice_names)):

  A_choice[choice_id, :,:,:, choice_id] = 1.0

A[1] = A_choice

## B matrix (4 arrays because 4 state factors)

B = utils.obj_array(num_factors)

B_context1 = np.zeros((len(D1_names), len(D1_names), len(context_action_names))) 
B_context1[:,:,0] = np.eye(len(D1_names))
B[0] = B_context1

B_context2 = np.zeros((len(D2_names), len(D2_names), len(context_action_names))) 
B_context2[:,:,0] = np.eye( len(D2_names))
B[1] = B_context2

B_context3 = np.zeros((len(D3_names), len(D3_names), len(context_action_names)))
B_context3[:,:,0] = np.eye( len(D3_names))
B[2] = B_context3

B_choice = np.zeros((len(choice_names), len(choice_names), len(choice_action_names)))

for choice_i in range(len(choice_names)):
    
  B_choice[choice_i, :, choice_i] = 1.0 ##

B[3] = B_choice

## C matrix

from pymdp.maths import softmax
C = utils.obj_array_zeros([3, 4])
C[0][1] = 0.6 #higher preference for high reward
C[0][2] = 0.3

## D matrix (high and low reward equaly likely for all decks in start.)
D = utils.obj_array(num_factors)
D_context1 = np.array([0.5,0.5])
D_context2 = np.array([0.5,0.5])
D_context3 = np.array([0.5,0.5])

D[0] = D_context1
D[1] = D_context2
D[2] = D_context3

D_choice = np.zeros(len(choice_names))
D_choice[choice_names.index("Start")] = 1.0
D[3] = D_choice

#--------------------------------------------------------------------------------

## making the environment
    
class omgeving(object):

  def __init__(self, rewardcontext = 'High'):

    self.Z = ['Good','Bad']
    
    #high or low reward context
    if rewardcontext == 'High':
      self.D1 = self.Z[utils.sample(np.array(Phigh_GB1))]
      self.D2 = self.Z[utils.sample(np.array(Phigh_GB2))]
      self.D3 = self.Z[utils.sample(np.array(Phigh_GB3))] # (good or bad)
    else:
      self.D1 = self.Z[utils.sample(np.array(Plow_GB1))]
      self.D2 = self.Z[utils.sample(np.array(Plow_GB2))]
      self.D3 = self.Z[utils.sample(np.array(Plow_GB3))] # (good or bad
    
    self.reward_obs_names = ['High', 'Low']

  def step(self, action):

    if action == "Start": 
      observed_reward = "Null"
      observed_choice   = "Start"

    elif action == "ChD1":
      observed_choice = "ChD1"
      if self.D1 == "Good":
        observed_reward = self.reward_obs_names[utils.sample(np.array([prob_win_good,round(1- prob_win_good,1)]))]
      else:
        observed_reward = self.reward_obs_names[utils.sample(np.array([prob_win_bad, round(1-prob_win_bad,1)]))]
        
    elif action == "ChD2":
      observed_choice = "ChD2"
      if self.D2 == "Good":
        observed_reward = self.reward_obs_names[utils.sample(np.array([prob_win_good,round(1- prob_win_good,1)]))]
      else:
        observed_reward = self.reward_obs_names[utils.sample(np.array([prob_win_bad, round(1-prob_win_bad,1)]))]
   
    elif action == "ChD3":
      observed_choice = "ChD3"
      if self.D3 == "Good":
        observed_reward = self.reward_obs_names[utils.sample(np.array([prob_win_good,round(1- prob_win_good,1)]))]
      else:
        observed_reward = self.reward_obs_names[utils.sample(np.array([prob_win_bad, round(1-prob_win_bad,1)]))]
    
    obs = [observed_reward, observed_choice]

    return obs

env_high = omgeving() #high reward context omgeving
env_low = omgeving(rewardcontext = 'low') #low reward context omgeving

## making the active inference loop for each information condition (equal/unequal)

forced = 6 #amount of forced choice trials
T = 12 #amount of trials

def run_active_inference_loop(my_agent, my_env, T = T, equal = False):

  #Initialize the first observation
  obs_label = ["Null", "Start"]  # agent observes a `Null` reward, and seeing itself in the `Start` location
  obs = [reward_obs_names.index(obs_label[0]), choice_obs_names.index(obs_label[1])]
  if verbose:
      print('Initial observation:',obs)
  chosendecks = [0] # (0 for start action)
  strategy = ''
  chosendecks = [0]
  valD1, valD2, valD3 = [],[],[]
  exploitaction = 0
  
  for t in range(T):
      #print(obs)
      # verdeling q(s) in de variable s
      qs = my_agent.infer_states(obs)  # agent changes beliefs about states based on observation
      q_pi, efe = my_agent.infer_policies() #based on beliefs agent gives value to actions
      
      if verbose:
          print("Beliefs about the decks reward:", qs[0], qs[1], qs[2])
          print('EFE for each action:', efe)
          print('Q_PI:', q_pi)
      #___________________________________________________________________________________
      
      ## FREE CHOICE TRIALS
      if t > forced:
        chosen_action_id = my_agent.sample_action()   #agent choses action with less negative expected free energy
        movement_id = int(chosen_action_id[3])
        choice_action = choice_action_names[movement_id]
        if verbose:
            print('chosen action id',chosen_action_id)
            print("movement id", movement_id)
            print("Chosen action:", choice_action)
            
        #store strategy that is used in the first free choice trial for plot
        if t == forced +1:
            #print('Chosen action in first free choice trial:', choice_action)
            if exploitaction is None:
                strategy = 'None'
            else:
                if choice_action == 'ChD1' and equal is False:  #0 seen deck
                    strategy = 'Direct'
                elif choice_action == exploitaction:  #deck that had the most 'high rewards' in forced trials
                    strategy = 'Exploit'
                else:
                    strategy = 'Random'
      #_____________________________________________________________________________________________
          
      else:  
          ## FORCED CHOICE TRIALS
          my_agent.sample_action() 
          
          #unequal condition
          if equal == False:
               if t < (1/3)*forced:
                   choice_action = 'ChD3'
               elif t <= forced:
                   choice_action = 'ChD2'
                   
          #equal info condition
          else: 
               if t < (1/3)*forced:
                   choice_action = 'ChD1'
               elif t < 2*((1/3)*forced):
                   choice_action = 'ChD3'
               elif t <= forced:
                   choice_action = 'ChD2'      
        
    #store the actions for choice plots (number of action)   
      if choice_action == 'ChD1':
          chosendecks.append(1)
      elif choice_action == 'ChD2':
          chosendecks.append(2)
      else:
          chosendecks.append(3)

      obs_label = my_env.step(choice_action) #use step methode in 'omgeving' to generate new observation
      obs = [reward_obs_names.index(obs_label[0]), choice_obs_names.index(obs_label[1])]
      
      if verbose:
          print(f'Action at time {t}: {choice_action}')
          print(f'Reward at time {t}: {obs_label[0]}')
          print(f'New observation:',choice_action,'&', reward_obs_names[obs[0]] + ' reward', '\n')
      #______________________________________________________________________________________________
      
      ##storing reward value for each deck in forced choice trials
      if choice_action == 'ChD1':
          valD1.append(reward_obs_names[obs[0]])
      elif choice_action == 'ChD2':
          valD2.append(reward_obs_names[obs[0]])
      else:
          valD3.append(reward_obs_names[obs[0]])
  
      if t == forced:      
          
          #to get the most rewarding deck in the forced choice trials
          #choosing this deck again will represent the exploitation action in the first free choice trial
          D2high,D3high = valD2.count('High')/len(valD2),valD3.count('High')/len(valD3)
          if len(valD1) != 0:
              D1high = valD1.count('High')/len(valD1)
          else:
              D1high = 0
          
          highest = max(D1high,D2high,D3high)

          if highest == D1high:
              exploitaction = 'ChD1'
          elif highest == D2high:
              exploitaction = 'ChD2'
          else:
              exploitaction = 'ChD3'
          
          #if the average reward is equal for at least 2 options -> delete participant    
          if D1high in (D2high,D3high) or D2high == D3high:
              exploitaction = None
        
          #print('Most average reward in forced choice trials:', exploitaction)
        
        
  return chosendecks, strategy  #returns data for plotting

def plots(eq = False, rewardcontext = env_low, horizon = 1):  
    
    N = 100               #amount of participants
    d = {}                  #for choices at each timepoint
    strategy_list = []      #to store the strategy at the first free choice trial
    
    #make dictionary for behavior
    for timepoints in range(T):
        d[timepoints] = []
    
    for i in range(N):   
        #run the model
        my_agent = Agent(A = A, B = B, C = C, D = D, inference_horizon = horizon)
        choices, strategy = run_active_inference_loop(my_agent, rewardcontext, T = T, equal = eq)  #on this line you can change between high or low reward context
        strategy_list.append(strategy)
        
        #fill in values for each timepoint
        W = 0
        for timepoints in range(T):
            d[timepoints].append(choices[timepoints]) #timepoints choices
            
    meanchoices = []
    sd_c_max = []
    sd_c_min = []
    SDB1min, SDB1max = [],[]
    #compute mean value for each timepoint and variance (min and max))
    for t in range(T):
        
        meanchoices.append(statistics.mean(d[t]))
        sd_c_max.append(statistics.mean(d[t]) + statistics.variance(d[t]))
        sd_c_min.append(statistics.mean(d[t]) - statistics.variance(d[t]))
    
    timepoints = []
    for t in range(T):
        timepoints.append(t)
    
    ## This shows which deck the agent is chosing over time
    plt.figure()
    ticks = ['Deck A', 'Deck B', 'Deck C']
    plt.yticks([])
    plt.yticks([1,2,3], ticks)
    plt.ylim(0,3.1)
    plt.scatter(timepoints, meanchoices)
    plt.plot(timepoints, meanchoices)
    plt.fill_between(timepoints,sd_c_min,sd_c_max, alpha= 0.3) 
    
    hor = str(horizon)
    if eq is True:
        plt.title('Behavior (equal condition), Horizon:'+hor)
    else:
        plt.title('Behavior (unequal condition), Horizon:'+hor)       
    plt.ylabel('Which deck?')
    plt.xlabel('Time')
    plt.show()
    
    return strategy_list,N

strat_UQ_Low_Horizon,N = plots()
strat_UQ_High_Horizon,N = plots(horizon = 12)
strat_EQ_Low_Horizon,N = plots(eq = True)
strat_EQ_High_Horizon,N = plots(eq = True, horizon = 12)

def barchart(x, N, title):
    No = x.count('None')
    Random = x.count('Random')/(N-No)
    Direct = x.count('Direct')/(N-No)
    Exploit = x.count('Exploit')/(N-No)
    
    plt.figure()
    plt.yticks(np.arange(0, 1.25, 0.25))
    plt.ylim(0,1)

    X_axis = np.arange(1)
    
    plt.bar(X_axis - 0.2, Direct, 0.2, label = 'Directed')
    plt.bar(X_axis -0.0, Exploit, 0.2, label = 'Exploit')
    plt.bar(X_axis + 0.2, Random, 0.2, label = 'Random')
    
    plt.xlabel("Context")
    plt.ylabel("Choice Probability")
    plt.title(title)
    plt.legend()
    plt.show()

barchart(strat_EQ_Low_Horizon,N, 'Equal, Low Horizon')
barchart(strat_EQ_High_Horizon,N, 'Equal, High Horizon')
barchart(strat_UQ_Low_Horizon,N, 'Unequal, Low Horizon')
barchart(strat_UQ_High_Horizon, N, 'Unequal, High Horizon')
