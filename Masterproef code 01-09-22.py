# -*- coding: utf-8 -*-
"""
Last update: 1/09/2022

Active inference model with 4 state factors, info conditions and mean plots of beliefs and behavior.
+ plot directed exploration vs random exploration vs exploitation.
+ high/low reward context argument (for now different probabilities of reward)
@author: Marcin Jedrych
"""

import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from pymdp import utils
from pymdp.agent import Agent
import statistics
verbose = False
  
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
Plow_GB1 = [0.5,0.5] 
Plow_GB2 = [1,0]  #best deck
Plow_GB3 = [0,1]  #worst deck
# reality high reward context 
Phigh_GB1 = [0.7,0.3]
Phigh_GB2 = [1,0]     #best deck
Phigh_GB3 = [0.5,0.5] #worst deck

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
  
 
# function for agent
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

###############################################################################
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

###############################################################################
## making the active inference loop for each information condition (equal/unequal)

forced = 6 #amount of forced choice trials

def run_active_inference_loop(my_agent, my_env, T = 5, equal = True):

  """ Initialize the first observation """
  obs_label = ["Null", "Start"]  # agent observes a `Null` reward, and seeing itself in the `Start` location
  obs = [reward_obs_names.index(obs_label[0]), choice_obs_names.index(obs_label[1])]
  if verbose:
      print('Initial observation:',obs)
  chosendecks = [0] # (0 for start action)
  
  deck1,deck2,deck3 = [],[],[]
  strategy = ''
  
  for t in range(T):
      #print(obs)
      # verdeling q(s) in de variable s
      qs = my_agent.infer_states(obs)  # agent changes beliefs about states based on observation
     
      
      #store the beliefs for plots about beliefs
      deck1.append(qs[0])
      deck2.append(qs[2])
      deck3.append(qs[1])

      q_pi, efe = my_agent.infer_policies() #based on beliefs agent gives value to actions
      
      if verbose:
          print("Beliefs about the decks reward:", qs[0], qs[1], qs[2])
          print('EFE for each action:', efe)
          print('Q_PI:', q_pi)
      
     ## FREE CHOICE TRIALS
      if t > forced:
        chosen_action_id = my_agent.sample_action()   #agent choses action with less negative expected free energy
  
        movement_id = int(chosen_action_id[3])
       
        choice_action = choice_action_names[movement_id]
        
        if verbose:
            print('chosen action id',chosen_action_id)
            print("movement id", movement_id)
            print("Chosen action:", choice_action)
  
        #store strategy that is used in the first free choice trial for plot        ! [nog veranderen zodat dit altijd werkt en niet enkel als deck 2 het best is]
        if t == forced +1:
            if choice_action == 'ChD1' and equal is False:  #0 seen deck
                strategy = 'Direct'
            elif choice_action == 'ChD2':  #most rewarding deck
                strategy = 'Exploit'
            else:
                strategy = 'Random'
      else:      
          my_agent.sample_action()
      
     ## FORCED CHOICE TRIALS
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

  return chosendecks, deck1, deck2, deck3, strategy  #returns data for plotting

T = 12

## function for choice plots and beliefs plot
def plots(a,b, eq = True, rewardcontext = env_low, otherplots = True):  
    
    N = 20               #amount of participants
    d = {}                  #for choices at each timepoint
    b1,b2,b3 = {},{},{}     #for beliefs at each timepoint
    strategy_list = []      #to store the strategy at the first free choice trial
    
    #make dictionaries for behavior and beliefs
    for timepoints in range(T):
        d[timepoints],b1[timepoints],b2[timepoints],b3[timepoints] = [],[],[],[]
    
    for i in range(N):   
        #run the model
        my_agent = Agent(A = A, B = B, C = C, D = D, inference_horizon = 1)
        choices, deck1, deck2, deck3, strategy = run_active_inference_loop(my_agent, rewardcontext, T = T, equal = eq)  #on this line you can change between high or low reward context
        strategy_list.append(strategy)
        
        #fill in values for each timepoint
        W = 0
        for timepoints in range(T):
            d[timepoints].append(choices[timepoints]) #timepoints choices
            b1[timepoints].append(deck1[timepoints][0]) #timepoints beliefs deck1
            b2[timepoints].append(deck2[timepoints][0]) #timepoints belief deck 2
            b3[timepoints].append(deck3[timepoints][0]) #timepoints belief deck 3
            
    meanchoices,meanb1,meanb2,meanb3 = [],[],[],[]
    sd_c_max,sd_b1_max,sd_b2_max,sd_b3_max = [],[],[],[]
    sd_c_min,sd_b1_min,sd_b2_min,sd_b3_min = [],[],[],[]
    SDB1min, SDB1max = [],[]
    #compute mean value for each timepoint and variance (min and max))
    for t in range(T):
        
        meanchoices.append(statistics.mean(d[t]))
        sd_c_max.append(statistics.mean(d[t]) + statistics.variance(d[t]))
        sd_c_min.append(statistics.mean(d[t]) - statistics.variance(d[t]))

        
        meanb1.append(statistics.mean(b1[t]))
        sd_b1_max.append(statistics.mean(b1[t]) + statistics.variance(b1[t]))
        sd_b1_min.append(statistics.mean(b1[t]) - statistics.variance(b1[t]))
        #SDB1max.append(statistics.mean(b1[t]) + statistics.stdev(b1[t])/2)    #(with sd)
        
        SDB1min.append(statistics.mean(b1[t]) - statistics.stdev(b1[t])/2)
        
        meanb2.append(statistics.mean(b2[t]))
        sd_b2_max.append(statistics.mean(b2[t]) + statistics.variance(b2[t]))
        sd_b2_min.append(statistics.mean(b2[t]) - statistics.variance(b2[t]))
        
        meanb3.append(statistics.mean(b3[t]))
        sd_b3_max.append(statistics.mean(b3[t]) + statistics.variance(b3[t]))
        sd_b3_min.append(statistics.mean(b3[t]) - statistics.variance(b3[t]))
    
    timepoints = []
    for t in range(T):
        timepoints.append(t)
    
    if otherplots is True:
        ## This shows which deck the agent is chosing over time
        plt.figure(a)
        ticks = ['Deck A', 'Deck B', 'Deck C']
        plt.yticks([])
        plt.yticks([1,2,3], ticks)
        plt.ylim(0,3.1)
        plt.scatter(timepoints, meanchoices)
        plt.plot(timepoints, meanchoices)
        plt.fill_between(timepoints,sd_c_min,sd_c_max, alpha= 0.3) 
        if a == 1:
            plt.title('Behavior (equal condition)')
        elif a == 3:
            plt.title('Behavior (unequal condition)')   
        plt.ylabel('Which deck?')
        plt.xlabel('Time')
        
        ## This shows the beliefs (about high reward) over time
        plt.figure(b)
        plt.plot(timepoints,meanb1,label = 'P(Deck 1 = High)')
        plt.plot(timepoints,meanb2,label = 'P(Deck 2 = High)')
        plt.plot(timepoints,meanb3,label = 'P(Deck 3 = High)')
        plt.fill_between(timepoints,sd_b1_min,sd_b1_max, alpha= 0.3)            #SDB1min,SDB1max for Sd
        plt.fill_between(timepoints,sd_b2_min,sd_b2_max, alpha= 0.3)
        plt.fill_between(timepoints,sd_b3_min,sd_b3_max, alpha= 0.3)
        plt.legend()
        if b == 2:
            plt.title('Beliefs (equal condition)')
        elif b == 4:
            plt.title('Beliefs (unequal condition)')    
        plt.show()
    
    return strategy_list,N

#-------------------------------------------------------------------------------
##plot equal condition
for i in range(10):
    if i == 0:
        Random1, Exploit1, Directed1 = [],[],[]
        strategy1, N = plots(1,2, eq = True, rewardcontext = env_low)
    else: 
        strategy1, N = plots(1,2, eq = True, rewardcontext = env_low, otherplots = False)
    random1 = strategy1.count('Random')/N
    Random1.append(random1)
    exploit1 = strategy1.count('Exploit')/N
    Exploit1.append(exploit1)
    directed1 = strategy1.count('Direct')/N
    Directed1.append(directed1)

SdRandom1 = statistics.stdev(Random1)
SdExploit1 = statistics.stdev(Exploit1)

##plot unequal condition
for i in range(10):
    if i == 0:
        Random2, Exploit2, Directed2 = [],[],[]
        strategy2, N = plots(3,4, eq = False, rewardcontext = env_low) 
    else:
        strategy2, N = plots(3,4, eq = False, rewardcontext = env_low, otherplots = False) 
    random2 = strategy2.count('Random')/N
    Random2.append(random2)
    exploit2 = strategy2.count('Exploit')/N
    Exploit2.append(exploit2)
    directed2 = strategy2.count('Direct')/N
    Directed2.append(directed2)
    
SdRandom2 = statistics.stdev(Random2)
SdExploit2 = statistics.stdev(Exploit2)
SdDirected2 = statistics.stdev(Directed2)

print('EQUAL CONDITION:')
print(f'Random: {random1} and Directed: {directed1}')

print('UNEQUAL CONDITION')
print(f'Random: {random2} and Directed: {directed2}')

##bar chart exploit, directed and random exploration:
plt.figure(5)
plt.yticks(np.arange(0, 1.25, 0.25))
plt.ylim(0,1)
conditions = ['Unequal','Equal']
random = [random2, random1]
directed = [directed2, directed1]
exploit = [exploit2 , exploit1]
X_axis = np.arange(len(conditions))

plt.bar(X_axis - 0.2, directed, 0.2, label = 'Directed', yerr = [SdDirected2,0])
plt.bar(X_axis -0.0, exploit, 0.2, label = 'Exploit', yerr = [SdExploit2,SdExploit1])
plt.bar(X_axis + 0.2, random, 0.2, label = 'Random', yerr = [SdRandom2,SdRandom1])

plt.xticks(X_axis, conditions)
plt.axvline(x=0.5, color = "black")
plt.xlabel("Conditions")
plt.ylabel("Choice Probability")
plt.title("Strategy")
plt.legend()
plt.show()

#--------------------------------------------------------------------------------

###########################################################
## plot high vs low reward context in unequal condition:

strategy1, N = plots(1,2, eq = False, rewardcontext = env_low, otherplots = False)
random1 = strategy1.count('Random')/N
exploit1 = strategy1.count('Exploit')/N
directed1 = strategy1.count('Direct')/N

##plot high reward condition
strategy2, N = plots(3,4, eq = False, rewardcontext = env_high, otherplots = False) 
random2 = strategy2.count('Random')/N
exploit2 = strategy2.count('Exploit')/N
directed2 = strategy2.count('Direct')/N

print('LOW REWARD CONTEXT:')
print(f'Random: {random1} and Directed: {directed1}')

print('HIGH REWARD CONTEXT')
print(f'Random: {random2} and Directed: {directed2}')

##bar chart
plt.figure(6)
plt.yticks(np.arange(0, 1.25, 0.25))
plt.ylim(0,1)
conditions = ['high reward','low reward']
random = [random2, random1]
directed = [directed2, directed1]
exploit = [exploit2 , exploit1]
X_axis = np.arange(len(conditions))

plt.bar(X_axis - 0.2, directed, 0.2, label = 'Directed')
plt.bar(X_axis -0.0, exploit, 0.2, label = 'Exploit')
plt.bar(X_axis + 0.2, random, 0.2, label = 'Random')

plt.xticks(X_axis, conditions)
plt.axvline(x=0.5, color = "black")
plt.xlabel("Context")
plt.ylabel("Choice Probability")
plt.title("Unequal info codition")
plt.legend()
plt.show()

###########################################################
## plot high vs low reward context in equal condition:

strategy1, N = plots(1,2, eq = True, rewardcontext = env_low, otherplots = False)
random1 = strategy1.count('Random')/N
exploit1 = strategy1.count('Exploit')/N
directed1 = strategy1.count('Direct')/N

##plot high reward condition
strategy2, N = plots(3,4, eq = True, rewardcontext = env_high, otherplots = False) 
random2 = strategy2.count('Random')/N
exploit2 = strategy2.count('Exploit')/N
directed2 = strategy2.count('Direct')/N

print('LOW REWARD CONTEXT:')
print(f'Random: {random1} and Directed: {directed1}')

print('HIGH REWARD CONTEXT')
print(f'Random: {random2} and Directed: {directed2}')

##bar chart
plt.figure(7)
plt.yticks(np.arange(0, 1.25, 0.25))
plt.ylim(0,1)
conditions = ['high reward','low reward']
random = [random2, random1]
directed = [directed2, directed1]
exploit = [exploit2 , exploit1]
X_axis = np.arange(len(conditions))

plt.bar(X_axis - 0.2, directed, 0.2, label = 'Directed')
plt.bar(X_axis -0.0, exploit, 0.2, label = 'Exploit')
plt.bar(X_axis + 0.2, random, 0.2, label = 'Random')

plt.xticks(X_axis, conditions)
plt.axvline(x=0.5, color = "black")
plt.xlabel("Context")
plt.ylabel("Choice Probability")
plt.title("Equal info codition")
plt.legend()
plt.show()