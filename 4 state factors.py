"""
A simulation of the experiment with 4 state factors. (with 3 decks)

Created on Tue Feb 17 12:11:06 2022
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


########
##A

A = utils.obj_array( num_modalities )

prob_win = [0.5,0.5] # what is the probability of high and low reward for each deck

#kansen volgens het geenertieve model
pH1_G = 0.7 #kans om high reward te zien als deck 1 goed is
pH1_B = 0.4 #kans om high reward te zien als deck 1 slecht is
pH2_G = 0.7  #kans om high reward te zien als deck 2 goed is
pH2_B = 0.4 #kans om high reward te zien als deck 2 slecht is
pH3_G = 0.7  #kans om high reward te zien als deck 3 goed is
pH3_B = 0.4  #kans om high reward te zien als deck 3 slecht is

# 3x2x2x2x4 =  96 cells
A_reward = np.zeros((len(reward_obs_names), len(D1_names), len(D2_names),len(D3_names), len(choice_names)))

# 4x2x2x2x4 = 128 cells
A_choice = np.zeros((len(choice_obs_names), len(D1_names), len(D2_names),len(D3_names), len(choice_names)))

#with probabilities in each cell
for choice_id, choice_name in enumerate(choice_names):
    
    if choice_name == 'Start':

        A_reward[0,:,:,:,choice_id] = 1.0

    elif choice_name == 'ChD1':
        
        for loop in range(2): # TV: dit is fout; je neemt een slice over 3 dimensions (nl 1:, :, :) terwijl je een slice over 2 dim moet nemen
							# kan je oplossen door 2 loops te nemen ipv 1	
            A_reward[1:,:,:,loop,choice_id] = np.array([[pH1_G,pH1_B],[1-pH1_G,1-pH1_B]])
        
    elif choice_name == 'ChD2':
        
        for loop in range(2):
            A_reward[1:,:,loop,:,choice_id] = np.array([[pH2_G,pH2_B],[1-pH2_G,1-pH2_B]])

    elif choice_name == 'ChD3':
        
        for loop in range(2):
            A_reward[1:,loop,:,:,choice_id] = np.array([[pH3_G,pH3_B],[1-pH3_G,1-pH3_B]])

A[0] = utils.norm_dist(A_reward)

A_choice = np.zeros((len(choice_obs_names), len(D1_names), len(D2_names),len(D3_names), len(choice_names)))

for choice_id in range(len(choice_names)):

  A_choice[choice_id, :,:,:, choice_id] = 1.0

A[1] = A_choice

##B (4 arrays because 4 state factors?)
B = utils.obj_array(num_factors)

B_context1 = np.zeros( (len(D1_names), len(D1_names), len(context_action_names))) 

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
  
  B_choice[choice_i, :, choice_i] = 1.0 # you can observe your actions without ambiguity

B[3] = B_choice

##C

from pymdp.maths import softmax

C = utils.obj_array_zeros([3, 4])
C[0][1] = 0.8 #higher preference for high reward
C[0][2] = 0.4


##D
D = utils.obj_array(num_factors)
D_context1 = np.array([0.4,0.6])
D_context2 = np.array([0.5,0.5])
D_context3 = np.array([0.5,0.5])
#high and low reward equaly likely for all decks in start.^

D[0] = D_context1
D[1] = D_context2
D[2] = D_context3

D_choice = np.zeros(len(choice_names))

D_choice[choice_names.index("Start")] = 1.0

D[3] = D_choice

############################################

my_agent = Agent(A = A, B = B, C = C, D = D)

print(my_agent.infer_policies()) 

class omgeving(object):

  def __init__(self, context = None, p_consist = 0.8):

    self.context_names = ['High','Low']

    if context == None:
      self.context = self.context_names[utils.sample(np.array(prob_win))] # randomly sample
    else:
      self.context = context

    self.p_consist = p_consist

    self.reward_obs_names = ['Null', 'High', 'Low']


  def step(self, action):

    if action == "Start":
      observed_reward = "Null"
      observed_choice   = "Start"

    elif action == "ChD1":
      observed_choice = "ChD1"
      if self.context == "High":
        observed_reward = self.reward_obs_names[utils.sample(np.array([0.0, self.p_consist, 1 - self.p_consist]))]
      else:
        observed_reward = self.reward_obs_names[utils.sample(np.array([0.0, 1 - self.p_consist, self.p_consist]))]
    
    elif action == "ChD2":
      observed_choice = "ChD2"
      if self.context == "High":
        observed_reward = self.reward_obs_names[utils.sample(np.array([0.0, self.p_consist, 1 - self.p_consist]))]
      else:
        observed_reward = self.reward_obs_names[utils.sample(np.array([0.0, 1 - self.p_consist, self.p_consist]))]
   
    elif action == "ChD3":
      observed_choice = "ChD3"
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
  HL = [obs_label[0]]
  #ent = np.zeros((T, num_factors))
  
  for t in range(T):
    print(obs)
    qs = my_agent.infer_states(obs)  # agent changes beliefs about states based on observation
    print("Beliefs about the decks reward: D1 =", qs[0],"D2 =", qs[1], 'D3 =',qs[2])

    q_pi, efe = my_agent.infer_policies() #based on beliefs agent gives value to actions
    print('EFE for each action:', efe)  #[Start, ChD1, ChD2]
    print('qpi:',q_pi)
    
    ##forced choice trials
    if t < 6:
        choice_action = 'ChD2'
    else:
        chosen_action_id = my_agent.sample_action()   #agent choses action with less negative expected free energy
        #print('chosen action id',chosen_action_id)
    
        movement_id = int(chosen_action_id[3])
        #print("movement id", movement_id)
         
        choice_action = choice_action_names[movement_id]
        print("Chosen action:", choice_action)
    
    chosendecks.append(choice_action)
    obs_label = my_env.step(choice_action)
    print("obslabel", obs_label)

    obs = [reward_obs_names.index(obs_label[0]), choice_obs_names.index(obs_label[1])]

    print(f'Action at time {t}: {choice_action}')
    print(f'Reward at time {t}: {obs_label[0]}')
    HL.append(obs_label[0][0])
    print(f'New observation:',choice_action,'&', reward_obs_names[obs[0]] + ' reward', '\n')
  return (chosendecks, HL)

p_consist = 0.8 # This is how consistent reward is with actual
env = omgeving(p_consist = p_consist)
T = 25

timepoints = [0]
for t in range(T):
    timepoints.append(t)
    
entr = run_active_inference_loop(my_agent, env, T = T)
#(entr[0] = list with chosen decks, entr[1] = list with H or L reward)

#This shows which deck the agent is chosing over time and which reward the agent gets at each timepoint
plt.scatter(timepoints, entr[0])
for i, txt in enumerate(entr[1]):
    plt.annotate(txt, (timepoints[i], entr[0][i]))    
plt.plot(timepoints, entr[0])
plt.title('Behavior of the model')
plt.ylabel('Which deck?')
plt.xlabel('Time')
plt.text(x = -1.25,y = 2.13,s = 'H = High reward' + '\n' + 'L = Low Reward')
plt.show()

#First agent is in 'Start', next there are 6 forced choices followed by free choice trials.

##my_agent.infer_states(obs) does not work -> model only choses deck 1 because all efe's are the same.