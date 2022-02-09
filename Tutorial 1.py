""" Turorial 1: 'active inference from scrath'.

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

##handy plotting functions

def plot_likelihood(matrix, xlabels = list(range(9)), ylabels = list(range(9)), title_str = "Likelihood distribution (A)"):
    """
    Plots a 2-D likelihood matrix as a heatmap
    """

    if not np.isclose(matrix.sum(axis=0), 1.0).all():
      raise ValueError("Distribution not column-normalized! Please normalize (ensure matrix.sum(axis=0) == 1.0 for all columns)")
    
    fig = plt.figure(figsize = (6,6))
    ax = sns.heatmap(matrix, xticklabels = xlabels, yticklabels = ylabels, cmap = 'gray', cbar = False, vmin = 0.0, vmax = 1.0)
    plt.title(title_str)
    plt.show()

def plot_grid(grid_locations, num_x = 3, num_y = 3 ):
    """
    Plots the spatial coordinates of GridWorld as a heatmap, with each (X, Y) coordinate 
    labeled with its linear index (its `state id`)
    """

    grid_heatmap = np.zeros((num_x, num_y))
    for linear_idx, location in enumerate(grid_locations):
      y, x = location
      grid_heatmap[y, x] = linear_idx
    sns.set(font_scale=1.5)
    sns.heatmap(grid_heatmap, annot=True, cbar = False, fmt='.0f', cmap='crest')

def plot_point_on_grid(state_vector, grid_locations):
    """
    Plots the current location of the agent on the grid world
    """
    state_index = np.where(state_vector)[0][0]
    y, x = grid_locations[state_index]
    grid_heatmap = np.zeros((3,3))
    grid_heatmap[y,x] = 1.0
    sns.heatmap(grid_heatmap, cbar = False, fmt='.0f')

def plot_beliefs(belief_dist, title_str=""):
    """
    Plot a categorical distribution or belief distribution, stored in the 1-D numpy vector `belief_dist`
    """

    if not np.isclose(belief_dist.sum(), 1.0):
      raise ValueError("Distribution not normalized! Please normalize")

    plt.grid(zorder=0)
    plt.bar(range(belief_dist.shape[0]), belief_dist, color='r', zorder=3)
    plt.xticks(range(belief_dist.shape[0]))
    plt.title(title_str)
    plt.show()

## The Basics: categorical distributions

from pymdp import utils

my_categorical = np.random.rand(3)
my_categorical = utils.norm_dist(my_categorical)


print(my_categorical.reshape(-1,1))
print(f'Integral of the distribution: {round(my_categorical.sum(), 2)}')

sampled_outcome = utils.sample(my_categorical)
print(f'Sampled outcome: {sampled_outcome}')

#plot_beliefs(my_categorical, title_str = "A random (unconditional) Categorical distribution")


##  conditional categorical distributions or likelihoods

# initialize it with random numbers
p_x_given_y = np.random.rand(3, 4)
print(p_x_given_y.round(3))

# normalize it
p_x_given_y = utils.norm_dist(p_x_given_y)
print(p_x_given_y.round(3))

print(p_x_given_y[:,0].reshape(-1,1))
print(f'Integral of P(X|Y=0): {p_x_given_y[:,0].sum()}')

## So column i of the matrix p_x_given_y represents the conditional probability of X, given the i-th level of the random variable Y, i.e. P(X|Y=i)

## taking expectations of random variables using matrix-vector products

p_y = np.array([0.75, 0.25]) # this is already normalized

# the columns here are already normalized 
p_x_given_y = np.array([[0.6, 0.5],
                        [0.15, 0.41], 
                        [0.25, 0.09]])

print(p_y.round(3).reshape(-1,1))
print(p_x_given_y.round(3))

##Calculate expected value of X, given our current belief about Y, i.e. P(Y) using a simple matrix-vector product, of the form Ax.

E_x_wrt_y = p_x_given_y.dot(p_y)  #(first version of dot product)

print(E_x_wrt_y)
print(f'Integral: {E_x_wrt_y.sum().round(3)}')


### a simple environment: grid world

import itertools

""" Create  the grid locations in the form of a list of (Y, X) tuples -- HINT: use itertools """
grid_locations = list(itertools.product(range(3), repeat = 2))
print(grid_locations)

#plot_grid(grid_locations)

## Building the generative model: A, B, C, and D

## “prior beliefs” about how hidden states relate to observations

n_states = len(grid_locations)
n_observations = len(grid_locations)

print(f'Dimensionality of hidden states: {n_states}')
print(f'Dimensionality of observations: {n_observations}')

""" Create the A matrix  """

A = np.zeros( (n_states, n_observations) )

""" Create an umambiguous or 'noise-less' mapping between hidden states and observations """

np.fill_diagonal(A, 1.0)

#plot_likelihood(A, title_str = "A matrix or $P(o|s)$")

A_noisy = A.copy()

A_noisy[0,0] = 1/3.0

A_noisy[1,0] = 1/3.0

A_noisy[3,0] = 1 /3.0

#plot_likelihood(A_noisy, title_str = 'modified A matrix where location (0,0) is "blurry"')

""" Let's make ake one grid location "ambiguous" in the sense that it could be easily confused with neighbouring locations """
my_A_noisy = A_noisy.copy()

# locations 3 and 7 are the nearest neighbours to location 6
my_A_noisy[3,6] = 1.0 / 3.0
my_A_noisy[6,6] = 1.0 / 3.0
my_A_noisy[7,6] = 1.0 / 3.0

#plot_likelihood(my_A_noisy, title_str = "Noisy A matrix now with TWO ambiguous locations")

##B-matrix, generative model's prior beliefs about (contrllable) transitions between hidden states over time

actions = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]

def create_B_matrix():
  B = np.zeros( (len(grid_locations), len(grid_locations), len(actions)) )

  for action_id, action_label in enumerate(actions):

    for curr_state, grid_location in enumerate(grid_locations):

      y, x = grid_location

      if action_label == "UP":
        next_y = y - 1 if y > 0 else y 
        next_x = x
      elif action_label == "DOWN":
        next_y = y + 1 if y < 2 else y 
        next_x = x
      elif action_label == "LEFT":
        next_x = x - 1 if x > 0 else x 
        next_y = y
      elif action_label == "RIGHT":
        next_x = x + 1 if x < 2 else x 
        next_y = y
      elif action_label == "STAY":
        next_x = x
        next_y = y
      new_location = (next_y, next_x)
      next_state = grid_locations.index(new_location)
      B[next_state, curr_state, action_id] = 1.0
  return B

B = create_B_matrix()

##explore what it looks to take an action, using matrix-vector product of an action-conditioned “slice” of the B array and a previous state vector

""" Define a starting location""" 
starting_location = (1,0)

"""get the linear index of the state"""
state_index = grid_locations.index(starting_location)

"""  and create a state vector out of it """
starting_state = utils.onehot(state_index, n_states)

#plot_point_on_grid(starting_state, grid_locations)

#?
#plot_beliefs(starting_state, "Categorical distribution over the starting state")

##Now let’s imagine we’re moving “RIGHT” - write the conditional expectation, that will create the state vector corresponding to the new state after taking a step to the right
""" Redefine the action here, just for reference """
actions = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]

""" Generate the next state vector, given the starting state and the B matrix"""
right_action_idx = actions.index("RIGHT") 
next_state = B[:,:, right_action_idx].dot(starting_state) # input the indices to the B matrix

""" Plot the next state, after taking the action """
#plot_point_on_grid(next_state, grid_locations)

#Now let’s imagine we’re moving “DOWN” from where we just landed - write the conditional expectation, that will create the state vector corresponding to the new state after taking a step down.
""" Generate the next state vector, given the previous state and the B matrix"""
prev_state = next_state.copy()
down_action_index = actions.index("DOWN")
next_state = B[:,:,down_action_index].dot(prev_state)

"""  Plot the new state vector, after making the movement """
#plot_point_on_grid(next_state, grid_locations)

###3. the prior over observaions: c vector & d vector

##THe (biased) generative model’s prior preference for particular observations, encoded in terms of probabilities.

""" Create an empty vector to store the preferences over observations """
C = np.zeros(n_observations)

""" Choose an observation index to be the 'desired' rewarding index, and fill out the C vector accordingly """
desired_location = (2,2) # choose a desired location
desired_location_index = grid_locations.index(desired_location) # get the linear index of the grid location, in terms of 0 through 8

C[desired_location_index] = 1.0 # set the preference for that location to be 100%, i.e. 1.0

"""  Let's look at the prior preference distribution """
#plot_beliefs(C, title_str = "Preferences over observations")

##THe generative model’s prior belief over hidden states at the first timestep.

""" Create a D vector, basically a belief that the agent has about its own starting location """

# create a one-hot / certain belief about initial state
D = utils.onehot(0, n_states)

# demonstrate hwo belief about initial state can also be uncertain / spread among different possible initial states
# alternative, where you have a degenerate/noisy prior belief
# D = utils.norm_dist(np.ones(n_states))

""" Let's look at the prior over hidden states """
#plot_beliefs(D, title_str = "Prior beliefs over states")


###Hidden state inferene, 
"""Hidden state inference proceeds by finding the 
setting of the optimal variational posterior q(st)
 that minimizes the variational free energy. """
 
from pymdp.maths import softmax
from pymdp.maths import spm_log_single as log_stable

""" Create an infer states function that implements the math we just discussed"""

def infer_states(observation_index, A, prior):

  """ Implement inference here -- NOTE: prior is already passed in, so you don't need to do anything with the B matrix. """
  """ This function has already been given P(s_t). The conditional expectation that creates "today's prior", using "yesterday's posterior", will happen *before calling* this function"""
  
  log_likelihood = log_stable(A[observation_index,:])

  log_prior = log_stable(prior)

  qs = softmax(log_likelihood + log_prior)
   
  return qs

#simulated single timestep of hidden state inference

qs_past = utils.onehot(4, n_states) # agent believes they were at location 4 -- i.e. (1,1) one timestep ago

last_action = "UP" # the agent knew it moved "UP" one timestep ago
action_id = actions.index(last_action) # get the action index for moving "UP"

##Get “today’s prior” using the past posterior and the past action, i.e. calculate: P(st)=Eq(st−1)[P(st|st−1,ut−1)]
##and choose an observation that is consistent with the new location

prior = B[:,:,action_id].dot(qs_past)

observation_index = 1

##Now run the infer_states() function, using the observation, the A matrix, and the prior we just calculated above

qs_new = infer_states(observation_index, A, prior)
print(qs_new)
#plot_beliefs(qs_new, title_str = "Beliefs about hidden states")

""" Get an observation that 'conflicts' with the prior """ #observation and prior disagree
observation_index = 2 # this is like the agent is seeing itself in location (0, 2)
qs_new = infer_states(observation_index, A, prior)
print(qs_new)
#plot_beliefs(qs_new)

""" Create an ambiguous A matrix """
A_partially_ambiguous = softmax(A)
print(A_partially_ambiguous.round(3))

""" ... and a noisy prior """
noisy_prior = softmax(prior) 
#plot_beliefs(noisy_prior)

###action seletion and expected free energy

##Nu gaan we functies schrijven om de verwachte toestanden te berekenenQ(st+1|ut), verwachte waarnemingenQ(ot+1|ut), de entropie vanP(o|s):H[A], en de KL-divergentie tussen de verwachte waarnemingen en de eerdere voorkeuren 

""" define component functions for computing expected free energy """

def get_expected_states(B, qs_current, action):
  """ Compute the expected states one step into the future, given a particular action """
  qs_u = B[:,:,action].dot(qs_current)

  return qs_u

def get_expected_observations(A, qs_u):
  """ Compute the expected observations one step into the future, given a particular action """

  qo_u = A.dot(qs_u)

  return qo_u

def entropy(A):
  """ Compute the entropy of a set of conditional distributions, i.e. one entropy value per column """

  H_A = - (A * log_stable(A)).sum(axis=0)

  return H_A

def kl_divergence(qo_u, C):
  """ Compute the Kullback-Leibler divergence between two 1-D categorical distributions"""
  
  return (log_stable(qo_u) - log_stable(C)).dot(qo_u)

##Laten we ons nu voorstellen dat we ons in een begintoestand bevinden, zoals (1,1). NB Dit is het generatieve proces waar we het over hebben - dwz de ware toestand van de wereld

""" Get state index, create state vector for (1,1) """

state_idx = grid_locations.index((1,1))
state_vector = utils.onehot(state_idx, n_states)
#plot_point_on_grid(state_vector, grid_locations)

##En laten we verder aannemen dat we beginnen met de (juiste) overtuiging over onze locatie, dat we op locatie zijn (1,1). Dus laten we gewoon onze huidige makenQ(st)gelijk aan de ware toestandsvector. Je zou dit kunnen zien alsof we slechts één 'stap' van gevolgtrekking hebben gemaakt met behulp van onze huidige observatie samen met precieze A/ Bmatrices.
""" Make qs_current identical to the true starting state """ 
qs_current = state_vector.copy()
#plot_beliefs(qs_current, title_str ="Where do we believe we are?")

## we willen naar (1,2)
""" Create a preference to be in (1,2) """

desired_idx = grid_locations.index((1,2))   # = index 5 in grid

C = utils.onehot(desired_idx, n_observations)

#plot_beliefs(C, title_str = "Preferences")

##evalueren van expected energy of actions go L and R (we houden het simpel nu)

left_idx = actions.index("LEFT")
right_idx = actions.index("RIGHT")

print(f'Action index of moving left: {left_idx}')
print(f'Action index of moving right: {right_idx}')

##En laten we nu de functies gebruiken die we zojuist hebben gedefinieerd, in combinatie met onze A, B, Carrays en onze huidige posterieure qs_current, om de verwachte vrije energieën voor de twee acties te berekenen

""" Compute the expected free energies for moving left vs. moving right """
G = np.zeros(2) # store the expected free energies for each action in here

"""
Compute G for MOVE LEFT here 
"""

qs_u_left = get_expected_states(B, qs_current, left_idx)
# alternative
# qs_u_left = B[:,:,left_idx].dot(qs_current)

H_A = entropy(A)
qo_u_left = get_expected_observations(A, qs_u_left)
# alternative
# qo_u_left = A.dot(qs_u_left)

predicted_uncertainty_left = H_A.dot(qs_u_left)
predicted_divergence_left = kl_divergence(qo_u_left, C)
G[0] = predicted_uncertainty_left + predicted_divergence_left

"""
Compute G for MOVE RIGHT here 
"""

qs_u_right = get_expected_states(B, qs_current, right_idx)
# alternative
# qs_u_right = B[:,:,right_idx].dot(qs_current)

H_A = entropy(A)
qo_u_right = get_expected_observations(A, qs_u_right)
# alternative
# qo_u_right = A.dot(qs_u_right)

predicted_uncertainty_right = H_A.dot(qs_u_right)
predicted_divergence_right = kl_divergence(qo_u_right, C)
G[1] = predicted_uncertainty_right + predicted_divergence_right


""" Now let's print the expected free energies for the two actions, that we just calculated """
print(f'Expected free energy of moving left: {G[0]}\n')
print(f'Expected free energy of moving right: {G[1]}\n')

##Laten we nu de formule gebruiken voor de posterior over acties, dat wil zeggen: Q(ut)=σ(−G) om de kansen van elke actie te berekenen

Q_u = softmax(-G)

""" and print the probability of each action """
print(f'Probability of moving left: {Q_u[0]}')
print(f'Probability of moving right: {Q_u[1]}')

##voor later nut, de verwachte vrije energieberekeningen in een functie verpakken

def calculate_G(A, B, C, qs_current, actions):

  G = np.zeros(len(actions)) # vector of expected free energies, one per action

  H_A = entropy(A) # entropy of the observation model, P(o|s)

  for action_i in range(len(actions)):
    
    qs_u = get_expected_states(B, qs_current, action_i) # expected states, under the action we're currently looping over
    qo_u = get_expected_observations(A, qs_u)           # expected observations, under the action we're currently looping over

    pred_uncertainty = H_A.dot(qs_u) # predicted uncertainty, i.e. expected entropy of the A matrix
    pred_div = kl_divergence(qo_u, C) # predicted divergence

    G[action_i] = pred_uncertainty + pred_div # sum them together to get expected free energy
  
  return G

"""Compleet recept voor actieve inferentie

1. Voorbeeld van een observatie Ot van de huidige toestand van het milieu

2. Voer gevolgtrekkingen uit over verborgen toestanden, dwz optimaliseren q(s) door minimalisering van vrije energie

3. Bereken verwachte vrije energie van acties G

4. Voorbeeldactie van de posterieure over acties Q(ut)∼σ(−G).

5. Gebruik de voorbeeldactie At om het generatieve proces te verstoren en terug te gaan naar stap 1."""

#Laten we beginnen met het maken van een klasse die de Grid World-omgeving (dwz het generatieve proces ) vertegenwoordigt waarin onze actieve inferentieagent zal navigeren. Merk op dat we het generatieve proces niet hoeven te specificeren in termen van Aen Bmatrices - het generatieve proces zal net zo willekeurig en complex zijn als de omgeving. De Aen B-matrices zijn slechts de representatie van de wereld en de taak door de agent, die in dit geval de Markoviaanse, geruisloze dynamiek van de wereld perfect weergeven.

class GridWorldEnv():
    
    def __init__(self,starting_state = (0,0)):

        self.init_state = starting_state
        self.current_state = self.init_state
        print(f'Starting state is {starting_state}')
    
    def step(self,action_label):

        (Y, X) = self.current_state

        if action_label == "UP": 
          
          Y_new = Y - 1 if Y > 0 else Y
          X_new = X

        elif action_label == "DOWN": 

          Y_new = Y + 1 if Y < 2 else Y
          X_new = X

        elif action_label == "LEFT": 
          Y_new = Y
          X_new = X - 1 if X > 0 else X

        elif action_label == "RIGHT": 
          Y_new = Y
          X_new = X +1 if X < 2 else X

        elif action_label == "STAY":
          Y_new, X_new = Y, X 
        
        self.current_state = (Y_new, X_new) # store the new grid location

        obs = self.current_state # agent always directly observes the grid location they're in 

        return obs

    def reset(self):
        self.current_state = self.init_state
        print(f'Re-initialized location to {self.init_state}')
        obs = self.current_state
        print(f'..and sampled observation {obs}')

        return obs
    
env = GridWorldEnv()

##Nu uitgerust met een omgeving en een generatief model (our A, B, C, en D), kunnen we de volledige actieve inferentielus coderen

""" Fill out the components of the generative model """

A = np.eye(n_observations, n_states)

B = create_B_matrix()

C = utils.onehot(grid_locations.index( (2, 2) ), n_observations) # make the agent prefer location (2,2) (lower right corner of grid world)

D = utils.onehot(grid_locations.index( (1,2) ), n_states) # start the agent with the prior belief that it starts in location (1,2) 

actions = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]

##Laten we nu de omgeving initialiseren met startstatus (1,2), zodat de agent nauwkeurige overtuigingen heeft over waar het begint.

env = GridWorldEnv(starting_state = (1,2))

##Laten we een functie schrijven die de hele actieve inferentielus uitvoert en deze vervolgens uitvoeren voor tijdstappen.T = 5

""" Write a function that, when called, runs the entire active inference loop for a desired number of timesteps"""

def run_active_inference_loop(A, B, C, D, actions, env, T = 5):

  """ Initialize the prior that will be passed in during inference to be the same as `D` """
  prior = D.copy() # initial prior should be the D vector

  """ Initialize the observation that will be passed in during inference - hint use env.reset()"""
  obs = env.reset() # initialize the `obs` variable to be the first observation you sample from the environment, before `step`-ing it.

  for t in range(T):

    print(f'Time {t}: Agent observes itself in location: {obs}')

    # convert the observation into the agent's observational state space (in terms of 0 through 8)
    obs_idx = grid_locations.index(obs)

    # perform inference over hidden states
    qs_current = infer_states(obs_idx, A, prior)

    #plot_beliefs(qs_current, title_str = f"Beliefs about location at time {t}")
    print(f"beliefs about location at time {t}", qs_current)

    # calculate expected free energy of actions
    G = calculate_G(A, B, C, qs_current, actions)
    
    # compute action posterior
    Q_u = softmax(-G)

    # sample action from probability distribution over actions
    chosen_action = utils.sample(Q_u)

    # compute prior for next timestep of inference
    prior = B[:,:,chosen_action].dot(qs_current) 

    # update generative process
    action_label = actions[chosen_action]

    obs = env.step(action_label)
  
  return qs_current

""" Run the function we just wrote, for T = 5 timesteps """
qs = run_active_inference_loop(A, B, C, D, actions, env, T = 5)

##(....) Laten we nu actieve gevolgtrekking maken met meerstappenbeleid

##We kunnen vertrouwen op een handige functie van pymdpde controlmodule die wordt aangeroepen construct_policies()om automatisch een lijst te genereren van alle beleidsregels die we willen onderhouden, voor een bepaald aantal controlestatussen (acties) en een gewenste tijdshorizon.

from pymdp.control import construct_policies

policy_len = 4
n_actions = len(actions)

# we have to wrap `n_states` and `n_actions` in a list for reasons that will become clear in Part II
all_policies = construct_policies([n_states], [n_actions], policy_len = policy_len)

print(f'Total number of policies for {n_actions} possible actions and a planning horizon of {policy_len}: {len(all_policies)}')

##Laten we onze verwachte vrije-energiefunctie herschrijven, maar nu herhalen we beleid (reeksen van acties), in plaats van acties (1-staps beleid)

def calculate_G_policies(A, B, C, qs_current, policies):

  G = np.zeros(len(policies)) # initialize the vector of expected free energies, one per policy
  H_A = entropy(A)            # can calculate the entropy of the A matrix beforehand, since it'll be the same for all policies

  for policy_id, policy in enumerate(policies): # loop over policies - policy_id will be the linear index of the policy (0, 1, 2, ...) and `policy` will be a column vector where `policy[t,0]` indexes the action entailed by that policy at time `t`

    t_horizon = policy.shape[0] # temporal depth of the policy

    G_pi = 0.0 # initialize expected free energy for this policy

    for t in range(t_horizon): # loop over temporal depth of the policy

      action = policy[t,0] # action entailed by this particular policy, at time `t`

      # get the past predictive posterior - which is either your current posterior at the current time (not the policy time) or the predictive posterior entailed by this policy, one timstep ago (in policy time)
      if t == 0:
        qs_prev = qs_current 
      else:
        qs_prev = qs_pi_t
        
      qs_pi_t = get_expected_states(B, qs_prev, action) # expected states, under the action entailed by the policy at this particular time
      qo_pi_t = get_expected_observations(A, qs_pi_t)   # expected observations, under the action entailed by the policy at this particular time

      kld = kl_divergence(qo_pi_t, C) # Kullback-Leibler divergence between expected observations and the prior preferences C

      G_pi_t = H_A.dot(qs_pi_t) + kld # predicted uncertainty + predicted divergence, for this policy & timepoint

      G_pi += G_pi_t # accumulate the expected free energy for each timepoint into the overall EFE for the policy

    G[policy_id] += G_pi
  
  return G

#Laten we een functie schrijven voor het berekenen van de actie posterieur, gegeven de posterieure waarschijnlijkheid van elk beleid:
    
def compute_prob_actions(actions, policies, Q_pi):
  P_u = np.zeros(len(actions)) # initialize the vector of probabilities of each action

  for policy_id, policy in enumerate(policies):
    P_u[int(policy[0,0])] += Q_pi[policy_id] # get the marginal probability for the given action, entailed by this policy at the first timestep
  
  P_u = utils.norm_dist(P_u) # normalize the action probabilities
  
  return P_u

##Nu kunnen we een nieuwe actieve inferentiefunctie schrijven, die gebruik maakt van temporeel-diepe planning

def active_inference_with_planning(A, B, C, D, n_actions, env, policy_len = 2, T = 5):

  """ Initialize prior, first observation, and policies """
  
  prior = D # initial prior should be the D vector

  obs = env.reset() # get the initial observation

  policies = construct_policies([n_states], [n_actions], policy_len = policy_len)

  for t in range(T):

    print(f'Time {t}: Agent observes itself in location: {obs}')

    # convert the observation into the agent's observational state space (in terms of 0 through 8)
    obs_idx = grid_locations.index(obs)

    # perform inference over hidden states
    qs_current = infer_states(obs_idx, A, prior)
    #plot_beliefs(qs_current, title_str = f"Beliefs about location at time {t}")
    print(f"beliefs about location at time {t}", qs_current)

    # calculate expected free energy of actions
    G = calculate_G_policies(A, B, C, qs_current, policies)

    # to get action posterior, we marginalize P(u|pi) with the probabilities of each policy Q(pi), given by \sigma(-G)
    Q_pi = softmax(-G)

    # compute the probability of each action
    P_u = compute_prob_actions(actions, policies, Q_pi)

    # sample action from probability distribution over actions
    chosen_action = utils.sample(P_u)

    # compute prior for next timestep of inference
    prior = B[:,:,chosen_action].dot(qs_current) 

    # step the generative process and get new observation
    action_label = actions[chosen_action]
    obs = env.step(action_label)
  
  return qs_current

##Voer het nu uit om te zien of de agent naar de locatie kan navigeren(2,2)

D = utils.onehot(grid_locations.index((0,0)), n_states) # let's have the agent believe it starts in location (0,0) (upper left corner) 
env = GridWorldEnv(starting_state = (0,0))
qs_final = active_inference_with_planning(A, B, C, D, n_actions, env, policy_len = 3, T = 10)


