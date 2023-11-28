# From The School of AI's Move 37 Course https://www.theschool.ai/courses/move-37-course/
# Coding demo by Colin Skow
# Forked from https://github.com/lazyprogrammer/machine_learning_examples/tree/master/rl
# Credit goes to LazyProgrammer
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
from grid_world import standard_grid
from utils import print_values, print_policy

# SMALL_ENOUGH is referred to by the mathematical symbol theta in equations
SMALL_ENOUGH = 1e-3
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

def calc_optimal_state_values():
  '''
  Iteratively compute the state values until the change in state values between iterations is below a certain threshold.
  '''
  return 

def calc_state_values(grid):
  '''
  Determine the value of each state via the bellman equation for state values. 
  
  '''
  ## calc_state_values: calculate max value for each state
  # save old grid 
  # iterate through each state 
    ## calc_max_value: given a state, calculate value for all possible actions from that state and select action that results in in max value. Save action and max value.
    # update correct index in grid with state_value
    # compare old_grid with grid, check to see if max(difference) < threshold

  return 

def calc_max_value(grid, values, state):
  '''

    Returns:
      action
      value 
  '''
  print('state = ', state)
  ## calc_max_value: given a state, calculate value for all possible actions from that state and select action that results in in max value. Save action and max value.
  grid.set_state(state)
  if grid.is_terminal(state): # if state == terminal state 
    return None, grid.rewards[state] # return appropriate reward 
  else:
    
    # initialize best action and best value
    best_a = None
    best_v = float('-inf')
    
    # iterate through each action 
    for action in grid.actions[state]:
      
      # get list of tuples (p, r, s`) corresponding to s, a
      transitions = grid.get_transition_probs(action) # returns a list of (probability, reward, s') transition tuples
      
      # for each possible next state, calculate expected value 
      v = 0
      for (prob, r, state_prime) in transitions:
        v += prob * (r + GAMMA * values[state_prime])
      
      # return best action corresponding to highest value
      if v > best_v:
        best_v = v
        best_a = action

    return best_a, best_v

if __name__ == '__main__':
  # this grid gives you a reward of -0.1 for every non-terminal state
  # we want to see if this will encourage finding a shorter path to the goal
  grid = standard_grid(obey_prob=0.5, step_cost=-0.5)

  # print rewards
  print("rewards:")
  print_values(grid.rewards, grid)

  # calculate accurate values for each square
  states = grid.all_states()
  print(states)
  test_state = (0,1)
  # state_value = calc_max_value(grid, test_state)
  # print(test_state, state_value)

  quit()



def best_action_value(grid, V, s):
  # finds the highest value action (max_a) from state s, returns the action and value
  best_a = None
  best_value = float('-inf')
  grid.set_state(s)
  # loop through all possible actions to find the best current action
  for a in ALL_POSSIBLE_ACTIONS:
    transititions = grid.get_transition_probs(a)
    expected_v = 0
    expected_r = 0
    for (prob, r, state_prime) in transititions:
      expected_r += prob * r
      expected_v += prob * V[state_prime]
    v = expected_r + GAMMA * expected_v
    if v > best_value:
      best_value = v
      best_a = a
  return best_a, best_value

def calculate_values(grid):
  # initialize V(s)
  V = {}
  states = grid.all_states()
  for s in states:
    V[s] = 0
  # repeat until convergence
  # V[s] = max[a]{ sum[s',r] { p(s',r|s,a)[r + gamma*V[s']] } }
  while True:
    # biggest_change is referred to by the mathematical symbol delta in equations
    biggest_change = 0
    for s in grid.non_terminal_states():
      old_v = V[s]
      _, new_v = best_action_value(grid, V, s)
      V[s] = new_v
      biggest_change = max(biggest_change, np.abs(old_v - new_v))

    if biggest_change < SMALL_ENOUGH:
      break
  return V

def initialize_random_policy():
  # policy is a lookup table for state -> action
  # we'll randomly choose an action and update as we learn
  policy = {}
  for s in grid.non_terminal_states():
    policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)
  return policy

def calculate_greedy_policy(grid, V):
  policy = initialize_random_policy()
  # find a policy that leads to optimal value function
  for s in policy.keys():
    grid.set_state(s)
    # loop through all possible actions to find the best current action
    best_a, _ = best_action_value(grid, V, s)
    policy[s] = best_a
  return policy


if __name__ == '__main__':
  # this grid gives you a reward of -0.1 for every non-terminal state
  # we want to see if this will encourage finding a shorter path to the goal
  grid = standard_grid(obey_prob=0.5, step_cost=-2.0)

  # print rewards
  print("rewards:")
  print_values(grid.rewards, grid)

  # calculate accurate values for each square
  V = calculate_values(grid)

  # calculate the optimum policy based on our values
  policy = calculate_greedy_policy(grid, V)

  # our goal here is to verify that we get the same answer as with policy iteration
  print("values:")
  print_values(V, grid)
  print("policy:")
  print_policy(policy, grid)
