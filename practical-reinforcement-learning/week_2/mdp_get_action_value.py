
def get_action_value(mdp, state_values, state, action, gamma):
    """ Computes Q(s,a) as in formula above """
    q = 0
    state_primes = mdp.get_next_states(state, action)
    
    for state_prime, prob in state_primes.items():
        reward = mdp.get_reward(state, action, state_prime)
        q += prob * (reward + gamma * state_values[state_prime])
    
    return q
