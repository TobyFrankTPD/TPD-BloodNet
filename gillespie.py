def gillespie_SIRstep(pop, params, community_params, t):
    """Calculates one step of the SIR model. Steps are stochastic and discrete
    (no fractional populations)
    
    ARGS: 
    params: list of model-specific parameters: 
        - beta: rate of infection
        - gamma: rate of recovery
        - inf_time: time it takes for an individual to become infectious
        - sympt_time: time it takes for an infectious individual to become symptomatic
        - N: population size
    pop:    values for each population bin at the current time step
        - S: susceptible population
        - E: exposed population with the disease who aren't infectious
        - I: infectious population that isn't symptomatic
        - Sy: symptomatic population
        - R: recovered population
        - TotI: total population with the disease
    community_params: values storing information about how the population is split into communities
        - num_communities: number of semi-isolated communities
        - movement_matrix: migration matrix for population movement between communities
    t: current time step

    RETURNS
    values for each population bin at the next time step, indexed as follows:
    Susceptible, Acquired, Infectious, Symptomatic, Recovered, Total Infections (A + I + Sy)
    """
    beta, gamma, inf_time, symp_time, N = params
    num_communities, movement_matrix = community_params
    pop_t = np.zeros((num_communities+1, 8))
    S, E, I, Sy, R, TotI = pop[t-1][i+1][0:6]

    p_acquired = beta * (I+Sy) * S
    p_recovered = Sy * gamma
    if t < inf_time: # prevents indexing error
        new_infectious = 0
    else:
        new_infectious = pop[t-inf_time][i][6]
    if t < symp_time: # prevents indexing error
        new_symptomatic = 0
    else:
        new_symptomatic = pop[t-symp_time][i][7]



    probs = []

    return output

def simulate_gillespie(pop, SIR_params, community_params, tmax, step_type="stochastic"):
    """Carries out a simulation of the model with the stated parameters,
    creating pop numpy matrices for each community that store information 
    about the simulated population/communities. Uses gillespie algorithm for
    decreased computation time
    
    ARGS:
    params: list of model-specific parameters: 
        - beta: rate of infection
        - gamma: rate of recovery
        - inf_time: time it takes for an individual to become infectious
        - sympt_time: time it takes for an infectious individual to become symptomatic
        - N: population size
    pop:    values for each population bin at a given time step
        - S: susceptible population
        - E: exposed population with the disease who aren't infectious
        - I: infectious population that isn't symptomatic
        - Sy: symptomatic population
        - R: recovered population
        - TotI: total population with the disease
    community_params: values storing information about how the population is split into communities
        - num_communities: number of semi-isolated communities
        - movement_matrix: migration matrix for population movement between communities
    tmax:   number of time steps to run the simulation
    step_type: determines whether steps should be calculated deterministically or stochastically.
        - Valid inputs: "stochastic" or "deterministic"

    RETURNS:
    pop_list: updated pop numpy matrices for each community. pop_list[0] contains population aggregate
    """

    t = 0
    while t < tmax:
        next_pop = gillespie_SIRstep(pop, SIR_params, community_params, t)
        pop = pop + [] # update pop matrix w/new time/events


    return pop