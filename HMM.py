import numpy as np
import operator

def HMM_encoder():

    b_prior_probs = {
        'B1': 0.333,
        'B2': 0.000,
        'B3': 0.000,
        'Bend': 0.000,
    }
    b_transition_probs = {
        'B1': {'B1': 0.625, 'B2': 0.375, 'B3': 0.000, 'Bend': 0.000},
        'B2': {'B1': 0.000, 'B2': 0.625, 'B3': 0.375, 'Bend': 0.000},
        'B3': {'B1': 0.000, 'B2': 0.000, 'B3': 0.625, 'Bend': 0.375},
        'Bend': {'B1': 0.000, 'B2': 0.000, 'B3': 0.000, 'Bend': 1.000},
    }
    # Parameters for end state is not required
    b_emission_paras = {
        'B1': (41.750, 2.773),
        'B2': (58.625, 5.678),
        'B3': (53.125, 5.418),
        'Bend': (None, None)
    }

    """Word CAR"""
    c_prior_probs = {
        'C1': 0.333,
        'C2': 0.000,
        'C3': 0.000,
        'Cend': 0.000,
    }
    c_transition_probs = {
        'C1': {'C1': 0.667, 'C2': 0.333, 'C3': 0.000, 'Cend': 0.000},
        'C2': {'C1': 0.000, 'C2': 0.000, 'C3': 1.000, 'Cend': 0.00},
        'C3': {'C1': 0.000, 'C2': 0.000, 'C3': 0.800, 'Cend': 0.200},
        'Cend': {'C1': 0.000, 'C2': 0.000, 'C3': 0.000, 'Cend': 1.000},
    }
    # Parameters for end state is not required
    c_emission_paras = {
        'C1': (35.667, 4.899),
        'C2': (43.667, 1.700),
        'C3': (44.200, 7.341),
        'Cend': (None, None)
    }

    """Word HOUSE"""
    h_prior_probs = {
        'H1': 0.333,
        'H2': 0.000,
        'H3': 0.000,
        'Hend': 0.000,
    }
    # Probability of a state changing to another state.
    h_transition_probs = {
        'H1': {'H1': 0.667, 'H2': 0.333, 'H3': 0.000, 'Hend': 0.000},
        'H2': {'H1': 0.000, 'H2': 0.857, 'H3': 0.143, 'Hend': 0.000},
        'H3': {'H1': 0.000, 'H2': 0.000, 'H3': 0.812, 'Hend': 0.188},
        'Hend': {'H1': 0.000, 'H2': 0.000, 'H3': 0.000, 'Hend': 1.000},
    }
    # Parameters for end state is not required
    h_emission_paras = {
        'H1': (45.333, 3.972),
        'H2': (34.952, 8.127),
        'H3': (67.438, 5.733),
        'Hend': (None, None)
    }

    return (b_prior_probs, b_transition_probs, b_emission_paras,
            c_prior_probs, c_transition_probs, c_emission_paras,
            h_prior_probs, h_transition_probs, h_emission_paras,)


def gaussian_prob(x, para_tuple):

    if list(para_tuple) == [None, None]:
        return 0.0

    mean, std = para_tuple
    gaussian_percentile = (2 * np.pi * std**2)**-0.5 * \
                          np.exp(-(x - mean)**2 / (2 * std**2))
    return gaussian_percentile


def viterbi(evidence_vector, states, prior_probs,
            transition_probs, emission_paras):

    sequence = []
    probability = 0.0

    dict_1 = {}
    dict_2 = {}
    path = {}

    if len(evidence_vector) == 0:
        return sequence, probability

    for state in states:
        dict_1[state] = gaussian_prob(evidence_vector[0], emission_paras[state])\
                        * prior_probs[state]
        path[state] = [state]
        if dict_1[state] > probability:
            probability = dict_1[state]
            sequence = path[state]

    if len(evidence_vector)== 1:
        return sequence, probability

    updated_path = {}
    for state in states:
        #print('flag')
        prob_ls = {}
        for pre_state in (transition_probs[state].keys()):
            probability = gaussian_prob(evidence_vector[1], emission_paras[state])\
                          * dict_1[pre_state] * transition_probs[pre_state][state]
            prob_ls[probability] =  pre_state
        probability = max(prob_ls)
        dict_2[state] = probability
        updated_path[state] = path[prob_ls[probability]] + [state]
    path = updated_path


    for i in range(len(evidence_vector)):
        if i == 0 or i ==1:
            continue
        updated_path = {}
        for state in states:
            prob_ls = {}
            for pre_state in (transition_probs[state].keys()):
                probability = gaussian_prob(evidence_vector[i], emission_paras[state])\
                              * dict_2[pre_state] * transition_probs[pre_state][state]
                prob_ls[probability] =  pre_state
            probability = max(prob_ls)
            dict_1[state] = probability
            updated_path[state] = path[prob_ls[probability]] + [state]
        path = updated_path
        dict_2 = dict_1
        dict_1 = {}

    probability = 0
    for state in dict_2:
        if dict_2[state] > probability:
            probability = dict_2[state]
            sequence = path[state]

    return sequence, probability


def prob_HMM():
    b_prior_probs = {
        'B1': 0.333,
        'B2': 0.000,
        'B3': 0.000,
        'Bend': 0.000,
    }
    # example: {'B1': {'B1' : (right-hand Y, left-hand Y), ... }
    b_transition_probs = {
        'B1': {'B1': (0.625, 0.700), 'B2': (0.375, 0.300), 'B3': (0.000, 0.), 'Bend': (0., 0.)},
        'B2': {'B1': (0., 0.), 'B2': (0.625, 0.05), 'B3': (0.375, 0.95), 'Bend': (0., 0.)},
        'B3': {'B1': (0., 0.), 'B2': (0., 0.), 'B3': (0.625, 0.727), 'Bend': (0.125, 0.091), 'C1': (0.125, 0.091), 'H1': (0.125, 0.091)},
        'Bend': {'B1': (0., 0.), 'B2': (0., 0.), 'B3': (0., 0.), 'Bend': (1., 1.)},
    }
    # example: {'B1': [(right-mean, right-std), (left-mean, left-std)] ...}
    b_emission_paras = {
        'B1': [(41.750, 2.773), (108.200, 17.314)],
        'B2': [(58.625, 5.678), (78.670, 1.886)],
        'B3': [(53.125, 5.418), (64.182, 5.573)],
        'Bend': [(None, None), (None, None)]
    }

    """Word Car"""
    c_prior_probs = {
        'C1': 0.333,
        'C2': 0.000,
        'C3': 0.000,
        'Cend': 0.000,
    }
    c_transition_probs = {
        'C1': {'C1': (0.667, 0.700), 'C2': (0.333, 0.300), 'C3': (0.000, 0.000), 'Cend': (0.000, 0.000)},
        'C2': {'C1': (0.000, 0.000), 'C2': (0.000, 0.625), 'C3': (1.00, 0.375), 'Cend': (0.000, 0.000)},
        'C3': {'C1': (0.000, 0.000), 'C2': (0.000, 0.000), 'C3': (0.800, 0.625), 'Cend': (0.067, 0.125), 'B1': (0.067, 0.125), 'H1': (0.067, 0.125)},
        'Cend': {'C1': (0.000, 0.000), 'C2': (0.000, 0.000), 'C3': (0.000, 0.000), 'Cend': (1.000, 1.000)},
    }
    c_emission_paras = {
        'C1': [(35.667, 4.899), (56.3, 10.659)],
        'C2': [(43.667, 1.700), (37.11, 4.306)],
        'C3': [(44.200, 7.341), (50.000, 7.826)],
        'Cend': [(None, None), (None, None)]
    }

    """Word HOUSE"""
    h_prior_probs = {
        'H1': 0.333,
        'H2': 0.000,
        'H3': 0.000,
        'Hend': 0.000,
    }
    h_transition_probs = {
        'H1': {'H1': (0.667, 0.700), 'H2': (0.333, 0.300), 'H3': (0.000, 0.000), 'Hend': (0.000, 0.000)},
        'H2': {'H1': (0.000, 0.000), 'H2': (0.857, 0.842), 'H3': (0.143, 0.158), 'Hend': (0.000, 0.000)},
        'H3': {'H1': (0.000, 0.000), 'H2': (0.000, 0.000), 'H3': (0.812, 0.824), 'Hend': ((0.063, 0.059)), 'B1': (0.063, 0.059), 'C1': (0.063, 0.059)},
        'Hend': {'H1': (0.000, 0.000), 'H2': (0.000, 0.000), 'H3': (0.000, 0.000), 'Hend': (1.000, 1.000)},
    }
    h_emission_paras = {
        'H1': [(45.333, 3.972), (53.600, 7.392)],
        'H2': [(34.952, 8.127), (37.168, 8.875)],
        'H3': [(67.438, 5.733), (74.176, 8.347)],
        'Hend': [(None, None), (None, None)]
    }

    return (b_prior_probs, b_transition_probs, b_emission_paras,
            c_prior_probs, c_transition_probs, c_emission_paras,
            h_prior_probs, h_transition_probs, h_emission_paras,)

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def multidimensional_viterbi(evidence_vector, states, prior_probs,
                             transition_probs, emission_paras):
    """Decode the most likely word phrases generated by the evidence vector.
    States, prior_probs, transition_probs, and emission_probs will now contain
    all the words from part_2_a.
    Evidence vector is a list of tuples where the first element of each tuple is the right
    hand coordinate and the second element is the left hand coordinate.
    """
    # TODO: complete this function.
    #raise NotImplementedError()

    sequence = []
    probability = 0.0

    dict_1 = {}
    dict_2 = {}
    path = {}
    updated_path = {}

    if len(evidence_vector) == 0:
        return sequence, probability

    for state in states:
        dict_1[state] = gaussian_prob(evidence_vector[0][0], emission_paras[state][0]) \
                 * gaussian_prob(evidence_vector[0][1], emission_paras[state][1]) \
                 * prior_probs[state]
        path[state] = [state]
        if dict_1[state] > probability:
            probability = dict_1[state]
            sequence = path[state]

    if len(evidence_vector)== 1:
        return sequence, probability

    for state in states:
        prob_ls = {}
        if state[0] == 'B':
            pre_states = ['B1', 'B2', 'B3', 'Bend']
        elif state[0] == 'C':
            pre_states = ['C1', 'C2', 'C3', 'Cend']
        elif state[0] == 'H':
            pre_states = ['H1', 'H2', 'H3', 'Hend']

        if state == 'B1':
            pre_states = ['B1', 'B2', 'B3', 'Bend', 'C3', 'H3']
        elif state == 'C1':
            pre_states = ['C1', 'C2', 'C3', 'Cend', 'B3', 'H3']
        elif state == 'H1':
            pre_states = ['H1', 'H2', 'H3', 'Hend', 'B3', 'C3']

        for pre_state in pre_states:
            probability = transition_probs[pre_state][state][0]\
                 * gaussian_prob(evidence_vector[1][0], emission_paras[state][0])\
                 * transition_probs[pre_state][state][1] \
                 * gaussian_prob(evidence_vector[1][1], emission_paras[state][1])\
                 * dict_1[pre_state]
            prob_ls[probability] =  pre_state
        probability = max(prob_ls)
        dict_2[state] = probability
        updated_path[state] = path[prob_ls[probability]] + [state]
    path = updated_path


    for i in range(len(evidence_vector)):
        if i == 0 or i == 1:
            continue
        updated_path = {}
        for state in states:
            prob_ls = {}
            if state[0] == 'B':
                pre_states = ['B1', 'B2', 'B3', 'Bend']
            elif state[0] == 'C':
                pre_states = ['C1', 'C2', 'C3', 'Cend']
            elif state[0] == 'H':
                pre_states = ['H1', 'H2', 'H3', 'Hend']

            if state == 'B1':
                pre_states = ['B1', 'B2', 'B3', 'Bend', 'C3', 'H3']
            elif state == 'C1':
                pre_states = ['C1', 'C2', 'C3', 'Cend', 'B3', 'H3']
            elif state == 'H1':
                pre_states = ['H1', 'H2', 'H3', 'Hend', 'B3', 'C3']

            for pre_state in pre_states:
                probability = transition_probs[pre_state][state][0]\
                     * gaussian_prob(evidence_vector[i][0], emission_paras[state][0])\
                     * transition_probs[pre_state][state][1] \
                     * gaussian_prob(evidence_vector[i][1], emission_paras[state][1])\
                     * dict_2[pre_state]
                prob_ls[probability] =  pre_state
            probability = max(prob_ls)
            dict_1[state] = probability
            updated_path[state] = path[prob_ls[probability]] + [state]
        path = updated_path
        dict_2 = dict_1
        dict_1 = {}

    probability = 0
    for state in dict_2:
        if dict_2[state] > probability:
            probability = dict_2[state]
            sequence = path[state]

    return sequence, probability
