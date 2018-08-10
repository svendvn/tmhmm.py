import numpy as np
import scipy.misc

cimport numpy as np
cimport cython

DTYPE = np.double
ctypedef np.double_t DTYPE_t


@cython.boundscheck(False)
def viterbi(sequence,
            np.ndarray[DTYPE_t, ndim=1] initial,
            np.ndarray[DTYPE_t, ndim=2] transitions,
            np.ndarray[DTYPE_t, ndim=2] emissions,
            char_map, label_map, name_map):
    """
    Compute the most probable path through the model given the sequence.

    This function implements Viterbi's algorithm in log-space.

    :param sequence str: a string over the alphabet specified by the model.
    :rtype: tuple(matrix, optimal_path)
    :return: a tuple consisting of the dynamic programming table and the
             optimal path.
    """

    sequence = sequence.upper()

    cdef int no_observations = len(sequence)
    cdef int no_states = len(initial)

    cdef float neginf = -np.inf

    # work in log space
    initial = np.log(initial)
    transitions = np.log(transitions)
    emissions = np.log(emissions)

    cdef np.ndarray[DTYPE_t, ndim=2] M = \
        np.zeros([2, no_states],dtype=DTYPE)
    cdef np.ndarray[np.int_t, ndim=2] P = \
        np.zeros([no_observations, no_states], dtype=np.int)

    cdef unsigned int i, j, k, max_state, next_state, observation
    cdef double max_state_prob, prob

    observation = char_map[sequence[0]]
    for i in range(no_states):
        M[0, i] = initial[i] + emissions[i, observation]

    for i in range(1, no_observations):
        observation = char_map[sequence[i]]
        for j in range(no_states):
            max_state = 0
            max_state_prob = neginf
            for k in range(no_states):
                prob = M[(i - 1) % 2, k] + transitions[k, j]
                if prob > max_state_prob:
                    max_state, max_state_prob = k, prob
            M[i % 2, j] = max_state_prob + emissions[j, observation]
            P[i, j] = max_state

    # TODO: figure out why stuff doesn't work when using cython without turning
    #       the range generator into a list first.
    # TODO: stuff crashes if one uses reversed(range(no_observations)), why?

    backtracked = []
    next_state = np.argmax(M[no_observations % 2,], axis=0)
    for i in list(range(no_observations - 1, -1, -1)):
        backtracked.append(label_map[next_state])
        next_state = P[i, next_state]

    return M, ''.join(reversed(backtracked))


@cython.boundscheck(False)
def forward(sequence, 
	    map_from_index_to_transition_matrix,
            np.ndarray[DTYPE_t, ndim=1] initial,
            np.ndarray[DTYPE_t, ndim=3] transitions,
            np.ndarray[DTYPE_t, ndim=2] emissions,
            char_map, label_map, name_map):
    """
    Compute the probability distribution of states after observing the sequence.

    This function implements the scaled Forward algorithm with a heteregenous hidden state space.

    :param sequence str: a string over the alphabet specified by the model.
    :rtype: tuple(matrix, constants)
    :return: the scaled dynamic programming table and the constants used to
             normalize it.
    """

    sequence = sequence.upper()

    cdef int no_observations = len(sequence)
    cdef int no_states = len(initial)

    cdef np.ndarray[DTYPE_t, ndim=2] M = \
        np.zeros([no_observations, no_states], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] constants = \
        np.zeros(no_observations, dtype=DTYPE)

    cdef unsigned int i, j, k, observation
    cdef double prob, state_sum

    observation = char_map[sequence[0]]
    for i in range(no_states):
        M[0, i] = initial[i] * emissions[i, observation]
    constants[0] = np.sum(M[0])
    M[0] = M[0] / constants[0]

    for i in range(1, no_observations):
        observation = char_map[sequence[i]]
        transition_index = map_from_index_to_transition_matrix(i)
        for j in range(no_states):
            state_sum = 0.0
            for k in range(no_states):
                state_sum += M[(i - 1), k] * transitions[transition_index, k, j]
            M[i, j] = state_sum * emissions[j, observation]
        constants[i] = np.sum(M[i])
        M[i] = M[i] / constants[i]

    return M, constants

@cython.boundscheck(False)
def forward2(sequence, 
            np.ndarray[DTYPE_t, ndim=1] initial,
            transition_generator,
            emission_generator,
            char_map, label_map, name_map):
    """
    Compute the probability distribution of states after observing the sequence.

    This function implements the scaled Forward algorithm with a heteregenous hidden state space.

    :param sequence str: a string over the alphabet specified by the model.
    :rtype: tuple(matrix, constants)
    :return: the scaled dynamic programming table and the constants used to
             normalize it.
    """

    sequence = sequence.upper()

    cdef int no_observations = len(sequence)
    cdef int no_states = len(initial)

    cdef np.ndarray[DTYPE_t, ndim=2] M = \
        np.zeros([no_observations, no_states], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] constants = \
        np.zeros(no_observations, dtype=DTYPE)

    cdef unsigned int i, j, k, observation
    cdef double prob, state_sum

    observation = char_map[sequence[0]]
    emission=emission_generator(0)
    for i in range(no_states):
        M[0, i] = initial[i] * emission[i, observation]
    constants[0] = np.sum(M[0])
    M[0] = M[0] / constants[0]

    for i in range(1, no_observations):
        observation = char_map[sequence[i]]
        transition = transition_generator(i)
        emission=emission_generator(i)
        for j in range(no_states):
            state_sum = 0.0
            for k in range(no_states):
                state_sum += M[(i - 1), k] * transition[k, j]
            M[i, j] = state_sum * emission[j, observation]
        constants[i] = np.sum(M[i])
        M[i] = M[i] / constants[i]

    return M, constants


@cython.boundscheck(False)
def backward(sequence,
             constants,
             np.ndarray[DTYPE_t, ndim=1] initial,
             np.ndarray[DTYPE_t, ndim=2] transitions,
             np.ndarray[DTYPE_t, ndim=2] emissions,
             char_map, label_map, name_map):
    """
    Compute the probability of being in some state and generating the rest of
    the sequence.

    This function implements the scaled backward algorithm.

    :param sequence str: a string over the alphabet specified by the model.
    :param constants np.ndarray: an array of the constants used to normalize
                                 the forward table.
    :rtype: np.ndarray
    :return: the scaled backward table.
    """

    sequence = sequence.upper()

    cdef int no_observations = len(sequence)
    cdef int no_states = len(initial)

    cdef np.ndarray[DTYPE_t, ndim=2] M = \
        np.zeros([no_observations, no_states], dtype=DTYPE)

    cdef unsigned int i, j, k, observation
    cdef double prob, state_sum

    M[no_observations-1] = 1.0 / constants[no_observations - 1]

    for i in range(no_observations-2, -1, -1):
        observation = char_map[sequence[i]]
        for j in range(no_states):
            state_sum = 0.0
            for k in range(no_states):
                state_sum += M[(i + 1), k] * transitions[j, k]
            M[i, j] = state_sum * emissions[j, observation]
        M[i] = M[i] / constants[i]

    return M

@cython.boundscheck(False)
def backward2(sequence,
             constants,
             np.ndarray[DTYPE_t, ndim=1] initial,
             transition_generator,
             emission_generator,
             char_map, label_map, name_map):
    """
    Compute the probability of being in some state and generating the rest of
    the sequence.

    This function implements the scaled backward algorithm.

    :param sequence str: a string over the alphabet specified by the model.
    :param constants np.ndarray: an array of the constants used to normalize
                                 the forward table.
    :rtype: np.ndarray
    :return: the scaled backward table.
    """

    sequence = sequence.upper()

    cdef int no_observations = len(sequence)
    cdef int no_states = len(initial)

    cdef np.ndarray[DTYPE_t, ndim=2] M = \
        np.zeros([no_observations, no_states], dtype=DTYPE)

    cdef unsigned int i, j, k, observation
    cdef double prob, state_sum

    M[no_observations-1] = 1.0 / constants[no_observations - 1]
    emission=emission_generator(0)

    for i in range(no_observations-2, -1, -1):
        observation = char_map[sequence[i]]
        transition = transition_generator(i)
        emission=emission_generator(i)

        for j in range(no_states):
            state_sum = 0.0
            for k in range(no_states):
                state_sum += M[(i + 1), k] * transition[j, k]
            M[i, j] = state_sum * emission[j, observation]
            M[i] = M[i] / constants[i]
    return M

@cython.boundscheck(False)
def backward2log(sequence,
             log_constants,
             np.ndarray[DTYPE_t, ndim=1] initial,
             transition_generator,
             log_emission_generator,
             char_map, label_map, name_map):
    """
    Compute the probability of being in some state and generating the rest of
    the sequence.

    This function implements the scaled backward algorithm.

    :param sequence str: a string over the alphabet specified by the model.
    :param constants np.ndarray: an array of the constants used to normalize
                                 the forward table.
    :rtype: np.ndarray
    :return: the scaled backward table.
    """

    sequence = sequence.upper()

    cdef int no_observations = len(sequence)
    cdef int no_states = len(initial)

    cdef np.ndarray[DTYPE_t, ndim=2] M = \
        np.zeros([no_observations, no_states], dtype=DTYPE)
        
    cdef np.ndarray[DTYPE_t, ndim=2] state_sum = \
                np.zeros([1, no_states],dtype=DTYPE)

    cdef unsigned int i, j, k, observation

    M[no_observations-1] = - log_constants[no_observations - 1]

    for i in range(no_observations-2, -1, -1):
        observation = char_map[sequence[i]]
        transition = transition_generator(i)
        log_emission=np.log(log_emission_generator(i))

        for j in range(no_states):
            for k in range(no_states):
                state_sum[0,k] = M[(i + 1), k] +np.log( transition[j, k])
            M[i, j] = scipy.misc.logsumexp(state_sum)+log_emission[j,observation]
            M[i] = M[i] - np.min(M[i,j])
    return np.exp(M)

@cython.boundscheck(False)
def forward2log(sequence, 
            np.ndarray[DTYPE_t, ndim=1] log_initial,
            transition_generator,
            log_emission_generator,
            char_map, label_map, name_map):
    """
    Compute the probability distribution of states after observing the sequence.

    This function implements the scaled Forward algorithm with a heteregenous hidden state space.

    :param sequence str: a string over the alphabet specified by the model.
    :rtype: tuple(matrix, constants)
    :return: the scaled dynamic programming table and the constants used to
             normalize it.
    """

    sequence = sequence.upper()

    cdef int no_observations = len(sequence)
    cdef int no_states = len(log_initial)

    cdef np.ndarray[DTYPE_t, ndim=2] M = \
        np.zeros([no_observations, no_states], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] constants = \
        np.zeros(no_observations, dtype=DTYPE)

    cdef unsigned int i, j, k, observation
    cdef double prob, state_sum

    observation = char_map[sequence[0]]
    log_emission=log_emission_generator(0)
    for i in range(no_states):
        M[0, i] = log_initial[i] + log_emission[i, observation]
    constants[0] = np.sum(M[0])
    M[0] = M[0] / constants[0]

    for i in range(1, no_observations):
        observation = char_map[sequence[i]]
        transition = transition_generator(i)
        emission=log_emission_generator(i)
        for j in range(no_states):
            state_sum = 0.0
            for k in range(no_states):
                state_sum += M[(i - 1), k] * transition[k, j]
            M[i, j] = state_sum * emission[j, observation]
        constants[i] = np.sum(M[i])
        M[i] = M[i] / constants[i]

    return M, constants
