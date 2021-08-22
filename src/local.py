"""
Module for local search.

Attempt hill climbing to find better states from a valid solution.
"""
import utils
import itertools
import numpy as np


def hill_climbing(classrooms, limits, cleaners, solution):
    """
    Attempt to improve a solution.

    :param cleaners: 2D array with each row having the available time slots for each cleaner.
    :param limits: 1D array with the max number of hours per cleaner. Negative value means no limit.
    :param classrooms: 2D array with each row having the available time slots for each classroom.
    :param solution: A solution to the problem.
    :return: An improved solution, if one is found.
    """
    # Step 1: improve attempting to balance the workload (std of classes per cleaner)

    # Step 2: improve trying to reduce total work-time (work + breaks) of personnel.

    n_time_slots = cleaners.shape[1]
    n_cleaners = cleaners.shape[0]
    n_classrooms = classrooms.shape[0]

    variables_exist = np.zeros((n_classrooms, n_time_slots, n_cleaners))
    variables = np.zeros((n_classrooms, n_time_slots, n_cleaners))

    for var, value in solution.items():
        c, t, k = utils.var_to_num(var)
        variables_exist[c, t, k] = 1
        variables[c, t, k] = value

    # I need to select only relevant variables.
    ones = np.nonzero(np.logical_and(variables_exist == 1, variables == 1))
    zeros = np.nonzero(np.logical_and(variables_exist == 1, variables == 0))

    # Only swapping ones with zeros does something.
    actions = itertools.product(ones, zeros)

    # Since we want to rebalance the total work time, without breaking the workload.
    pass
