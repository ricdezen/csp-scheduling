"""
Module for local search.

Attempt hill climbing to find better states from a valid solution.
"""
import utils
import itertools
import numpy as np


def total_working_time(variables) -> int:
    """
    Sum the difference between start and end time for each worker.

    :param variables: The current assignment.
    :return: The sum of the working hours for each worker.
    """
    _, _, K = variables.shape
    total_time = 0
    for k in range(K):
        # When does the person work.
        working_when = np.sum(variables[:, :, k], axis=0)
        working_slots = np.nonzero(working_when)[0]
        # Does not work, skip
        if len(working_slots) == 0:
            continue
        # Works last - first + 1 total slots.
        total_time += working_slots[-1] - working_slots[0] + 1
    return total_time


def total_std(variables) -> float:
    """
    :param variables: The current assignment.
    :return: The standard deviations of the workload per worker.
    """
    _, _, K = variables.shape
    time_per_worker = [np.sum(variables[:, :, k]) for k in range(K)]
    return float(np.std(np.array(time_per_worker)))


def act(variables, action):
    """
    Apply action to variables, modifying them.
    Assumes the action is valid.
    """
    variables[action[0]] = 1
    variables[action[1]] = 0


def revert(variables, action):
    """
    Do the opposite of what the action does.
    Assumes the action is valid.
    """
    variables[action[0]] = 0
    variables[action[1]] = 1


def hill_climbing(classrooms, limits, cleaners, max_consecutive, solution):
    """
    Attempt to improve a solution.

    :param cleaners: 2D array with each row having the available time slots for each cleaner.
    :param limits: 1D array with the max number of hours per cleaner. Negative value means no limit.
    :param classrooms: 2D array with each row having the available time slots for each classroom.
    :param max_consecutive: Maximum consecutive slots.
    :param solution: A solution to the problem.
    :return: An improved solution, if one is found.
    """
    # Step 1: improve attempting to balance the workload (std of classes per cleaner)

    # Step 2: improve trying to reduce total work-time (work + breaks) of personnel.

    n_classrooms, n_time_slots, n_cleaners = utils.problem_size(cleaners, limits, classrooms)

    # Used later to check limits.
    slots_per_cleaner = utils.classrooms_per_cleaner(n_cleaners, solution)

    variables_exist = np.zeros((n_classrooms, n_time_slots, n_cleaners))
    variables = np.zeros((n_classrooms, n_time_slots, n_cleaners))

    for var, value in solution.items():
        c, t, k = utils.var_to_num(var)
        variables_exist[c, t, k] = 1
        variables[c, t, k] = value

    # I need to select only relevant variables.
    ones = np.nonzero(np.logical_and(variables_exist == 1, variables == 1))
    zeros = np.nonzero(np.logical_and(variables_exist == 1, variables == 0))

    ones = list(zip(*ones))
    zeros = list(zip(*zeros))

    # Only swapping ones with zeros does something.
    actions = list(itertools.product(zeros, ones))

    print(f"I have {len(actions)} possible actions.")

    def is_valid_action(action):
        one_c, one_t, one_k = action[1]
        zero_c, zero_t, zero_k = action[0]

        # Check if classroom can be cleaned and worker can work.
        if not cleaners[zero_k][zero_t] or not classrooms[zero_c][zero_t]:
            return False

        # Needs to be same class.
        if one_c == zero_c:
            # Destination worker must not be already somewhere else.
            if 1 in variables[:, zero_t, zero_k]:
                return False
        else:
            return False

        # Ensure I do not exceed the limit of the new worker
        if 0 <= limits[zero_k] < slots_per_cleaner[zero_k] + 1:
            return False

        # Ensure no consecutive 5 slot shifts are formed.
        # TODO not optimized
        working_when = np.copy(np.sum(variables[:, :, zero_k], axis=0))
        working_when[zero_t] = 1  # Pretend to assign the classroom.
        for i in range(0, n_time_slots - max_consecutive - 1):
            if np.sum(working_when[i:i + max_consecutive + 1]) > max_consecutive:
                return False

        # Check worker is only in one place.

        return True

    valid_actions = list(filter(is_valid_action, actions))

    print(f"I have {len(valid_actions)} valid actions that I can take.")

    working_time = list()
    workload_std = list()
    for a in valid_actions:
        # Apply action
        act(variables, a)
        # Compute working time
        working_time.append(total_working_time(variables))
        workload_std.append(total_std(variables))
        # Revert action
        revert(variables, a)

    # I must be able to compute the total cost.
    sorted_actions = list(sorted(zip(valid_actions, working_time, workload_std), key=lambda x: (x[1], x[2])))
    print(sorted_actions)
    utils.draw_solution(cleaners, limits, classrooms, solution)
