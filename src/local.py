"""
Module for local search.

Attempt hill climbing to find better states from a valid solution.
"""
import utils
import itertools
import numpy as np

from common import Problem


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


def hill_climbing(problem: Problem, variables: np.ndarray):
    """
    Attempt to improve a state via hill climbing. Passing a copy of variables is recommended because it WILL be modified
    at each step.

    :param problem: The problem for which to improve the state.
    :param variables: The current assignment.
    :return: An improved solution, if one is found.
    """
    return _hill_climbing(
        problem.existing_variables,
        problem.classrooms,
        problem.workers,
        problem.limits,
        problem.max_consecutive,
        problem.n_classrooms,
        problem.n_time_slots,
        problem.n_workers,
        variables
    )


def _hill_climbing(
        existing_variables: np.ndarray,
        classrooms: np.ndarray,
        workers: np.ndarray,
        limits: np.ndarray,
        max_consecutive,
        n_classrooms: np.ndarray,
        n_time_slots: np.ndarray,
        n_workers: np.ndarray,
        variables: np.ndarray
):
    """
    Implementation of `hill_climbing`. Does not need to cache references at each recursive step.
    """
    # Step 1: improve attempting to balance the workload (std of classes per cleaner)

    # Step 2: improve trying to reduce total work-time (work + breaks) of personnel.
    # Cache some constants.

    # Used later to check limits.
    slots_per_cleaner = utils.workload(variables)

    print(slots_per_cleaner)

    # I need to select only relevant variables.
    ones = np.nonzero(np.logical_and(existing_variables == 1, variables == 1))
    zeros = np.nonzero(np.logical_and(existing_variables == 1, variables == 0))

    ones = list(zip(*ones))
    zeros = list(zip(*zeros))

    # Only swapping ones with zeros does something.
    actions = list(itertools.product(zeros, ones))

    print(f"I have {len(actions)} possible actions.")

    def is_valid_action(action):
        FLAG = False
        one_c, one_t, one_k = action[1]
        zero_c, zero_t, zero_k = action[0]

        if one_c == zero_c and one_k == zero_k == 6:
            FLAG = True
            print(f"Action of moving class {one_c} for guy 6 from {one_t} to {zero_t}.")

        # Check if classroom can be cleaned and worker can work.
        if not workers[zero_k][zero_t] or not classrooms[zero_c][zero_t]:
            if FLAG:
                print("Rejected at condition 1.")
            return False

        # Needs to be same class.
        if one_c == zero_c:
            # Destination worker must not be already somewhere else.
            if 1 in variables[:, zero_t, zero_k]:
                if FLAG:
                    print("Rejected at condition 2.")
                return False
        else:
            if FLAG:
                print("Rejected at condition 3.")
            return False

        # Ensure I do not exceed the limit of the new worker (if it was changed)
        if zero_k != one_k and (0 <= limits[zero_k] < slots_per_cleaner[zero_k] + 1):
            if FLAG:
                print("Rejected at condition 4.")
            return False

        # Ensure no consecutive 5 slot shifts are formed.
        # TODO not optimized
        working_when = np.sum(variables[:, :, zero_k], axis=0)
        working_when[zero_t] = 1  # Pretend to assign the classroom.
        for i in range(0, n_time_slots - max_consecutive - 1):
            if np.sum(working_when[i:i + max_consecutive + 1]) > max_consecutive:
                if FLAG:
                    print("Rejected at condition 5.")
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
        working_time.append(utils.total_working_time(variables))
        workload_std.append(utils.workload_std(variables))
        # Revert action
        revert(variables, a)

    # I must be able to compute the total cost.
    sorted_actions = list(sorted(zip(valid_actions, working_time, workload_std), key=lambda x: (x[1], x[2])))

    print(sorted_actions[:100])

    print("But why tho")
    # print(list(filter(lambda x: x[0][2] == 6 and x[1][2] == 6, actions)))

    # TODO Currently does nothing.
    return variables
