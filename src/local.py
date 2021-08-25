"""
Module for local search.

Attempt hill climbing to find better states from a valid solution.
"""
import math

import utils
import itertools
import numpy as np

from typing import Optional
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
    Attempt to improve a state via hill climbing.

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
        np.copy(variables)
    )


def _hill_climbing(
        existing_variables: np.ndarray,
        classrooms: np.ndarray,
        workers: np.ndarray,
        limits: np.ndarray,
        max_consecutive: int,
        n_classrooms: int,
        n_time_slots: int,
        n_workers: int,
        variables: np.ndarray
) -> np.ndarray:
    """
    Actual implementation of the `hill_climbing` function. Does not need to cache references at each recursive step.
    """
    # Try to reduce total work-time (work + breaks) of personnel, without penalizing std of workload.

    # Info on current state.
    slots_per_cleaner = utils.workload(variables)
    working_when_all = [np.sum(variables[:, :, w], axis=0) for w in range(n_workers)]
    current_working_time = utils.total_working_time(variables)
    current_workload_std = utils.workload_std(variables)

    # Select ones and zeros.
    ones = np.nonzero(np.logical_and(existing_variables == 1, variables == 1))
    zeros = np.nonzero(np.logical_and(existing_variables == 1, variables == 0))

    ones = list(zip(*ones))
    zeros = list(zip(*zeros))

    # Only swapping ones with zeros does something.
    actions = list(itertools.product(zeros, ones))

    # Defined as local to reduce overhead.
    def is_valid_action(action):
        one_c, one_t, one_k = action[1]
        zero_c, zero_t, zero_k = action[0]

        # Check if classroom can be cleaned and worker can work.
        if not workers[zero_k][zero_t] or not classrooms[zero_c][zero_t]:
            return False

        # Needs to be same class.
        if one_c == zero_c:
            # Destination worker must not be already somewhere else.
            if 1 in variables[:, zero_t, zero_k]:
                return False
        else:
            return False

        # Ensure I do not exceed the limit of the new worker (if it was changed)
        if zero_k != one_k and (0 <= limits[zero_k] < slots_per_cleaner[zero_k] + 1):
            return False

        # Ensure no consecutive slot shifts are formed.
        working_when = working_when_all[zero_k]
        working_when[zero_t] = 1  # Pretend to assign the classroom.
        start = max(0, zero_t - max_consecutive)
        end = min(zero_t, n_time_slots - max_consecutive - 1)
        for i in range(start, end + 1):
            if np.sum(working_when[i:i + max_consecutive + 1]) > max_consecutive:
                # Reset working_when
                working_when[zero_t] = 0
                return False
        # Reset working_when
        working_when[zero_t] = 0

        return True

    valid_actions = list(filter(is_valid_action, actions))

    # If no valid actions are allowed, this is a dead end.
    if not valid_actions:
        print(f"{utils.who()} Reached a dead end.")
        return variables

    print(f"{utils.who()} There are {len(valid_actions)} valid actions that I can take.")

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

    # Sort actions by cost. Priority to reducing working time, then std.
    sorted_actions = sorted(zip(valid_actions, working_time, workload_std), key=lambda x: (x[1], x[2]))

    # Std in new state can be interpreted as follows:
    # - Equal: I am moving an hour of a single worker.
    # - Lower: I am moving a classroom from a worker to another one with less hours.
    # - Higher: I am moving a classroom from a worker to another one that already has more hours.
    for good_action in sorted_actions:
        a, new_time, new_std = good_action
        # When the working time decreases, the std must remain lower or equal.
        if new_time < current_working_time and new_std <= current_workload_std:
            print(f"{utils.who()} {a} has been chosen.")
            act(variables, a)
            return _hill_climbing(
                existing_variables,
                classrooms,
                workers,
                limits,
                max_consecutive,
                n_classrooms,
                n_time_slots,
                n_workers,
                variables
            )
        # When the working time remains equal, the std must be lower.
        if new_time == current_working_time and new_std < current_workload_std:
            print(f"{utils.who()} {a} has been chosen.")
            act(variables, a)
            return _hill_climbing(
                existing_variables,
                classrooms,
                workers,
                limits,
                max_consecutive,
                n_classrooms,
                n_time_slots,
                n_workers,
                variables
            )
        # Else, the action does not improve upon our situation.

    # Reached local maxima
    return variables


def energy(variables) -> float:
    """
    :return: Combined fitness score for a state.
    """
    return utils.total_working_time(variables) + utils.workload_std(variables)


def acceptance(e_old, e_new, temperature) -> float:
    """
    :return: The acceptance probability in [0, 1] of a new state.
    """
    return 1 if e_new < e_old else math.exp((e_old - e_new) / temperature)


def simulated_annealing(
        problem: Problem,
        variables: np.ndarray,
        max_steps: int = 1000,
        initial_t: float = 1000,
        cooling_factor: float = 0.99,
        max_dead_ends: int = 10
):
    """
    Attempt to improve a state via simulated annealing.

    :param problem: The problem for which to improve the state.
    :param variables: The current assignment.
    :param max_steps: Maximum number of steps that can be taken. Defaults to 1000.
    :param initial_t: The initial temperature. Defaults to 1000.
    :param cooling_factor: The multiplicative cooling factor between steps. Defaults to 0.99.
    :param max_dead_ends: The maximum number of consecutive dead ends before stopping.
    :return: The resulting state, if one is found.
    """
    # Geometric cooling scheme. From initial temperature
    variables = np.copy(variables)
    temperature = initial_t
    dead_ends = 0
    for _ in range(max_steps):
        new_variables = _simulated_annealing(
            problem.existing_variables,
            problem.classrooms,
            problem.workers,
            problem.limits,
            problem.max_consecutive,
            problem.n_classrooms,
            problem.n_time_slots,
            problem.n_workers,
            variables,
            temperature
        )
        # Count consecutive dead ends.
        if new_variables is None:
            dead_ends += 1
        else:
            # Don't need to assign variables = new_variables because they point to the same object.
            dead_ends = 0
        # Check max dead ends.
        if dead_ends > max_dead_ends:
            return variables
        temperature = temperature * cooling_factor

    return variables


def _simulated_annealing(
        existing_variables: np.ndarray,
        classrooms: np.ndarray,
        workers: np.ndarray,
        limits: np.ndarray,
        max_consecutive: int,
        n_classrooms: int,
        n_time_slots: int,
        n_workers: int,
        variables: np.ndarray,
        temperature: float
) -> Optional[np.ndarray]:
    """
    Actual implementation of the `simulated_annealing` function. Not recursive, applies a single step.
    WILL modify variables.
    """
    # Try to reduce total work-time (work + breaks) and std.

    # Info on current state.
    slots_per_cleaner = utils.workload(variables)
    current_energy = energy(variables)
    working_when_all = [np.sum(variables[:, :, w], axis=0) for w in range(n_workers)]

    # Select ones and zeros.
    ones = np.nonzero(np.logical_and(existing_variables == 1, variables == 1))
    zeros = np.nonzero(np.logical_and(existing_variables == 1, variables == 0))

    ones = list(zip(*ones))
    zeros = list(zip(*zeros))

    # Only swapping ones with zeros does something.
    actions = list(itertools.product(zeros, ones))

    # Defined as local to reduce overhead.
    def is_valid_action(action):
        one_c, one_t, one_k = action[1]
        zero_c, zero_t, zero_k = action[0]

        # Check if classroom can be cleaned and worker can work.
        if not workers[zero_k][zero_t] or not classrooms[zero_c][zero_t]:
            return False

        # Needs to be same class.
        if one_c == zero_c:
            # Destination worker must not be already somewhere else.
            if 1 in variables[:, zero_t, zero_k]:
                return False
        else:
            return False

        # Ensure I do not exceed the limit of the new worker (if it was changed)
        if zero_k != one_k and (0 <= limits[zero_k] < slots_per_cleaner[zero_k] + 1):
            return False

        # Ensure no consecutive slot shifts are formed.
        working_when = working_when_all[zero_k]
        working_when[zero_t] = 1  # Pretend to assign the classroom.
        start = max(0, zero_t - max_consecutive)
        end = min(zero_t, n_time_slots - max_consecutive - 1)
        for i in range(start, end + 1):
            if np.sum(working_when[i:i + max_consecutive + 1]) > max_consecutive:
                # Reset working_when
                working_when[zero_t] = 0
                return False
        # Reset working_when
        working_when[zero_t] = 0

        return True

    valid_actions = list(filter(is_valid_action, actions))

    # If no valid actions are allowed, this is a dead end.
    if not valid_actions:
        print(f"{utils.who()} Reached a dead end.")
        return None

    print(f"{utils.who()} There are {len(valid_actions)} valid actions that I can take.")

    new_energies = list()
    for a in valid_actions:
        # Apply action
        act(variables, a)
        # Compute energy
        new_energies.append(energy(variables))
        # Revert action
        revert(variables, a)

    # Randomly select a new action until one is accepted.
    order = np.random.permutation(len(valid_actions))
    for i in order:
        a = valid_actions[i]
        e = new_energies[i]
        if np.random.rand() < acceptance(current_energy, e, temperature):
            print(f"{utils.who()} {a} has been chosen.")
            act(variables, a)
            return variables

    # No action could be accepted.
    return variables
