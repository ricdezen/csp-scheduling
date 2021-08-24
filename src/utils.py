import re
import inspect
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple, Dict
from matplotlib import cm
from matplotlib import patches
from common import Problem

_variable_regex = re.compile(r"X_(\d+)_(\d+)_(\d+)")


def num_to_var(classroom: int, time_slot: int, worker: int) -> str:
    return f"X_{classroom}_{time_slot}_{worker}"


def var_to_num(var: str) -> Tuple[int, int, int]:
    classroom, time_slot, worker = _variable_regex.search(var).groups()
    return int(classroom), int(time_slot), int(worker)


def problem_size(workers, limits, classrooms) -> Tuple[int, int, int]:
    """
    Infer problem size.
    :return: A tuple made of the number of classrooms, the number of timeslots, the number of workers.
    :raises: ValueError if the time slots of the cleaners are more or less than the ones for the classrooms.
             ValueError if there are not enough limits to cover the workers.
    """
    # Constants can be inferred directly from data size.
    if workers.shape[1] != classrooms.shape[1]:
        raise ValueError(f"{workers.shape[1]} time slots for workers, but {classrooms.shape[1]} for classrooms.")
    if workers.shape[0] != limits.shape[0]:
        raise ValueError(f"{workers.shape[0]} workers found but {limits.shape[0]} time slot limits.")

    return classrooms.shape[0], workers.shape[1], workers.shape[0]


def existing_variables(workers, classrooms) -> np.ndarray:
    """
    Get which variables exist. Assumes workers and classrooms are valid according to `problem_size`.

    :return: A matrix where matrix[i,j,k] indicates whether variable X_i_j_k exists in the problem.
    """
    n_classrooms, n_time_slots, n_workers = classrooms.shape[0], workers.shape[1], workers.shape[0]
    variables_exist = np.zeros((n_classrooms, n_time_slots, n_workers))
    for c in range(n_classrooms):
        for t in range(n_time_slots):
            for w in range(n_workers):
                if workers[w][t] and classrooms[c][t]:
                    variables_exist[c, t, w] = 1
    return variables_exist


def dictionary_to_matrix(problem: Problem, solution: dict) -> np.ndarray:
    """
    :param problem: The problem to which the solution refers to, used to get size.
    :param solution: The solution as returned by the csp solver.
    :return: A 3D matrix representing the solution, where matrix[i,j,k] is the value of X_i_j_k.
    """
    variables = np.zeros((problem.n_classrooms, problem.n_time_slots, problem.n_workers))
    for var, value in solution.items():
        c, t, w = var_to_num(var)
        variables[c, t, w] = value
    return variables


def matrix_to_dictionary(problem: Problem, variables: np.ndarray) -> Dict:
    """
    :param problem: The problem to which the assignment refers to.
    :param variables: The assignment to be converted to dictionary.
    :return: A dictionary representing the assignment in the context of the given problem.
    """
    solution = dict()
    for c in range(problem.n_classrooms):
        for t in range(problem.n_time_slots):
            for w in range(problem.n_workers):
                if problem.existing_variables[c, t, w]:
                    solution[num_to_var(c, t, w)] = variables[c, t, w]
    return solution


def workload(variables) -> List[int]:
    """
    :param variables: The current assignment matrix.
    :return: A list with the working time slots per worker.
    """
    return [int(np.sum(variables[:, :, w])) for w in range(variables.shape[2])]


def total_working_time(variables) -> int:
    """
    Sum the difference between start and end time for each worker.

    :param variables: The current assignment matrix.
    :return: The sum of the working hours for each worker.
    """
    _, _, n_workers = variables.shape
    total_time = 0
    for w in range(n_workers):
        # When does the person work.
        working_when = np.sum(variables[:, :, w], axis=0)
        working_slots = np.nonzero(working_when)[0]
        # Does not work, skip
        if len(working_slots) == 0:
            continue
        # Works last - first + 1 total slots.
        total_time += working_slots[-1] - working_slots[0] + 1
    return total_time


def workload_std(variables) -> float:
    """
    :param variables: The current assignment matrix.
    :return: The standard deviation of the rooms per cleaner.
    """
    return float(np.std(np.array(workload(variables))))


def print_solution(solution):
    """
    Prints the given solution in a human-readable format.
    """
    selected = [var for var in solution if solution[var]]
    for var in selected:
        classroom, time, worker = var_to_num(var)
        print(f"Classroom {classroom} is going to be cleaned at hour {time} by {worker}")


def plot_by_classroom(problem: Problem, solution: Dict):
    """
    Show a solution as a time (rows) by classroom (columns) time-table, with cells being color-coded to workers.

    :param problem: The problem.
    :param solution: The solution to display.
    """
    colors = cm.tab20(range(problem.n_workers))

    # Plot solution
    figure = plt.figure()
    axes = figure.add_axes([0.12, 0.1, 0.65, 0.8])
    _plot_by_classroom(axes, problem, solution, colors)

    # Add a legend
    items = [patches.Patch(color=colors[w], label=f"Worker {w}") for w in range(problem.n_workers)]
    axes.legend(handles=items, bbox_to_anchor=(1.3, 1), loc='upper right')

    plt.show()


def _plot_by_classroom(axes, problem: Problem, solution: Dict, colors):
    selected = [var for var in solution if solution[var]]

    image = np.zeros((problem.n_time_slots, problem.n_classrooms, 4))
    for c in range(problem.n_classrooms):
        for t in range(problem.n_time_slots):
            if not problem.classrooms[c][t]:
                image[t, c] = (0.5, 0.5, 0.5, 1)

    for var in selected:
        classroom, time, worker = var_to_num(var)
        image[time, classroom] = colors[worker]

    axes.imshow(image, aspect="auto")

    # Set horizontal classroom ticks.
    axes.set_xticks(range(0, problem.n_classrooms))
    axes.set_xticklabels(range(1, problem.n_classrooms + 1), rotation=270)

    # Set vertical time ticks.
    axes.set_yticks(range(0, problem.n_time_slots))
    axes.set_yticklabels(problem.time_ticks)

    axes.set_xlabel("Classrooms")
    axes.set_ylabel("Time")


def plot_by_worker(problem: Problem, solution: Dict):
    """
    Show a solution as a worker (rows) by time (columns) time-table, with cells being color-coded to workers.

    :param problem: The problem.
    :param solution: The solution to display.
    """
    colors = cm.tab20(range(problem.n_workers))

    # Plot solution
    figure = plt.figure()
    axes = figure.add_axes([0.12, 0.1, 0.65, 0.8])
    _plot_by_worker(axes, problem, solution, colors)

    # Add a legend
    items = [patches.Patch(color=colors[w], label=f"Worker {w}") for w in range(problem.n_workers)]
    axes.legend(handles=items, bbox_to_anchor=(1.3, 1), loc='upper right')

    plt.show()


def _plot_by_worker(axes, problem: Problem, solution: Dict, colors):
    selected = [var for var in solution if solution[var]]

    image = np.zeros((problem.n_workers, problem.n_time_slots, 4))
    for w in range(problem.n_workers):
        for t in range(problem.n_time_slots):
            if not problem.workers[w][t]:
                image[w, t] = (0.5, 0.5, 0.5, 1)

    for var in selected:
        classroom, time, worker = var_to_num(var)
        image[worker, time] = colors[worker]

    axes.imshow(image, aspect="auto")

    # Set horizontal time ticks.
    axes.set_xticks(range(0, problem.n_time_slots))
    axes.set_xticklabels(problem.time_ticks, rotation=270)

    # Remove vertical worker ticks.
    axes.set_yticks([])

    axes.set_xlabel("Time")


def plot_together(problem: Problem, solution: Dict):
    colors = cm.tab20(range(problem.n_workers))

    # Plot
    figure, axes = plt.subplots(2)
    _plot_by_classroom(axes[0], problem, solution, colors)
    _plot_by_worker(axes[1], problem, solution, colors)

    # Show legend.
    items = [patches.Patch(color=colors[w], label=f"Worker {w}") for w in range(problem.n_workers)]
    plt.legend(handles=items, loc='lower right')

    plt.show()


def random_choice(array) -> tuple:
    """
    :param array: A multidimensional array with the shape attribute.
    :return: A tuple of the index of a random choice in the array.
    """
    choice = np.random.choice(range(array.shape[0]))
    # Recursion to lower dimension.
    if len(array.shape) > 1:
        return (choice,) + random_choice(array[choice])
    return choice,


def who() -> str:
    """
    :return: The name of the function that called this function, surrounded by square brackets.
    """
    return "[" + inspect.stack()[1].function + "]"
