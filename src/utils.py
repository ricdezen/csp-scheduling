import re
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple
from matplotlib import cm

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


def workload(variables) -> List[int]:
    """
    :param variables: The current assignment matrix.
    :return: A list with the working time slots per worker.
    """
    return [int(np.sum(variables[:, :, k])) for k in range(variables.shape[2])]


def total_working_time(variables) -> int:
    """
    Sum the difference between start and end time for each worker.

    :param variables: The current assignment matrix.
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
        classroom, time, cleaner = var_to_num(var)
        print(f"Classroom {classroom} is going to be cleaned at hour {time} by {cleaner}")


def draw_solution(workers, limits, classrooms, solution) -> None:
    """
    Show a solution as a time (rows) by class (columns) time-table, with cells being color-coded to cleaners.

    :param workers: The worker's schedule.
    :param limits: The time slot limits for each worker.
    :param classrooms: The available slots for each classroom.
    :param solution: The solution to display.
    """
    n_classrooms, n_time_slots, n_workers = problem_size(workers, limits, classrooms)

    selected = [var for var in solution if solution[var]]
    colors = cm.tab20(range(n_workers))

    image = np.zeros((n_time_slots, n_classrooms, 4))
    for c in range(n_classrooms):
        for t in range(n_time_slots):
            if not classrooms[c][t]:
                image[t, c] = (0.5, 0.5, 0.5, 1)

    for var in selected:
        classroom, time, worker = var_to_num(var)
        image[time, classroom] = colors[worker]

    plt.imshow(image)
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
