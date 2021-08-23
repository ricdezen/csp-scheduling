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


def problem_size(cleaners, limits, classrooms) -> Tuple[int, int, int]:
    """
    Infer problem size.
    :return: A tuple made of the number of classrooms, the number of timeslots, the number of workers.
    :raises: ValueError if the time slots of the cleaners are more or less than the ones for the classrooms.
             ValueError if there are not enough limits to cover the workers.
    """
    # Constants can be inferred directly from data size.
    if cleaners.shape[1] != classrooms.shape[1]:
        raise ValueError(f"{cleaners.shape[1]} time slots for cleaners, but {classrooms.shape[1]} for classrooms.")
    if cleaners.shape[0] != limits.shape[0]:
        raise ValueError(f"{cleaners.shape[0]} cleaners found but {limits.shape[0]} time slot limits.")

    return classrooms.shape[0], cleaners.shape[1], cleaners.shape[0]


def classrooms_per_cleaner(n_workers, solution) -> List[int]:
    """
    :param n_workers: The total number of workers. Needed cause some might not have corresponding variables.
    :param solution: The solution to evaluate.
    :return: A list containing how many classrooms are assigned to each cleaner.
    """
    vars_per_worker = {x: list() for x in range(n_workers)}
    selected = [var for var in solution if solution[var]]
    for var in selected:
        # Add each variable to its cleaner.
        _, _, worker = var_to_num(var)
        vars_per_worker[worker].append(var)
    # Return how many working time slots for each cleaner.
    return [len(vars_per_worker[c]) for c in range(n_workers)]


def solution_std(n_workers, solution) -> float:
    """
    :param n_workers: The total number of workers. Needed cause some might not have corresponding variables.
    :param solution: The solution to evaluate.
    :return: The standard deviation of the rooms per cleaner.
    """
    return float(np.std(np.array(classrooms_per_cleaner(n_workers, solution))))


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
