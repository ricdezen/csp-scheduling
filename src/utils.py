import re
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple
from matplotlib import cm

_variable_regex = re.compile(r"X_(\d+)_(\d+)_(\d+)")


def num_to_var(classroom: int, time_slot: int, cleaner: int) -> str:
    return f"X_{classroom}_{time_slot}_{cleaner}"


def var_to_num(var: str) -> Tuple[int, int, int]:
    classroom, time_slot, cleaner = _variable_regex.search(var).groups()
    return int(classroom), int(time_slot), int(cleaner)


def classrooms_per_cleaner(n_cleaners, solution) -> List[int]:
    """
    :param n_cleaners: The total number of cleaners. Needed cause some might not have corresponding variables.
    :param solution: The solution to evaluate.
    :return: A list containing how many classrooms are assigned to each cleaner.
    """
    vars_per_cleaner = {x: list() for x in range(n_cleaners)}
    selected = [var for var in solution if solution[var]]
    for var in selected:
        # Add each variable to its cleaner.
        _, _, cleaner = var_to_num(var)
        vars_per_cleaner[cleaner].append(var)
    # Return how many working time slots for each cleaner.
    return [len(vars_per_cleaner[c]) for c in range(n_cleaners)]


def solution_std(n_cleaners, solution) -> float:
    """
    :param n_cleaners: The total number of cleaners. Needed cause some might not have corresponding variables.
    :param solution: The solution to evaluate.
    :return: The standard deviation of the rooms per cleaner.
    """
    return float(np.std(np.array(classrooms_per_cleaner(n_cleaners, solution))))


def print_solution(solution):
    """
    Prints the given solution in a human-readable format.
    """
    selected = [var for var in solution if solution[var]]
    for var in selected:
        classroom, time, cleaner = var_to_num(var)
        print(f"Classroom {classroom} is going to be cleaned at hour {time} by {cleaner}")


def draw_solution(n_classrooms: int, n_time_slots: int, n_cleaners: int, solution) -> None:
    """
    Show a solution as a time (rows) by class (columns) time-table, with cells being color-coded to cleaners.

    :param n_classrooms: The number of classrooms.
    :param n_time_slots: The number of time-slots.
    :param n_cleaners: The number of cleaners.
    :param solution: The solution to display.
    """
    selected = [var for var in solution if solution[var]]
    colors = cm.tab20(range(n_cleaners))

    image = np.zeros((n_time_slots, n_classrooms, 4))
    for var in selected:
        classroom, time, cleaner = var_to_num(var)
        image[time, classroom] = colors[cleaner]

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
