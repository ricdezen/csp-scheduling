import re
import numpy as np
import matplotlib.pyplot as plt

from typing import List
from matplotlib import cm


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
        _, _, cleaner = re.search(r"X_(\d+)_(\d+)_(\d+)", var).groups()
        vars_per_cleaner[int(cleaner)].append(var)
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
        classroom, time, cleaner = re.search(r"X_(\d+)_(\d+)_(\d+)", var).groups()
        print(f"Classroom {classroom} is going to be cleaned at hour {time} by {cleaner}")


def draw_solution(n_classrooms, n_time_slots, n_cleaners, solution) -> None:
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
        classroom, time, cleaner = re.search(r"X_(\d+)_(\d+)_(\d+)", var).groups()
        image[int(time), int(classroom)] = colors[int(cleaner)]

    plt.imshow(image)
    plt.show()
