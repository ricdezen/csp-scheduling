import utils
import numpy as np

from problem import ProblemWrapper as Problem

MAX_CONSECUTIVE_SLOTS = 4


def main():
    cleaners = np.array([
        [0] * 16,
        [0] * 16,
        [0] * 16,
        [1] * 16,
        [1] * 16,
        [1] * 16,
        [1] * 16
    ])
    limits = np.array([
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        4
    ])
    classrooms = np.array([[1] * 16] * 41)
    # Simulate some classes not being available all the time.
    for _ in range(41 // 5):
        classrooms[np.random.choice(range(41))] = [1] * 4 + [0] * 8 + [1] * 4

    problem = Problem(cleaners, limits, classrooms, MAX_CONSECUTIVE_SLOTS)

    utils.draw_solution(problem, problem.solution())


if __name__ == "__main__":
    main()
