import utils
import local
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

    base_solution = problem.solution()
    utils.draw_solution(problem, base_solution)

    better_solution_hill = local.hill_climbing(problem, utils.dictionary_to_matrix(problem, base_solution))
    better_solution_ann = local.simulated_annealing(problem, utils.dictionary_to_matrix(problem, base_solution))

    print(utils.total_working_time(better_solution_hill))
    print(utils.workload_std(better_solution_hill))
    utils.draw_solution(problem, utils.matrix_to_dictionary(problem, better_solution_hill))

    print(utils.total_working_time(better_solution_ann))
    print(utils.workload_std(better_solution_ann))
    utils.draw_solution(problem, utils.matrix_to_dictionary(problem, better_solution_ann))


if __name__ == "__main__":
    main()
