import utils
import local
import numpy as np

from problem import ProblemWrapper as Problem

MAX_CONSECUTIVE_SLOTS = 4


def main():
    cleaners = np.array([
        [1] * 16,
        [1] * 16,
        [1] * 16,
        [1] * 16
    ])
    limits = np.array([
        -1,
        -1,
        -1,
        4
    ])
    classrooms = np.array([[1] * 16] * 40)
    # Simulate some classes not being available all the time.
    for _ in range(8):
        classrooms[np.random.choice(range(40))] = [1] * 4 + [0] * 8 + [1] * 4

    problem = Problem(cleaners, limits, classrooms, MAX_CONSECUTIVE_SLOTS)

    base_solution = problem.solution()
    utils.plot_by_classroom(problem, base_solution)
    utils.plot_by_worker(problem, base_solution)

    better_solution_hill = local.hill_climbing(problem, utils.dictionary_to_matrix(problem, base_solution))
    better_solution_ann = local.simulated_annealing(problem, utils.dictionary_to_matrix(problem, base_solution))

    print(utils.total_working_time(better_solution_hill))
    print(utils.workload_std(better_solution_hill))
    utils.plot_by_classroom(problem, utils.matrix_to_dictionary(problem, better_solution_hill))

    print(utils.total_working_time(better_solution_ann))
    print(utils.workload_std(better_solution_ann))
    utils.plot_by_classroom(problem, utils.matrix_to_dictionary(problem, better_solution_ann))

    """
    results = dict()
    for max_temp in range(1000, 10001, 1000):
        for max_iters in range(1000, 2000, 100):
            for cooling_factor in range(990, 1000, 1):
                """


if __name__ == "__main__":
    main()
