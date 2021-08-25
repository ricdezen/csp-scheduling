import utils
import local
import time
import random
import numpy as np

from typing import Tuple, Dict
from problem import ProblemWrapper as Problem

MAX_CONSECUTIVE_SLOTS = 4


# Each time slot is 15 minutes.
# We care about the availability after 1 p.m. since that is when the classrooms become available.
# Of course in a real use we would have an UI to select an arbitrary start time and such.
# In total we have 22 time slots (13:00-18:30 @ 15 min).
#
# 3 workers end their shift at 13:30.
# 4 start at 14:00 and end at 18:30.
# 40% of classrooms is freed as soon as 13:00. The rest at 14:00.
# 10 classroom are used again between 15:00 and 17:00 and have to be cleaned twice, therefore are accounted for twice.

def get_data():
    cleaners = np.array([
        [1, 1] + [0] * 20,
        [1, 1] + [0] * 20,
        [1, 1] + [0] * 20,
        [0] * 4 + [1] * 18,
        [0] * 4 + [1] * 18,
        [0] * 4 + [1] * 18,
        [0] * 4 + [1] * 18
    ])
    # No limits were specified.
    limits = np.array([-1] * 7)
    classrooms = np.array([[0, 0, 0, 0] + [1] * 18] * 41)

    # Random 40% free after 13:00
    for _ in range(int(41 * 0.4)):
        random.choice(classrooms)[0:2] = 1

    # Last 10 are used twice -> duplicate them.
    new_classrooms = list()
    for i in range(1, 11):
        # Unusable from 15
        classrooms[-i][8:] = 0
        # Count them again after 17
        new_classrooms.append([0] * 16 + [1] * 6)

    classrooms = np.concatenate((classrooms, np.array(new_classrooms)), axis=0)

    return cleaners, limits, classrooms


def benchmark(problem):
    scores_base = list()
    scores_hill = list()
    scores_ann = list()
    params_ann = list()

    best_base_score = float("inf")
    best_hill_score = float("inf")
    best_ann_score = float("inf")

    best_base = None
    best_hill = None
    best_ann = None
    best_ann_params = None

    solutions = problem.solutions()
    for initial_t in range(1000, 5001, 1000):
        for max_steps in range(500, 1001, 100):
            for cooling_factor in range(990, 1000, 1):
                cooling_factor = cooling_factor / 1000
                ann_params = (initial_t, max_steps, cooling_factor)
                print(f"Running with {ann_params}.")
                start = time.time()

                # Raw solution.
                solution = next(solutions)
                solution_mat = utils.dictionary_to_matrix(problem, solution)
                base_score = utils.total_working_time(solution_mat) + utils.workload_std(solution_mat)
                if base_score < best_base_score:
                    best_base_score = base_score
                    best_base = solution
                scores_base.append(base_score)

                # Hill climbing
                hill_mat = local.hill_climbing(problem, solution_mat)
                hill = utils.matrix_to_dictionary(problem, hill_mat)
                hill_score = utils.total_working_time(hill_mat) + utils.workload_std(hill_mat)
                if hill_score < best_hill_score:
                    best_hill_score = hill_score
                    best_hill = hill
                scores_hill.append(hill_score)

                # Optimize with Simulated Annealing (takes a while).
                ann_mat = local.simulated_annealing(problem, solution_mat, initial_t, max_steps, cooling_factor)
                ann = utils.matrix_to_dictionary(problem, ann_mat)
                ann_score = utils.total_working_time(ann_mat) + utils.workload_std(ann_mat)
                if ann_score < best_ann_score:
                    best_ann_params = ann_params
                    best_ann_score = ann_score
                    best_ann = ann
                params_ann.append(ann_params)
                scores_ann.append(ann_score)

                print(f"Took {time.time() - start} seconds.")

    print(f"Mean score for base solution: {np.mean(scores_base)}")
    print(f"Mean score for Hill Climbing: {np.mean(scores_hill)}")
    print(f"Mean score for Simulated Annealing: {np.mean(scores_ann)}")

    print(f"Mean score for base solution: {best_base_score}")
    utils.plot_together(problem, best_base)
    print(f"Mean score for Hill Climbing: {best_hill_score}")
    utils.plot_together(problem, best_hill)
    print(f"Mean score for Simulated Annealing: {best_ann_score}")
    print(f"Which was obtained with {best_ann_params}")
    utils.plot_together(problem, best_ann)


def main():
    cleaners, limits, classrooms = get_data()

    problem = Problem(cleaners, limits, classrooms, MAX_CONSECUTIVE_SLOTS)

    # First 1000 solutions.
    solutions = problem.solutions()
    solutions = [next(solutions) for _ in range(1000)]

    # Random base solution.
    base_solution = random.choice(solutions)
    base_solution_mat = utils.dictionary_to_matrix(problem, base_solution)

    # Optimize with Hill Climbing.
    better_solution_hill_mat = local.hill_climbing(problem, base_solution_mat)
    better_solution_hill = utils.matrix_to_dictionary(problem, better_solution_hill_mat)

    # Optimize with Simulated Annealing (takes a while).
    better_solution_ann_mat = local.simulated_annealing(problem, base_solution_mat)
    better_solution_ann = utils.matrix_to_dictionary(problem, better_solution_ann_mat)

    # Plot raw CSP solution.
    print(f"Total working time slots: {utils.total_working_time(base_solution_mat)}")
    print(f"Workload standard deviation: {utils.workload_std(base_solution_mat)}")
    print()
    # utils.plot_together(problem, base_solution)

    # Show score and Hill climbing solution.
    print("Hill Climbing results:")
    print(f"Total working time slots: {utils.total_working_time(better_solution_hill_mat)}")
    print(f"Workload standard deviation: {utils.workload_std(better_solution_hill_mat)}")
    print()
    # utils.plot_together(problem, better_solution_hill)

    # Show score and plot Annealing solution.
    print("Simulated Annealing results:")
    print(f"Total working time slots: {utils.total_working_time(better_solution_ann_mat)}")
    print(f"Workload standard deviation: {utils.workload_std(better_solution_ann_mat)}")
    print()
    # utils.plot_together(problem, better_solution_ann)

    # benchmark(problem)


# Base: 74.32 71.34 72.32
# Hill: 70.44 70.75 69.75
# Anne: 66.26 66.62 66.44

if __name__ == "__main__":
    main()
