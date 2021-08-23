import utils
import constraint
import numpy as np
import local

from common import Problem
from constraint import ExactSumConstraint, MaxSumConstraint, BacktrackingSolver


class ProblemWrapper(Problem):

    def __init__(self, workers: np.ndarray, limits: np.ndarray, classrooms: np.ndarray, max_consecutive: int):
        """
        :param workers: A numpy array containing the time slots at which each worker is available, one worker per row.
        :param limits: A numpy array containing the maximum time slots each worker can work in a day.
        :param classrooms: A numpy array containing the time slots in which each classroom can be cleaned.
        :param max_consecutive: The maximum number of consecutive time slots a worker can work.
        """
        super().__init__()

        # Infer problem size.
        self.n_classrooms, self.n_time_slots, self.n_workers = utils.problem_size(workers, limits, classrooms)

        # Copy the original data.
        self.workers = np.copy(workers)
        self.limits = np.copy(limits)
        self.classrooms = np.copy(classrooms)
        self.max_consecutive = max_consecutive

        # Make csp problem
        self.csp = get_csp_problem(workers, limits, classrooms, max_consecutive)

        # Cache existing variables.
        self.existing_variables = utils.existing_variables(workers, classrooms)

    # Get single unoptimized csp solution.
    def solution(self):
        return self.csp.getSolution()

    # Get an iterator to the raw csp solutions.
    def solutions(self):
        for solution in self.csp.getSolutionIter():
            yield solution


def get_csp_problem(workers, limits, classrooms, max_consecutive) -> constraint.Problem:
    """
    :param workers: 2D array with each row having the available time slots for each worker.
    :param limits: 1D array with the max number of hours per cleaner. Negative value means no limit.
    :param classrooms: 2D array with each row having the available time slots for each classroom.
    :param max_consecutive: The maximum number of consecutive time slots a worker can work.
    :return: The csp problem.
    """
    n_classrooms, n_time_slots, n_workers = utils.problem_size(workers, limits, classrooms)

    print(f"{utils.who()} We have: {n_time_slots} time slots, {n_workers} cleaners and {n_classrooms} classrooms.")

    problem = constraint.Problem()
    problem.setSolver(BacktrackingSolver(True))

    # I need to construct my variables.
    variables = list()
    variables_by_worker = {x: list() for x in range(n_workers)}
    variables_by_classroom = {x: list() for x in range(n_classrooms)}
    variables_by_time_slot = {x: list() for x in range(n_time_slots)}
    for c in range(n_classrooms):
        for t in range(n_time_slots):
            for k in range(n_workers):
                if workers[k][t] and classrooms[c][t]:
                    # Variable exists.
                    var = utils.num_to_var(c, t, k)
                    variables.append(var)
                    variables_by_worker[k].append(var)
                    variables_by_classroom[c].append(var)
                    variables_by_time_slot[t].append(var)

    print(f"{utils.who()} The problem has {len(variables)} variables.")

    # Ensure there are no classrooms with no variables.
    for c in range(n_classrooms):
        if not variables_by_classroom[c]:
            raise ValueError(f"Classroom {c} cannot be cleaned, problem has no solution.")

    # Variables can either be 0 or 1.
    problem.addVariables(variables, [0, 1])

    # Constraint 1: each classroom must be cleaned only once.
    for c in range(n_classrooms):
        # Ensure classroom has variables.
        if variables_by_classroom[c]:
            problem.addConstraint(ExactSumConstraint(1), variables_by_classroom[c])

    print(f"{utils.who()} Added constraints of type 1")

    # Constraint 2: no ubiquitous cleaners.
    for k in range(n_workers):
        for t in range(n_time_slots):
            possible_classrooms = set(variables_by_worker[k]).intersection(set(variables_by_time_slot[t]))
            # Ensure the cleaner can clean a certain class at a certain moment.
            if possible_classrooms:
                problem.addConstraint(MaxSumConstraint(1), list(possible_classrooms))

    print(f"{utils.who()} Added constraints of type 2")

    # Constraint 3: cleaners need a break every 4 consecutive classrooms.
    for k in range(n_workers):
        vars_by_worker = set(variables_by_worker[k])
        for x in range(0, n_time_slots - max_consecutive):
            # For each possible 5 consecutive time slots, max 4 classrooms.
            possible_classrooms = set()
            # Essentially sum over all class on the 5 consecutive time slots.
            for j in range(x, x + max_consecutive + 1):
                # Variables of this cleaner in this time slot.
                possible_classrooms = possible_classrooms.union(
                    vars_by_worker.intersection(set(variables_by_time_slot[j]))
                )
            # Ensure this is a plausible combination.
            if possible_classrooms:
                problem.addConstraint(MaxSumConstraint(max_consecutive), possible_classrooms)

    print(f"{utils.who()} Added constraints of type 3")

    # Constraint 4: cleaners have some upper total time slot limit.
    for k in range(n_workers):
        if limits[k] >= 0 and variables_by_worker[k]:
            problem.addConstraint(MaxSumConstraint(limits[k]), variables_by_worker[k])

    print(f"{utils.who()} Added constraints of type 4")

    return problem
