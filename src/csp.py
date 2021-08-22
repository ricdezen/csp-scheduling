import utils
import constraint
import numpy as np
from constraint import ExactSumConstraint, MaxSumConstraint, BacktrackingSolver


def get_problem(cleaners, limits, classrooms) -> constraint.Problem:
    """
    :param cleaners: 2D array with each row having the available time slots for each cleaner.
    :param limits: 1D array with the max number of hours per cleaner. Negative value means no limit.
    :param classrooms: 2D array with each row having the available time slots for each classroom.
    :return: The full problem.
    :raises: ValueError if the time slots of the cleaners are more or less than the ones for the classrooms.
             ValueError if there are not enough limits to cover the cleaners.
    """
    # Constants can be inferred from above.
    if cleaners.shape[1] != classrooms.shape[1]:
        raise ValueError(f"{cleaners.shape[1]} time slots for cleaners, but {classrooms.shape[1]} for classrooms.")
    if cleaners.shape[0] != limits.shape[0]:
        raise ValueError(f"{cleaners.shape[0]} cleaners found but {limits.shape[0]} hourly limits.")

    T = cleaners.shape[1]
    K = cleaners.shape[0]
    N = classrooms.shape[0]
    MAX_CONSECUTIVE_SLOTS = 4

    print(f"We have: {T} time slots, {K} cleaners and {N} classrooms.")

    problem = constraint.Problem()
    problem.setSolver(BacktrackingSolver(True))

    # I need to construct my variables.
    variables = list()
    variables_by_cleaner = {x: list() for x in range(K)}
    variables_by_classroom = {x: list() for x in range(N)}
    variables_by_time_slot = {x: list() for x in range(T)}
    for c in range(N):
        for t in range(T):
            for k in range(K):
                if cleaners[k][t] and classrooms[c][t]:
                    # Variable exists.
                    var = f"X_{c}_{t}_{k}"
                    variables.append(var)
                    variables_by_cleaner[k].append(var)
                    variables_by_classroom[c].append(var)
                    variables_by_time_slot[t].append(var)

    print(f"The problem has {len(variables)} variables.")

    # Ensure there are no classrooms with no variables.
    for c in range(N):
        if not variables_by_classroom[c]:
            raise ValueError(f"Classroom {c} cannot be cleaned, problem has no solution.")

    # Variables can either be 0 or 1.
    problem.addVariables(variables, [0, 1])

    # Constraint 1: each classroom must be cleaned only once.
    for c in range(N):
        # Ensure classroom has variables.
        if variables_by_classroom[c]:
            problem.addConstraint(ExactSumConstraint(1), variables_by_classroom[c])

    print(f"Added constraints of type 1")

    # Constraint 2: no ubiquitous cleaners.
    for k in range(K):
        for t in range(T):
            possible_classrooms = set(variables_by_cleaner[k]).intersection(set(variables_by_time_slot[t]))
            # Ensure the cleaner can clean a certain class at a certain moment.
            if possible_classrooms:
                problem.addConstraint(MaxSumConstraint(1), list(possible_classrooms))

    print(f"Added constraints of type 2")

    # Constraint 3: cleaners need a break every 4 consecutive classrooms.
    for k in range(K):
        vars_by_cleaner = set(variables_by_cleaner[k])
        for x in range(0, T - MAX_CONSECUTIVE_SLOTS):
            # For each possible 5 consecutive time slots, max 4 classrooms.
            possible_classrooms = set()
            # Essentially sum over all class on the 5 consecutive time slots.
            for j in range(x, x + MAX_CONSECUTIVE_SLOTS + 1):
                # Variables of this cleaner in this time slot.
                possible_classrooms = possible_classrooms.union(
                    vars_by_cleaner.intersection(set(variables_by_time_slot[j]))
                )
            # Ensure this is a plausible combination.
            if possible_classrooms:
                problem.addConstraint(MaxSumConstraint(MAX_CONSECUTIVE_SLOTS), possible_classrooms)

    print(f"Added constraints of type 3")

    # Constraint 4: cleaners have some upper total time slot limit.
    for k in range(K):
        if limits[k] >= 0 and variables_by_cleaner[k]:
            problem.addConstraint(MaxSumConstraint(limits[k]), variables_by_cleaner[k])

    print(f"Added constraints of type 4")

    return problem


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

    problem = get_problem(cleaners, limits, classrooms)

    utils.print_solution(problem.getSolution())


if __name__ == "__main__":
    main()
