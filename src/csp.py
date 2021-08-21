import re
import constraint

from constraint import ExactSumConstraint, MaxSumConstraint


def print_solution(solution):
    selected = [var for var in solution if solution[var]]
    for var in selected:
        classroom, time, cleaner = re.search(r"X_(\d+)_(\d+)_(\d+)", var).groups()
        print(f"Classroom {classroom} is going to be cleaned at hour {time} by {cleaner}")


def main():
    cleaners = [
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1]
    ]
    classrooms = [
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1]
    ]

    # Constants can be inferred from above.
    T = 4
    K = 4
    N = 4

    problem = constraint.Problem()

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

    # Each cleaner can only clean a single classroom at any given moment.
    for k in range(K):
        for t in range(T):
            possible_classrooms = set(variables_by_cleaner[k]).intersection(set(variables_by_time_slot[t]))
            # Ensure the cleaner can clean a certain class at a certain moment.
            if possible_classrooms:
                problem.addConstraint(MaxSumConstraint(1), list(possible_classrooms))

    for sol in problem.getSolutions():
        print_solution(sol)
        print("")


if __name__ == "__main__":
    main()
