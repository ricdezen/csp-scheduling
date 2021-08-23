import utils
import constraint
import csp
import numpy as np


class ProblemWrapper(object):

    def __init__(self, workers: np.ndarray, limits: np.ndarray, classrooms: np.ndarray):
        """
        :param workers: A numpy array containing the time slots at which each worker is available, one worker per row.
        :param limits: A numpy array containing the maximum time slots each worker can work in a day.
        :param classrooms: A numpy array containing the time slots in which each classroom can be cleaned.
        """
        # Infer problem size.
        self.n_classrooms, self.n_time_slots, self.n_workers = utils.problem_size(workers, limits, classrooms)

        # Copy the original arrays.
        self.workers = np.copy(workers)
        self.limits = np.copy(limits)
        self.classrooms = np.copy(classrooms)

        # Make csp problem
        self.csp = csp.get_csp_problem(workers, limits, classrooms)
