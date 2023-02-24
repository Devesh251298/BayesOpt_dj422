from typing import Union

import numpy as np
from scipy.stats import norm

from acquisition_functions.abstract_acquisition_function import AcquisitionFunction
from gaussian_process import GaussianProcess


class ExpectedImprovement(AcquisitionFunction):
    def _evaluate(self,
                  gaussian_process: GaussianProcess,
                  data_points: np.ndarray
                  ) -> np.ndarray:
        """
        Evaluates the acquisition function at all the data points
        :param gaussian_process:
        :param data_points: numpy array of dimension n x m where n is the number of elements to evaluate
        and m is the number of variables used to calculate the objective function
        :return: a numpy array of shape n x 1 (or a float) representing the estimation of the acquisition function at
        each point
        """

        mean, std = GaussianProcess.get_gp_mean_std(gaussian_process, data_points)
        mean, std = mean.reshape(-1, 1), std.reshape(-1, 1)
        best_value = np.min(gaussian_process._array_objective_function_values)
        
        improvement = (mean - best_value) * norm.cdf((mean - best_value) / std) + std * norm.pdf((mean - best_value) / std)
    
        return -improvement

    




