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
        mean = mean.reshape(-1,1)
        std = std.reshape(-1, 1)
        ## calculate the best value
        best_value = np.min(gaussian_process._array_objective_function_values)
        
        # if the variance is 0, we set the improvement to 0
        gamma = (best_value-mean)/std

        return gamma*std*norm.cdf(gamma) + std*norm.pdf(gamma)

    




