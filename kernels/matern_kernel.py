import numpy as np

from kernels.abstract_kernel import Kernel


class MaternKernel(Kernel):
    def get_covariance_matrix(self, X: np.ndarray, Y: np.ndarray):
        """
        :param X: numpy array of size n_1 x m for which each row (x_i) is a data point at which the objective function can be evaluated
        :param Y: numpy array of size n_2 x m for which each row (y_j) is a data point at which the objective function can be evaluated
        :return: numpy array of size n_1 x n_2 for which the value at position (i, j) corresponds to the value of
        k(x_i, y_j), where k represents the kernel used.
        """
    
        ## TODO: Implement the Matern kernel
        ## Hint: You can use the Gaussian kernel as a template

        amplitude = np.exp(self.log_amplitude)
        length_scale = np.exp(self.log_length_scale)
        noise_scale = np.exp(self.log_noise_scale)

        n_1, m = X.shape
        n_2, m = Y.shape

        cov = np.zeros((n_1, n_2))

        for i in range(len(X)):
            for j in range(len(Y)):
                dist = np.linalg.norm(X[i]-Y[j])
                cov[i][j] = amplitude ** 2 * (1 + np.sqrt(3)*dist/length_scale) * np.exp(-np.sqrt(3)*dist/length_scale)

        return cov