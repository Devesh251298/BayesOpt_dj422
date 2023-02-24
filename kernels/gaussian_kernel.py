import numpy as np

from kernels.abstract_kernel import Kernel


class GaussianKernel(Kernel):
    def __init__(self,
                 log_amplitude: float,
                 log_length_scale: float,
                 log_noise_scale: float,
                 ):
        super(GaussianKernel, self).__init__(log_amplitude,
                                             log_length_scale,
                                             log_noise_scale,
                                             )

    def get_covariance_matrix(self,
                              X: np.ndarray,
                              Y: np.ndarray,
                              ) -> np.ndarray:
        """
        :param X: numpy array of size n_1 x m for which each row (x_i) is a data point at which the objective function can be evaluated
        :param Y: numpy array of size n_2 x m for which each row (y_j) is a data point at which the objective function can be evaluated
        :return: numpy array of size n_1 x n_2 for which the value at position (i, j) corresponds to the value of
        k(x_i, y_j), where k represents the kernel used.
        """
        amplitude = np.exp(self.log_amplitude)
        length_scale = np.exp(self.log_length_scale)
        noise_scale = np.exp(self.log_noise_scale)

        print(X.shape, Y.shape)

        n_1, m = X.shape
        n_2, m = Y.shape

        # Compute the squared distance matrix
        X_sq = np.sum(X ** 2, axis=1).reshape(n_1, 1)
        Y_sq = np.sum(Y ** 2, axis=1).reshape(1, n_2)
        dist_sq = X_sq + Y_sq - 2 * np.dot(X, Y.T)

        # Compute the covariance matrix
        cov = amplitude ** 2 * np.exp(-0.5 * dist_sq / length_scale ** 2)

        # Add the noise
        # cov += noise_scale ** 2 * np.eye(n_1, n_2)

        return cov

    def __call__(self,
                 X: np.ndarray,
                 Y: np.ndarray,
                 ) -> np.ndarray:
        return self.get_covariance_matrix(X, Y)

