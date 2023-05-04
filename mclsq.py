import numpy as np
import tensorflow as tf
import time

class BSDEOptionPricingEuropean(object):
    def __init__(self, S0, K, T, r, sigma, N, M, opt_type, degree = 3):
        if not isinstance(S0, (int, float)) or S0 <= 0:
            raise ValueError('S0 must be positive.')
        if not isinstance(K, (int, float)) or K <= 0:
            raise ValueError('K must be positive.')
        if not isinstance(T, (int, float)) or T <= 0:
            raise ValueError('T must be positive.')
        if not isinstance(r, (int, float)) or r < 0:
            raise ValueError('r must be non-negative.')
        if not isinstance(sigma, (int, float)) or sigma <= 0:
            raise ValueError('sigma must be positive.')
        if not isinstance(N, int) or N <= 0:
            raise ValueError('N must be a positive integer.')
        if not isinstance(M, int) or M <= 0:
            raise ValueError('M must be a positive integer.')
        if not isinstance(degree, int) or degree < 0:
            raise ValueError('degree must be a non-negative integer.')


        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.N = N
        self.M = M
        self.opt_type = self._get_opt_type(opt_type)
        self.degree = degree
        self.dt = T / N

    def _get_opt_type(self, opt_type):
        if not isinstance(opt_type, str):
            raise TypeError('Option type should be a string!')
        if opt_type.lower() == 'call':
            return 'call'
        elif opt_type.lower() == 'put':
            return 'put'
        else:
            raise TypeError('Invalid option type! It should be call or put')

    def _generate_stock_paths(self):
        dW = np.random.normal(0, np.sqrt(self.dt), (self.M, self.N))
        S = np.zeros((self.M, self.N + 1))
        S[:, 0] = self.S0
        S[:, 1:] = np.exp(np.cumsum((self.r - 0.5 * self.sigma ** 2) * self.dt + self.sigma * dW, axis=1))
        S *= self.S0
        return S


    def _payoff_func(self, S_T):
        if self.opt_type == 'call':
            return np.maximum(S_T - self.K, 0)
        else:
            return np.maximum(self.K - S_T, 0)

    #def _generate_regression_basis(self, S_t):
    #    X = np.zeros((S_t.shape[0], self.degree + 1))
    #    for i in range(self.degree + 1):
    #        X[:, i] = S_t ** i
    #    return X

    def _generate_regression_basis(self, S_t):
        return tf.stack([tf.ones_like(S_t, dtype=tf.float64), S_t, S_t**2, S_t**3], axis=1)


   
    def _bsde_solver(self):
        S = self._generate_stock_paths()
        Y = tf.constant(self._payoff_func(S[:, -1]), dtype=tf.float64)
        Z = tf.zeros((self.M, self.N), dtype=tf.float64)

        for t in range(self.N - 1, 0, -1):
            dt = self.T / self.N
            discount_factor = tf.exp(tf.cast(-self.r * dt, dtype=tf.float64))
            Y = Y * discount_factor

            X = self._generate_regression_basis(S[:, t])

            XtX = tf.matmul(tf.transpose(X), X)
            L = tf.cholesky(XtX)

            XtY = tf.matmul(tf.transpose(X), Y[:, tf.newaxis])
            beta_Y = tf.cholesky_solve(L, XtY)
            Y = tf.squeeze(tf.matmul(X, beta_Y))

            diff = (S[:, t] - S[:, t - 1]) / (S[:, t - 1] * self.sigma * np.sqrt(dt))
            XtZ = tf.matmul(tf.transpose(X), diff[:, tf.newaxis])
            beta_Z = tf.cholesky_solve(L, XtZ)
            Z = tf.concat([Z[:, :t - 1], tf.matmul(X, beta_Z), Z[:, t:]], axis=1)

        with tf.Session() as sess:
            Y0_np, Z0_np = sess.run([Y, Z[:, 0]])
            Y0 = np.mean(Y0_np * np.exp(-self.r * dt))
            Z0 = np.mean(Z0_np)

        return Y0, Z0



            
    def run(self):
        start_timer = time.time()
        Y0, Z0 = self._bsde_solver()
        print('Runtime: ' + str(time.time() - start_timer))
        print('Option price: ' + str(Y0))
        print('Hedge ratio: ' + str(Z0))




class BSDEOptionPricingAmerican(BSDEOptionPricingEuropean):
    def __init__(self, S0, K, T, r, sigma, N, M, opt_type, lambda_=1.0):
        super(BSDEOptionPricingAmerican, self).__init__(S0, K, T, r, sigma, N, M, opt_type)
        self.lambda_ = lambda_



    def _bsde_solver(self):
        S = self._generate_stock_paths()
        Y = tf.constant(self._payoff_func(S[:, -1]), dtype=tf.float64)
        Z = tf.zeros((self.M, self.N), dtype=tf.float64)
        K = tf.zeros_like(Y)

        for t in range(self.N - 1, 0, -1):
            dt = self.T / self.N
            discount_factor = tf.exp(tf.cast(-self.r * dt, dtype=tf.float64))
            Y = Y * discount_factor
            K = K * discount_factor

            X = self._generate_regression_basis(S[:, t])

            #Y_tf = Y[:, tf.newaxis]
            #K_tf = K[:, tf.newaxis]

            XtX = tf.matmul(tf.transpose(X), X)
            L = tf.cholesky(XtX)

            XtY = tf.matmul(tf.transpose(X), Y_tf)
            XtK = tf.matmul(tf.transpose(X), K_tf)

            beta_Y = tf.cholesky_solve(L, XtY)
            beta_K = tf.cholesky_solve(L, XtK)

            continuation_value = tf.matmul(X, beta_Y)
            K_value = tf.matmul(X, beta_K)

            exercise_value = tf.constant(self._payoff_func(S[:, t]), dtype=tf.float64)

            exercise_indices = tf.logical_and(exercise_value > tf.squeeze(continuation_value),
                                                   exercise_value > tf.squeeze(K_value))

            Y = tf.where(exercise_indices, exercise_value, Y)
            K = tf.where(exercise_indices,
                         tf.maximum(tf.squeeze(continuation_value) - exercise_value,
                                    K - self.lambda_ * tf.squeeze(K_value)),
                         K)

            diff = (S[:, t] - S[:, t - 1]) / (S[:, t - 1] * self.sigma * np.sqrt(dt))
            Z_tf = tf.where(exercise_indices, diff, tf.zeros_like(diff, dtype=tf.float64))
            Z = tf.concat([Z[:, :t - 1], Z_tf[:, tf.newaxis], Z[:, t:]], axis=1)

        with tf.Session() as sess:
            Y0_np, Z0_np = sess.run([Y, Z[:, 0]])
            Y0 = np.mean(Y0_np * np.exp(-self.r * dt))
            Z0 = np.mean(Z0_np)

        return Y0, Z0

