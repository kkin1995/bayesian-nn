import numpy as np
import matplotlib.pyplot as plt

class DiffEq:
    def __init__(self, N, epsilon, p0, x0):
        self.N = N
        self.epsilon = epsilon
        self.p0 = p0
        self.x0 = x0
        self.xs = np.zeros(N)
        self.xs[0] = x0
        self.ps = np.zeros(N)
        self.ps[0] = p0
        self.time_steps = np.zeros(N)


class Euler(DiffEq):
    """
    Class implementation of the Euler algorithm to numerically solve differential equations

    Parameters
    ------------

    N: int. Number of iterations
    epsilon: float. Step size
    p0: float. Initial value of the momentum
    x0: float. Initial value of the position
    U: function defining the potential energy
    dUdx: function defining the derivative of the potential energy

    Returns
    -----------

    xs: Array of position values
    ps: Array of momentum values
    """
    def __init__(self, N, epsilon, p0, x0, U, dUdx):
        super().__init__(N, epsilon, p0, x0)
        self.U = U
        self.dUdx = dUdx
        
    def run(self):
        for i in range(1, self.N):
            self.ps[i] = self.ps[i-1] - self.epsilon * self.dUdx(self.xs[i-1])
            self.xs[i] = self.xs[i-1] + self.epsilon * self.ps[i-1]
        
        return self.xs, self.ps

class ModifiedEuler(DiffEq):
    """
    Class implementation of the Modified Euler algorithm to numerically solve differential equations.
    The Modified Euler algorithm differs from the Euler algorithm in the computation of the position coordinate xs.

    Parameters
    ------------

    N: int. Number of iterations
    epsilon: float. Step size
    p0: float. Initial value of the momentum
    x0: float. Initial value of the position
    U: function defining the potential energy
    dUdx: function defining the derivative of the potential energy

    Returns
    -----------

    xs: Array of position values
    ps: Array of momentum values
    """
    def __init__(self, N, epsilon, p0, x0, U, dUdx):
        super().__init__(N, epsilon, p0, x0)
        self.U = U
        self.dUdx = dUdx
    
    def run(self):
        for i in range(1, self.N):
            self.ps[i] = self.ps[i-1] - epsilon * self.dUdx(self.xs[i-1])
            self.xs[i] = self.xs[i-1] + epsilon * self.ps[i]

        return self.xs, self.ps
 
        

if __name__ == "__main__":
    # Harmonic Potential - k, m = 1
    def U(x):
        return (1 / 2) * x**2

    def dUdx(x):
        return x

    N = 100
    epsilon = 0.1
    p0 = 0
    x0 = 1

    euler_algorithm = Euler(N, epsilon, p0, x0, U, dUdx)
    #euler_algorithm = ModifiedEuler(N, epsilon, p0, x0, U, dUdx)
    xs, ps = euler_algorithm.run()

    plt.scatter(xs, ps, s = 1)
    plt.title("Phase Space")
    plt.xlabel("x")
    plt.ylabel("p")
    plt.show()