# File for defining the RQE solution concept
import numpy as np
from scipy.special import logsumexp


class RQE:
    def __init__(
        self,
        tau1=1.0,
        tau2=1.0,
        epsilon1=1.0,
        epsilon2=1.0,
        lr=0.1,
        max_iter=500,
        br_iters=50,
    ):
        self.tau1 = tau1
        self.tau2 = tau2
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.lr = lr
        self.max_iter = max_iter
        self.br_iters = br_iters
        self.EPS = 1e-12

    def project_simplex(self, p):
        p = np.asarray(p)
        if np.sum(p) == 1 and np.all(p >= 0):
            return p
        sorted_p = np.sort(p)[::-1]
        tmp_sum = 0.0
        for i in range(len(p)):
            tmp_sum += sorted_p[i]
            t = (tmp_sum - 1) / (i + 1)
            if i == len(p) - 1 or sorted_p[i + 1] <= t:
                break
        theta = t
        return np.maximum(p - theta, 0)

    def entropic_risk(self, pi_i, R_i, pi_minus_i, tau):
        logits = -(R_i @ pi_minus_i) * tau
        log_sum = logsumexp(logits)
        return log_sum / tau

    def solve_rqe_single(self, R_i, pi_minus_i, tau, epsilon):
        A_i = R_i.shape[0]
        pi_i = np.ones(A_i) / A_i

        for _ in range(self.max_iter):
            logits = -(R_i @ pi_minus_i) * tau
            denom = logsumexp(logits)
            grad_risk = -(R_i @ pi_minus_i) * np.exp(logits - denom)
            grad_entropy = np.log(np.clip(pi_i, self.EPS, 1)) + 1
            grad = grad_risk + epsilon * grad_entropy

            pi_i -= self.lr * grad
            pi_i = self.project_simplex(pi_i)

        return pi_i

    def compute_rqe(self, R1, R2):
        A1, A2 = R1.shape
        pi1 = np.ones(A1) / A1
        pi2 = np.ones(A2) / A2

        for _ in range(self.br_iters):
            pi1 = self.solve_rqe_single(R1, pi2, tau=self.tau1, epsilon=self.epsilon1)
            pi2 = self.solve_rqe_single(R2.T, pi1, tau=self.tau2, epsilon=self.epsilon2)

        return pi1, pi2


if __name__ == "__main__":
    # Example usage
    R1 = np.array([[1, -1], [-1, 1]])
    R2 = np.array([[-1, 1], [1, -1]])

    rqe_solver = RQE(tau1=1.0, tau2=1.0, epsilon1=0.01, epsilon2=0.01)
    pi1, pi2 = rqe_solver.compute_rqe(R1, R2)

    print("Computed policies:")
    print("Player 1:", pi1)
    print("Player 2:", pi2)
