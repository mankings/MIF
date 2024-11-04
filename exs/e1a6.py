import numpy as np

def main():
    # (return, risk)
    asset1 = (0.1, 0.1)
    asset2 = (0.15, 0.2)
    relation_coeff = 0.8

    # build returns vector
    r = np.array([asset1[0], asset2[0]])

    # calculate the covariance matrix
    cov_matrix = np.array([[asset1[1]**2, relation_coeff*asset1[1]*asset2[1]], [relation_coeff*asset1[1]*asset2[1], asset2[1]**2]])

    # unitary vector
    u = np.array([1, 1])

    # calculate the efficient frontier formula
    a1 = np.transpose(r) @ np.linalg.inv(cov_matrix) @ r
    a2 = np.transpose(r) @ np.linalg.inv(cov_matrix) @ u
    a3 = np.transpose(u) @ np.linalg.inv(cov_matrix) @ u

    d = a1*a3 - a2**2

    print("a1: ", a1)
    print("a2: ", a2)
    print("a3: ", a3)
    print("d: ", d)

    k1 = (a3/d)
    k2 = (-2 * a2/d)
    k3 = (a1/d)

    print(f"Final formula: {k1:.2f} e^2 + {k2:.2f} e + {k3:.2f}")

    print(f"Return of the wallet with minimum risk: {a2/a3:.2f}")

if __name__ == "__main__":
    main()