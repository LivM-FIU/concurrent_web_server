import numpy as np

class PureALS:
    """
    Pure-Python ALS implementation (NumPy only).
    Compatible with implicit.als.AlternatingLeastSquares API subset.
    """

    def __init__(self, factors=64, regularization=0.08, iterations=20, random_state=42):
        self.factors = factors
        self.reg = regularization
        self.iterations = iterations
        self.random_state = random_state

        self.user_factors = None
        self.item_factors = None

    def fit(self, matrix):
        """
        Train ALS using alternating updates:
        matrix: CSR matrix items × users
        """
        np.random.seed(self.random_state)

        num_items, num_users = matrix.shape

        # Initialize latent factors
        self.item_factors = np.random.normal(0, 0.01, (num_items, self.factors))
        self.user_factors = np.random.normal(0, 0.01, (num_users, self.factors))

        # Precompute identity
        I = np.eye(self.factors)

        # For fast CSR access
        matrix_csr = matrix.tocsr()
        matrix_csc = matrix.tocsc()

        for _ in range(self.iterations):
            # Update user factors
            for u in range(num_users):
                start, end = matrix_csc.indptr[u], matrix_csc.indptr[u + 1]
                item_ids = matrix_csc.indices[start:end]
                ratings = matrix_csc.data[start:end]

                if len(item_ids) == 0:
                    continue

                V = self.item_factors[item_ids]
                A = V.T @ V + self.reg * I
                b = V.T @ ratings
                self.user_factors[u] = np.linalg.solve(A, b)

            # Update item factors
            for i in range(num_items):
                start, end = matrix_csr.indptr[i], matrix_csr.indptr[i + 1]
                user_ids = matrix_csr.indices[start:end]
                ratings = matrix_csr.data[start:end]

                if len(user_ids) == 0:
                    continue

                U = self.user_factors[user_ids]
                A = U.T @ U + self.reg * I
                b = U.T @ ratings
                self.item_factors[i] = np.linalg.solve(A, b)

        return self
