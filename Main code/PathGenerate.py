import numpy as np
from numpy import linalg as LA

# Generate a new path based on the given path, the signal path reshape matrix (p,q), 
# output length (default 200), and signal parameter m (default 2), where the output length <= q**m.
# Example: given FBM path P with length 300. Assume the signal path reshape matrix is 
# 10*30, and we want to generate a 300 length path based on the given path, the code is below:
#     Straight minus distance: model = fbmResample(path, (10,30), 0, 300, 2);
#     Normal distance: model = fbmResample(path, (10,30), 0, 300, 2);

# Reference: https://github.com/bottler/iisignature
class fbmResample:
    def __init__(self, path, path_shape, distance, output_length=200, signal_m=2):
        try:
            if distance in [1,2]:
                self.path = path
                self.path_shape = path_shape
                self.target_length = output_length
                self.sig = signal_m
                self.dist = distance
            else:
                raise TypeError('Invalid distance')
            if output_length > path_shape[1]**signal_m:
                raise ValueError('Invalid signal m')
        except TypeError as te:
            print('Please input the right distance: Straight minus distance: 1; Normal distance: 2.')
        except ValueError as ve:
            print('If the reshape matrix of path is (p,q), the output length should smaller q**m.')

#     Estimate Covariance matrix
    def cov_empirical(self, X, m, n, l):
        X_cov = np.zeros((m , m))
        for i in range(l, n-m+2):
            x_line = X[i-1 : (i+m)]
            X_cov = X_cov + np.dot(x_line.reshape(-1,1), 
                                   x_line.reshape(1,-1))
        return X_cov/(n-m-l+2)

#     Define the normal distance
    def distance_Norm(self, X_up, X_down, mean_result):
        cov_up = self.cov_empirical(X_up, len(X_up), len(X_up), 1)
        cov_down = self.cov_empirical(X_down, len(X_up), len(X_up), 1)
        eigs, eig_vers = LA.eig(cov_up - cov_down)
        cov_matrix = np.dot(eig_vers.dot(np.abs(np.diag(eigs))),
                            eig_vers.transpose())
        return np.random.multivariate_normal(mean_result, cov_matrix,1)[0]

#     Calculate the mean value of multivariate normal distribution
    def mean_Norm(self, A):
        m,n = A.shape
        mean_result = []
        for i in range(n):
            one_col = A[1:m, i] - A[:m-1, i]
            mean_result.append(np.mean(one_col))
        return np.array(mean_result)
    
#     Define the matrix Q and R with straight minus or normal distance
    def Q_R(self, A, m, n, dist_type, mean_result):
        Q,R = [],[]
        for i in range(1,n):
            if dist_type == 1:
                q_i = A[i] - A[i-1]
            elif dist_type == 2:
                q_i = self.distance_Norm(A[i-1], A[i],mean_result)
            R_j = [q_i]
            for j in range(2,m+1):
                r_j = np.kron(R_j[j-2], q_i)/j
                R_j.append(r_j)
            Q.append(q_i)
            R.append(R_j)
        return [np.asarray(Q, dtype="object"), np.asarray(R, dtype="object")]

#     Generate new path
    def sig_p(self, Q,R,m,n):
        s = [R[0].tolist()]
        for c in range(1,n-1):
            s_new = []
            for j in range(1,m+1):
                s_j = s[c-1][j-1]
                for k in range(1,j):
                    s_j = s_j + np.kron(s[c-1][k-1], R[c,j-k-1])
                s_j = s_j + R[c,j-1]
                s_new.append(s_j)
            s.append(s_new)
        if m == 1:
            return s[-1][0]
        else:
            output = []
            for i in s[-1]:
                output = output + i.reshape(-1,).tolist()
            return output

#     Change back increment into FBM path
    def back_fbm(self, A):
        A = np.insert(A,0,0)
        one_up_triangle = np.triu(np.ones(len(A)))
        return np.dot(A, one_up_triangle)
    
#     Main process to generate new path 
    def path_generate(self):
        X_all = (self.path[1:] - self.path[:-1]).reshape(self.path_shape)
        mean_result = self.mean_Norm(X_all)

        Q,R = self.Q_R(X_all, self.sig, len(X_all), self.dist, mean_result)
        Q = np.asarray(Q,dtype=object)
        R = np.asarray(R,dtype=object)
        sig_res = self.sig_p(Q, R, self.sig, len(X_all))
        fbm_new = self.back_fbm(sig_res)
        return fbm_new[:self.target_length]
