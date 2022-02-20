import random
import numpy as np
class GramSchmidt:

    def normOperationsCount(self,matA):
        if matA.ndim == 1:
            rows, columns = matA.shape[0],1
        else:
            rows,columns = matA.shape
        self.multiplactions = self.multiplactions + rows*columns
        self.additions      = self.additions + rows*columns-1

    def matrixMultiplicationCount(self,matA,matB):
        if matA.ndim == 1:
            matAR,matAC = matA.shape[0],1
        else:
            matAR,matAC = matA.shape

        if matB.ndim == 1:
            matBR,matBC = matB.shape[0],1
        else:
            matBR,matBC = matB.shape

        if matAC == matBR:
            self.multiplactions = self.multiplactions + matAR*matAC*matBC
            self.additions      = self.additions + matAR*matAC*(matBC - 1)

    def norm_a_minus_qr(self, Q, R):
        QR = Q @ R
        # print(f" QR =\n{QR} and \n gramSchidt.matrix=\n{gramSchidt.matrix}")
        AminsQR = np.subtract(np.around(self.matrix, 4), np.around(QR, 4))
        # print(f" AminsQR is \n {AminsQR}")
        normOfAminsQR = np.linalg.norm(AminsQR)
        return normOfAminsQR

    def diag_sign(self, v):
        "Compute the signs of the diagonal of matrix A"
        print(f"v={v}")
        print(f"np.diag(v)={np.diag(v)}")
        print(f"np.sign(np.diag(v))={np.sign(np.diag(v))}")
        print(f"np.diag(np.sign(np.diag(v)))={np.diag(np.sign(np.diag(v)))}")
        D = np.diag(np.sign(np.diag(v)))
        return D

    def adjust_sign(self,Q, R):
        """
        Adjust the signs of the columns in Q and rows in R to
        impose positive diagonal of Q
        """
        D = self.diag_sign(Q)
        Q[:, :] = Q @ D
        self.matrixMultiplicationCount(Q, D)
        R[:, :] = D @ R
        self.matrixMultiplicationCount(D,R)
        return Q, R

    def QR_Decomposition(self):
        ##A    = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        A    = self.matrix
        n, m = A.shape  # get the shape of A

        Q = np.empty((n, n))  # initialize matrix Q
        u = np.empty((n, n))  # initialize matrix u


        u[:, 0] = A[:, 0]
        Q[:, 0] = u[:, 0] / np.linalg.norm(u[:, 0])
        ## every time norm is calucated for a vector of n size
        # there will be n multiplications and n-1 additions i.,e
        self.normOperationsCount(u[:, 0])
        self.divisions = self.divisions + n
        print(f" what is A[:, 0] = {A[:, 0]}")
        print(f" what is Q[:, 0] = {Q[:, 0]}")
        print(f" what is np.linalg.norm(u[:, 0]) = {np.linalg.norm(u[:, 0])}")
        for i in range(1, n-1):
            u[:, i] = A[:, i]
            for j in range(i):
                print(f" what is u[:, i] = {u[:, i]}")
                print(f" what is A[:, i] = {A[:, i]}")
                print(f" what is Q[:, j] = {Q[:, j]}")
                print(f" what is (A[:, i] @ Q[:, j]) = {(A[:, i] @ Q[:, j])}")
                print(f" what is (A[:, i] @ Q[:, j]) * Q[:, j] = {(A[:, i] @ Q[:, j]) * Q[:, j]}")
                u[:, i] -= (A[:, i] @ Q[:, j]) * Q[:, j]  # get each u vector
                print(f" after what is u[:, i] = {u[:, i]}")

            Q[:, i] = u[:, i] / np.linalg.norm(u[:, i])  # compute each e vetor
            self.normOperationsCount(u[:, 0])
            self.divisions = self.divisions + n

        R = np.zeros((n, m))
        print(f"-------------------------while finding R------------")
        for i in range(n):
            for j in range(i, m):
                print(f"what A[:, j]={A[:, j]}")
                print(f"what Q[:, i]={Q[:, i]}")
                print(f"what A[:, j] @ Q[:, i]={A[:, j] @ Q[:, i]}")
                R[i, j] = A[:, j] @ Q[:, i]
                self.matrixMultiplicationCount(A[:, j],Q[:, i])

        return Q, R

    def __generateMbyNMaxtrx(self):
        matrix = []
        for i in range(self.m):
            row = []
            for j in range(self.n):
                row.append(round(random.uniform(10.50, 50.50), 4))
            matrix.append(row)
        return np.array(matrix)

    # def __frobeniusNorm(self):
    #     fro = 0.0
    #     for i in range(self.m):
    #         for j in range(self.n):
    #             fro = fro + round((self.matrix[i][j]**2),4)
    #     return fro**0.5

    def __init__(self,m,n):
        if m <= n:
            raise  RuntimeError(f"rows {m} must be greater than columns {n}")
        self.m = m
        self.n = n
        self.matrix = self.__generateMbyNMaxtrx()
        self.fnom   = np.linalg.norm(self.matrix)
        self.rank   = np.linalg.matrix_rank(self.matrix)
        self.additions = 0
        self.multiplactions = 0
        self.divisions = 0
    def __init__(self,matrix_a):
        self.m = len(matrix_a)
        self.n = len(matrix_a[0])
        if self.m <= self.n:
            raise  RuntimeError(f"rows {self.m} must be greater than columns {self.n}")
        self.matrix = np.array(matrix_a)
        self.fnom   = np.linalg.norm(self.matrix)
        self.rank   = np.linalg.matrix_rank(self.matrix)
        self.additions = 0
        self.multiplactions = 0
        self.divisions = 0

def inputValidator(value):
    try:
        val = int(value)
    except ValueError:
        raise RuntimeError(f"{val} That's not an int!.  Please enter positve integer")

    if(val <=0) :
            raise RuntimeError(f" {val }That's not a positve integer. Please enter positve integer")
    return val


if __name__ == '__main__':
    rows       = inputValidator(input("please enter number of rows of a Matrix. It accepts only integer >0"))
    columns    = inputValidator(input("please enter number of columns of a Matrix. It accepts only integer >0"))
    #gramSchidt = GramSchmidt(rows,columns)
    gramSchidt = GramSchmidt([[2.0,3.0],[4.0,3.0],[2.0,8.0]])
    print("---------------------------Fnorm-----------------------------------")
    print(f"matrix={gramSchidt.matrix}")
    print(f"Given matrix is {rows} X {columns} matrix and its Fnorm = {gramSchidt.fnom}")
    print("---------------------------Can we apply Gram-Schmidt Algorithm?-----------------------------------")
    if(gramSchidt.rank < columns):
        print(f"it appears that the columns are not linearly independent as number of columns \n"
              f" are {columns} and rank is {gramSchidt.rank} and hence Gram-Schmidt Algorithm \n"
              f"cannot be applied")
    else:
        print(f"it appears that the columns are linearly independent as number of columns \n "
              f" are {columns} and rank is {gramSchidt.rank} and \n hence Gram-Schmidt Algorithm can be "
              f"applied ")

    print("----------Program is now calculating the Q and R matrices i.,e QR decomposition---------")
    Q,R = gramSchidt.QR_Decomposition()
    Q,R = gramSchidt.adjust_sign(Q,R)

    print(f"After QR decomposition \n Q is \n {Q} and \n R is \n {R}")

    print(f"-----------------calculating the norm for [A-(Q.R)]-----------------------------")
    normOfAminsQR   =   gramSchidt.norm_a_minus_qr(Q,R)
    print(f" normOfAminsQR norm[A-(Q.R)]= {normOfAminsQR}")
    print(f" number of operations "
          f"\n Total additions       = {gramSchidt.additions}"
          f"\n Total multiplications = {gramSchidt.multiplactions} "
          f"\n Total divisions       = {gramSchidt.divisions} ")




