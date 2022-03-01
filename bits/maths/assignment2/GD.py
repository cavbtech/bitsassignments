import random
import math
from numpy import array, zeros
import copy
import seaborn as sns
import matplotlib.pyplot as plt

def random_matrix(rows, columns):
    matrix = zeros((rows, columns))
    for row_elm in range(rows):
        for col_elm in range(columns):
            number = round(random.random(), 4)
            while True:
                if not len(str(number).split('.')[-1]) == 4:
                    number = round(random.random(), 4)
                else:
                    break
            matrix[row_elm][col_elm] = number
    return matrix


def calculate_l_infinity_norm(matrix):
    greatest_val = 0
    rows, cols = matrix.shape
    for row_vec in matrix.tolist():
        if greatest_val < sum([abs(i) for i in row_vec]):
            greatest_val = sum([abs(i) for i in row_vec])
    return greatest_val


def calculate_l2_norm(matrix):
    sumup = 0
    rows, columns = matrix.shape
    for row_elm in range(rows):
        for col_elm in range(columns):
            sumup += matrix[row_elm][col_elm] ** 2
    return math.sqrt(sumup)


class GradientDescent:

    def __init__(self, A, error=1e-4):
        self.A = A
        self.b = random_matrix(A.shape[0], 1)
        self.error = error

    def generate_tau(self, gk):
        numerator = gk.transpose() @ gk
        denominator = gk.transpose() @ self.A.transpose() @ self.A @ gk
        return numerator / denominator

    def make_iterations(self):
        print(f"self.A.shape[1]={self.A.shape[1]}")
        x = random_matrix(self.A.shape[1], 1)
        print(f"x={x}")
        norm = round(calculate_l2_norm(x), 5)
        print(f"self.b={self.b}")
        print(f"self.A @ x={self.A @ x}")
        print(f"(self.A @ x - self.b)={(self.A @ x - self.b)}")
        print(f"(self.A @ x - self.b) ** 2={((self.A @ x - self.b) ** 2)}")
        print(f"0.5 * ((self.A @ x - self.b) ** 2) ={0.5 * ((self.A @ x - self.b) ** 2)}")
        function = 0.5 * ((self.A @ x - self.b) ** 2)
        function_norm = round(calculate_l2_norm(function), 5)
        iteration = 0
        x_list, fx_list = [], []
        print(f"self.A.transpose() {self.A.transpose()}")

        while abs(norm) > self.error:
            df = self.A.transpose() @ self.A @ x - self.A.transpose() @ self.b  # ∇f(x) = A⊤Ax − A⊤b.
            if (iteration == 0):
                print(f"self.A.transpose()={self.A.transpose()}")
                print(f"self.A.transpose() @ self.A={self.A.transpose() @ self.A}")
                print(f"self.A.transpose() @ self.A @ x={self.A.transpose() @ self.A @ x}")
                print(f"self.A.transpose() @ self.b={self.A.transpose() @ self.b}")
                print(f"df={df}")
            tau_value = self.generate_tau(gk=df)
            xprev = x
            x = x - tau_value * df
            x_minus_prev = x - xprev
            function = 0.5 * ((self.A @ x - self.b) ** 2)
            function_norm = round(calculate_l2_norm(function), 5)
            norm = round(calculate_l2_norm(x_minus_prev), 5)
            x_list.append(x)
            fx_list.append(function_norm)
            iteration = iteration + 1
        return x_list, fx_list, iteration


def create_plot(fx_list):
    fx_list = [i for i in fx_list if i > 0]
    x_step = list(range(len(fx_list)))
    plt.style.use('ggplot')
    plt.figure(figsize=(20, 8))
    sns.lineplot(x=x_step, y=fx_list)
    plt.xlabel('Steps')
    plt.ylabel('Cost')
    plt.title('Gradient Descent Cost Function')
    plt.show()

if __name__ == '__main__':
    matrix = random_matrix(2,3)
    print(f"matrix={matrix}")
    gd = GradientDescent(matrix)

    ##print(f"data={data}")
    #print(f"data={data[0]}, {data[1]}, {data[2]}")
