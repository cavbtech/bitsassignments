import random
import seaborn as sns
import matplotlib.pyplot as plt

def vector_add(vector_a, vector_b, is_subtract=False):
    """
    Utility function to add two vectors
    :param vector_a:
    :param vector_b:
    :param is_subtract:
    :return:
    """
    a_len = len(vector_a)
    b_len = len(vector_b)
    if(a_len == b_len):
        r_vector = generate_vector(a_len)
        for i in range(0,a_len):
            if(is_subtract):
                r_vector[i] = vector_a[i] - vector_b[i]
            else:
                r_vector[i] = vector_a[i] + vector_b[i]
    else:
        raise Exception(f" To add or subtract vector_a={vector_a} length {a_len} and "
                        f" vector_b={vector_b} length {b_len} should match "
                        f" they are not matching")
    return r_vector

def vector_subtract(vector_a,vector_b):
    return vector_add(vector_a,vector_b,True)


def get_column_vector_from_matrix(matrix_a,column_n):
    """
    extracts a column from a matrix of mxn
    :param matrix_a:
    :param column_n:
    :return:
    """
    global g_additions
    global g_multiplications
    global g_divisions
    a_rows = len(matrix_a)
    a_cols = len(matrix_a[0])
    col_vector =  generate_vector(a_rows, False)
    if(column_n<a_cols):
        for i in range(a_rows):
            col_vector[i] = matrix_a[i][column_n]
    else:
        raise Exception(f"column_n={column_n} must be less than total number"
                        f" of columns={a_cols} for matrix {matrix_a}")
    return col_vector


def matrix_transpose(matrix_a):
    """
    Utility function to transpose a matrix
    :param matrix_a:
    :return:
    """
    a_rows = len(matrix_a)
    a_cols = len(matrix_a[0])
    result = generate_m_by_n_matrix(a_cols, a_rows)
    for i in range(a_cols):
        for j in range(a_rows):
            result[i][j] = matrix_a[j][i]
    return result

def vector_transpose(vector_a):
    """
    Utility function to transpose a matrix
    :param vector_a:
    :return:
    """
    a_rows = len(vector_a)
    result = generate_vector(a_rows)
    for i in range(a_rows):
            result[i] = vector_a[i]
    return result


def vector_scalar_product(vector_a,scalar):
    """
    Multiplying a scalar with a vector
    :param vector_a:
    :param scalar:
    :return:
    """
    a_len    = len(vector_a)
    r_vector = generate_vector(a_len)
    for i in range(a_len):
        r_vector[i] = vector_a[i] * scalar
    return r_vector

def square_of_vector(vector_a):
    a_len    = len(vector_a)
    r_vector = generate_vector(a_len)
    for i in range(a_len):
        r_vector[i] = vector_a[i] * vector_a[i]
    return r_vector



def vector_dot_product(vector_a, vector_b):
    """
    Simple dot product of a vector returns a scalar
    :param vector_a:
    :param vector_b:
    :return:
    """
    a_len  = len(vector_a)
    b_len  = len(vector_b)
    d_prod = 0.0
    if(a_len==b_len):
        for i in range(0,a_len):
            d_prod += vector_a[i] * vector_b[i]
    else:
        raise  Exception(f" vector_a={vector_a} and vector_b={vector_b} are of not same size.  Dot product"
                         f" can be performed only the size is same ")
    return d_prod

def matrix_add(matrix_a,matrix_b,is_subtract=False):
    """
    Utility function to add two matrices
    :param matrix_a:
    :param matrix_b:
    :param is_subtract:
    :return:
    """
    a_rows = len(matrix_a)
    a_cols = len(matrix_a[0])
    b_rows = len(matrix_b)
    b_cols = len(matrix_b[0])
    if(a_rows == b_rows and a_cols==b_cols):
        r_matrix = generate_m_by_n_matrix(a_rows, a_cols)
        for i in range(a_rows):
            for j in range(a_cols):
                if(is_subtract):
                    r_matrix[i][j] = matrix_a[i][j] - matrix_b[i][j]
                else:
                    r_matrix[i][j] = matrix_a[i][j] + matrix_b[i][j]
        pass
    else:
        raise Exception(f" To add or subtract matrix_a={matrix_a} dimensions {a_rows}X{a_cols} and "
                        f" matrix_b={matrix_b} dimensions {b_rows}X{b_cols} should match "
                        f" they are not matching")
    return r_matrix

def matrix_subtract(matrix_a,matrix_b):
    """
    matrix_subtract calls matrix_add with subtract option to True
    :param matrix_a:
    :param matrix_b:
    :return:
    """
    return matrix_add(matrix_a,matrix_b,True)

def matrix_product(matrix_a, matrix_b):
    """
     utility function to do  matrix_product
    :param matrix_a:
    :param matrix_b:
    :return:
    """
    a_rows = len(matrix_a)
    b_rows = len(matrix_b)
    a_cols = len(matrix_a[0])
    flag   = False
    try:
        b_cols = len(matrix_b[0])
    except :
        ## quick fix for matrix and vector product
        b_cols    = 1
        flag      = True
    if a_cols == b_rows:
        result = generate_m_by_n_matrix(a_rows,b_cols)
        for i in range(a_rows):
            # iterating by column by B
            for j in range(b_cols):
                # iterating by rows of B
                for k in range(b_rows):
                    if( flag==True) :

                        result[i][j] += matrix_a[i][k] * matrix_b[k]
                    else:
                        result[i][j] += matrix_a[i][k] * matrix_b[k][j]
    else:
        raise Exception(f" To multiply matrix_a={matrix_a}  and matrix_b ={matrix_b} matrix_a's "
                        f" columns length must be equal to matrix_b rows size.  Here, matrix_a's "
                        f" dimensions are {a_rows}X{a_cols} and matrix_b dimensions are {b_rows}X{b_cols}"
                        f" a_cols= {a_cols} and b_rows={b_rows} are not matching" )
    return result

def generate_vector(rows, is_random=False):
    """
    Generates a vector with 0.0.  is_random is True then it generates with random values
    :param rows:
    :param is_random:
    :return:
    """
    global g_additions
    global g_multiplications
    global g_divisions
    vector = []
    for i in range(rows):
        if (is_random):
            vector.append(round(random.uniform(10.50, 50.50), 4))
        else:
            vector.append(0.0)
    return vector


def generate_m_by_n_matrix(rows, cols, is_random=False):
    """
    Generate a matrix with random values.  if is_random is False then it fills with 0.0
    :param rows:
    :param cols:
    :param is_random:
    :return:
    """

    matrix = []
    for i in range(rows):
        row = []
        for j in range(cols):
            if (is_random):
                row.append(round(random.uniform(10.50, 50.50), 4))
            else:
                row.append(0.0)
        matrix.append(row)
    return matrix


def calculate_l_infinity_norm(matrix):
    greatest_val = 0
    rows, cols = len(matrix), len(matrix[0])
    for row in range(rows):
        curr_row_abs_sum = 0
        for col in range(cols):
            curr_row_abs_sum = curr_row_abs_sum + abs(matrix[row][col])
        if greatest_val < curr_row_abs_sum:
            greatest_val = curr_row_abs_sum
    return greatest_val

def calculate_l2_norm(matrix):
    sumup = 0
    rows = len(matrix)

    cols = len(matrix[0])
    fro = 0.0
    for i in range(rows):
        for j in range(cols):
            sumup = sumup + round((matrix[i][j]**2),4)
    return fro**0.5

def calculate_l2_norm_vector(vector):
    rows = len(vector)
    sumup = 0.0
    for i in range(rows):
            sumup = sumup + round((vector[i]**2),4)
    return sumup**0.5


class GradientDescent:

    def __init__(self, matrix, error=1e-4):
        self.A   = matrix
        self.AT  = matrix_transpose(self.A)
        # For 2X3 matrix the b is going to be 2X1
        self.b   = generate_vector(len(matrix), True)
        self.error = error

    def generate_tau(self, gk):

        gk_matrix = generate_m_by_n_matrix(len(gk),1,False)
        for i in range(0,len(gk)):
            for j in range(0,1):
                gk_matrix[i][j] = gk[i]
        gkTranspose = matrix_transpose(gk_matrix)

        numerator   =  matrix_product(gkTranspose, gk_matrix)
        denominator =  matrix_product(matrix_product(matrix_product(gkTranspose,self.AT),self.A), gk_matrix)
        return numerator[0][0] / denominator[0][0]

    def make_iterations(self):
        # some randome number
        x             = generate_vector(len(matrix[0]),True)
        norm          = round(calculate_l2_norm_vector(x), 5)
        Ax            = matrix_product(self.A, x)
        vAx           = get_column_vector_from_matrix(Ax,0)
        Ax_b          = vector_subtract(vAx, self.b)
        Ax_b_sqr      = square_of_vector(Ax_b)
        function      = vector_scalar_product(Ax_b_sqr, 0.5)
        function_norm = round(calculate_l2_norm_vector(function),5)
        iteration = 0
        x_list, fx_list = [], []

        while abs(norm) > self.error:
            # ∇f(x) = A⊤Ax − A⊤b.
            Ax = matrix_product(self.A, x)
            vAx = get_column_vector_from_matrix(Ax, 0)
            ATx= matrix_product(self.AT,vAx)
            ATb= matrix_product(self.AT,self.b)
            vATx = get_column_vector_from_matrix(ATx, 0)
            vATb = get_column_vector_from_matrix(ATb, 0)
            df = vector_subtract(vATx,vATb)
            tau_value = self.generate_tau(gk=df)
            xprev        = x
            x            = vector_subtract(x,vector_scalar_product (df, tau_value))
            x_minus_prev = vector_subtract(x, xprev)
            Ax       = matrix_product(self.A, x)
            vAx      = get_column_vector_from_matrix(Ax, 0)
            Ax_b     = vector_subtract(vAx,self.b)
            Ax_b_sqr = square_of_vector(Ax_b)
            function = vector_scalar_product(Ax_b_sqr,0.5)
            function_norm = round(calculate_l2_norm_vector(function), 5)
            norm = round(calculate_l2_norm_vector(x_minus_prev), 5)

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
    print("plotting")
    plt.xlabel('Steps')
    plt.ylabel('Cost')
    plt.title('Gradient Descent Cost Function')
    plt.show()

# Function to display matrix
def display(matrix):
    for i in matrix:
        print('\t'.join(map(str, i)))

def inputValidator(value):
    try:
        val = int(value)
    except ValueError:
        raise RuntimeError(f"{val} That's not an int!.  Please enter positve integer with 4 digits")

    if(val <=0) :
            raise RuntimeError(f" {val }That's not an int!.  Please enter positve integer with 4 digits")
    return val

if __name__ == '__main__':
    last4digts     =  inputValidator(input("please enter number last 4 digits of your mobile number"))
    strLast4digts  =  str(last4digts).replace('0','3')
    last4digts     =  int(strLast4digts)
    rows           =  int(last4digts/100)
    cols           =  last4digts%100
    print(f"-------------------Given input matrix is with rows {rows} and columns {cols} ")
    matrix         =  generate_m_by_n_matrix(rows,cols,True)
    #matrix = generate_m_by_n_matrix(12, 11, True)
    display(matrix)
    l_infinity_norm= calculate_l_infinity_norm(matrix)
    print(f"for the above matrix l-infinity norm is  {l_infinity_norm}")

    print(" find the gradient descent ")
    gd             = GradientDescent(matrix)
    data = gd.make_iterations()

    create_plot(data[1])
