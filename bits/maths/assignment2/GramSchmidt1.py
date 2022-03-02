import random


g_additions =0
g_multiplications = 0
g_divisions=0

def matrix_transpose(matrix_a):
    """
    Utility function to transpose a matrix
    :param matrix_a:
    :return:
    """
    global g_additions
    global g_multiplications
    global g_divisions
    a_rows = len(matrix_a)
    a_cols = len(matrix_a[0])
    result = generate_m_by_n_matrix(a_cols, a_rows)
    for i in range(a_cols):
        for j in range(a_rows):
            result[i][j] = matrix_a[j][i]
    return result

def matrix_product(matrix_a, matrix_b):
    """
     utility function to do  matrix_product
    :param matrix_a:
    :param matrix_b:
    :return:
    """
    global g_additions
    global g_multiplications
    global g_divisions
    a_rows = len(matrix_a)
    b_rows = len(matrix_b)
    a_cols = len(matrix_a[0])
    b_cols = len(matrix_b[0])
    if a_cols == b_rows:
        result = generate_m_by_n_matrix(a_rows,b_cols)
        for i in range(a_rows):
            # iterating by column by B
            for j in range(b_cols):
                # iterating by rows of B
                for k in range(b_rows):
                    result[i][j] += matrix_a[i][k] * matrix_b[k][j]
                    g_multiplications += 1
                    g_additions       += 1
        g_additions=g_additions-b_rows
    else:
        raise Exception(f" To multiply matrix_a={matrix_a}  and matrix_b ={matrix_b} matrix_a's "
                        f" columns length must be equal to matrix_b rows size.  Here, matrix_a's "
                        f" dimensions are {a_rows}X{a_cols} and matrix_b dimensions are {b_rows}X{b_cols}"
                        f" a_cols= {a_cols} and b_rows={b_rows} are not matching" )
    return result


def matrix_add(matrix_a,matrix_b,is_subtract=False):
    """
    Utility function to add two matrices
    :param matrix_a:
    :param matrix_b:
    :param is_subtract:
    :return:
    """
    global g_additions
    global g_multiplications
    global g_divisions
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
                g_additions+=1
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

def vector_add(vector_a, vector_b, is_subtract=False):
    """
    Utility function to add two vectors
    :param vector_a:
    :param vector_b:
    :param is_subtract:
    :return:
    """
    global g_additions
    global g_multiplications
    global g_divisions
    a_len = len(vector_a)
    b_len = len(vector_b)
    if(a_len == b_len):
        r_vector = generate_vector(a_len)
        for i in range(0,a_len):
            if(is_subtract):
                r_vector[i] = vector_a[i] - vector_b[i]
            else:
                r_vector[i] = vector_a[i] + vector_b[i]
            g_additions+=1
        pass
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

def set_column_vector_from_matrix(matrix_a,column_n,vector_a):
    """
    sets a column of  a matrix of mxn with vector values
    :param matrix_a:
    :param column_n:
    :param vector_a:
    :return:
    """
    global g_additions
    global g_multiplications
    global g_divisions
    a_rows = len(matrix_a)
    a_cols = len(matrix_a[0])
    if(column_n<a_cols):
        for i in range(a_rows):
            matrix_a[i][column_n] = vector_a[i]
    else:
        raise Exception(f"column_n {column_n} must be less than total number of columns{a_cols} for matrix {matrix_a}")
    return matrix_a

def matrix_diagonal_vector(matrix_a):
    """
    matrix_diagonal_vector to obtain the diagnal values of a matrix
    :param matrix_a:
    :return:
    """
    global g_additions
    global g_multiplications
    global g_divisions
    a_rows = len(matrix_a)
    a_cols = len(matrix_a[0])
    d_vector =generate_vector(a_rows)
    if(a_cols==a_rows):
        for i in range(a_rows):
            d_vector[i] = matrix_a[i][i]
    else:
        raise Exception(f"""Diagonal vector can be constructed only if the matrix is nxn matrix.
                          Given matrix is {a_rows} X {a_cols} matrix """)
    return d_vector

def vector_sign(vector_a):
    """
    get signs of a vector.  Returns 1 for +ve or 0 values and -1 for -ve values
    :param vector_a:
    :return:
    """
    global g_additions
    global g_multiplications
    global g_divisions

    a_len       = len(vector_a)
    s_vector    = generate_vector(a_len)
    for i in range(a_len):
        if(vector_a[i]>=0):
            s_vector[i] = 1
        else:
            s_vector[i] = -1
    return s_vector

def diag_vector_to_matrix(vector_a):
    """
    convert a diagnal vector to nXn matrix
    :param vector_a:
    :return:
    """
    global g_additions
    global g_multiplications
    global g_divisions
    a_len       = len(vector_a)
    a_matrix    = generate_m_by_n_matrix(a_len, a_len)
    for i in range(a_len):
            a_matrix[i][i] = vector_a[i]
    return a_matrix

def vector_dot_product(vector_a, vector_b):
    """
    Simple dot product of a vector returns a scalar
    :param vector_a:
    :param vector_b:
    :return:
    """
    global g_additions
    global g_multiplications
    global g_divisions
    a_len  = len(vector_a)
    b_len  = len(vector_b)
    d_prod = 0.0
    if(a_len==b_len):
        for i in range(0,a_len):
            d_prod += vector_a[i] * vector_b[i]
        g_multiplications+=a_len
        g_additions += a_len-1
    else:
        raise  Exception(f" vector_a={vector_a} and vector_b={vector_b} are of not same size.  Dot product"
                         f" can be performed only the size is same ")
    return d_prod

def vector_scalar_add(vector_a, scalar):
    """
    Adding a scalar to a vector
    :param vector_a:
    :param scalar:
    :return:
    """
    global g_additions
    global g_multiplications
    global g_divisions
    a_len = len(vector_a)
    for i in a_len:
        vector_a[i] += scalar
    g_additions += a_len
    return vector_a

def vector_scalar_product(vector_a,scalar):
    """
    Multiplying a scalar with a vector
    :param vector_a:
    :param scalar:
    :return:
    """
    global g_additions
    global g_multiplications
    global g_divisions
    a_len    = len(vector_a)
    r_vector = generate_vector(a_len)
    for i in range(a_len):
        r_vector[i] = vector_a[i] * scalar
    g_multiplications += a_len
    return r_vector

def matrix_scalar_product(matrix_a, scalar):
    """
    multiplying a scalar to a matrix
    :param matrix_a:
    :param scalar:
    :return:
    """
    global g_additions
    global g_multiplications
    global g_divisions
    rows = len(matrix_a)
    cols = len(matrix_a[0])
    result_matrix = generate_m_by_n_matrix(rows,cols,False)
    for i in range(0,rows):
        for j in range(0, cols):
            result_matrix[i][j] = matrix_a[i][j] * scalar
    g_multiplications += (rows*cols)
    return result_matrix



## Generates m X n matrix
def generate_m_by_n_matrix(rows, cols, is_random=False):
    """
    Generate a matrix with random values.  if is_random is False then it fills with 0.0
    :param rows:
    :param cols:
    :param is_random:
    :return:
    """
    global g_additions
    global g_multiplications
    global g_divisions
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

# Function to display matrix
def display(matrix):
    for i in matrix:
        print('\t'.join(map(str, i)))

# Function to swap needed during rank calculation for RREF pivoting
def swap(matrix, row1, row2, col):
    """
    Swap matrix rows
    :param matrix:
    :param row1:
    :param row2:
    :param col:
    :return:
    """
    global g_additions
    global g_multiplications
    global g_divisions
    for i in range(col):
        temp = matrix[row1][i]
        matrix[row1][i] = matrix[row2][i]
        matrix[row2][i] = temp
    return matrix

def copy_matrix(matrix_a):
    """
    copy matrix deep copies the input matrix.  This helps in avoiding references
    :param matrix_a:
    :return:
    """
    global g_additions
    global g_multiplications
    global g_divisions
    rows   = len(matrix_a)
    cols   = len(matrix_a[0])
    result = generate_m_by_n_matrix(rows,cols)
    for i in range(rows):
       for j in range(cols):
           result[i][j] = matrix_a[i][j]
    return result

def rank_of_matrix(matrix):
    """
    calculates the rank of a matrix using RREF
    :param matrix:
    :return:
    """
    global g_additions
    global g_multiplications
    global g_divisions
    cols     = len(matrix[0])
    rows     = len(matrix)
    rank     = cols
    lmatrix  = copy_matrix(matrix)
    for row in range(0, rank, 1):

        # Before we visit current row
        # 'row', we make sure that
        # mat[row][0],....mat[row][row-1]
        # are 0.

        # Diagonal element is not zero
        if lmatrix[row][row] != 0:
            for col in range(0, rows, 1):
                if col != row:

                    # This makes all entries of current
                    # column as 0 except entry 'mat[row][row]'
                    multiplier = (lmatrix[col][row] / lmatrix[row][row])
                    g_divisions+=1
                    for i in range(rank):
                        lmatrix[col][i] -= (multiplier * lmatrix[row][i])
                        g_multiplications += 1
                        g_additions += 1

        # Diagonal element is already zero.
        # Two cases arise:
        # 1) If there is a row below it
        # with non-zero entry, then swap
        # this row with that row and process
        # that row
        # 2) If all elements in current
        # column below mat[r][row] are 0,
        # then remove this column by
        # swapping it with last column and
        # reducing number of columns by 1.
        else:
            reduce = True

            # Find the non-zero element
            # in current column
            for i in range(row + 1, rows, 1):

                # Swap the row with non-zero
                # element with this row.
                if lmatrix[i][row] != 0:
                    lmatrix = swap(lmatrix, row, i, rank)
                    reduce  = False
                    break

            # If we did not find any row with
            # non-zero element in current
            # column, then all values in
            # this column are 0.
            if reduce:

                # Reduce number of columns
                rank -= 1

                # copy the last column here
                for i in range(0, rows, 1):
                    lmatrix[i][row] = lmatrix[i][rank]

            # process this row again
            row -= 1

    # self.Display(Matrix, self.R,self.C)
    return (rank)

def vector_frobenius_norm(vector):
    """
    Finds norm of a vector
    :param vector:
    :return:
    """
    global g_additions
    global g_multiplications
    global g_divisions
    rows = len(vector)
    fro = 0.0
    for i in range(rows):
        fro = fro + round((vector[i]**2),4)
        g_additions+=1
        g_multiplications+=1
    return fro**0.5


def frobenius_norm(matrix):
    """
    Finds the norm of a matrix
    :param matrix:
    :return:
    """
    global g_additions
    global g_multiplications
    global g_divisions
    rows = len(matrix)
    cols = len(matrix[0])
    fro = 0.0
    for i in range(rows):
        for j in range(cols):
            fro = fro + round((matrix[i][j]**2),4)
            g_additions += 1
            g_multiplications += 1
    return fro**0.5

def norm_a_minus_qr(matrix, q, r):
    """
    norm_a_minus_qr, is requirement in Q1 part 4
    :param matrix:
    :param q:
    :param r:
    :return:
    """
    global g_additions
    global g_multiplications
    global g_divisions
    qr = matrix_product(q,r)
    a_minus_qr = matrix_subtract(matrix,qr)
    # print(f" AminsQR is \n {AminsQR}")
    normOfAminsQR = frobenius_norm(a_minus_qr)
    return normOfAminsQR

def q_decomposition(matrix_a):
    """
    Requirement of Assignement Q-1 part 3
    :param matrix_a:
    :return:
    """
    global g_additions
    global g_multiplications
    global g_divisions
    rows    = len(matrix_a)
    columns = len(matrix_a[0])
    a = copy_matrix(matrix_a)
    q = generate_m_by_n_matrix(rows,rows,False) # initialize matrix Q
    u = generate_m_by_n_matrix(rows,rows,False) # initialize matrix u

    u_0 = get_column_vector_from_matrix(a,0) # assign first row to U

    u   = set_column_vector_from_matrix(u,0,u_0)
    q_0 = vector_scalar_product(u_0,  (1/vector_frobenius_norm(u_0)))
    g_divisions+=1
    q   = set_column_vector_from_matrix(q, 0, q_0)

    # print(f" what is u_0 = {u_0}")
    # print(f" what is q_0 = {q_0}")
    # print(f" what is vector_frobenius_norm(u_0)) = {vector_frobenius_norm(u_0)}")

    for i in range(1,columns):
        u_i = get_column_vector_from_matrix(a, i)
        u   = set_column_vector_from_matrix(u,i,u_i)
        for j in range(i):
            # aq is a scalar
            a_i         = get_column_vector_from_matrix(a,i)
            q_j         = get_column_vector_from_matrix(q, j)
            aq          = vector_dot_product(a_i, q_j)
            # q is vector and aq is scalar
            aqq_vector  = vector_scalar_product(q_j, aq)
            u_i        = vector_subtract(u_i, aqq_vector)
            u = set_column_vector_from_matrix(u, i, u_i)
        u_i     = get_column_vector_from_matrix(u, i)
        q_i     = vector_scalar_product(u_i,  (1/vector_frobenius_norm(u_i)))
        g_divisions += 1
        q       = set_column_vector_from_matrix(q, i, q_i)
    return q

def qr_decomposition(matrix_a,q):
    """
    Requirement of Assignment Q1 part 4
    :param matrix_a:
    :param q:
    :return:
    """
    global g_additions
    global g_multiplications
    global g_divisions
    rows    = len(matrix_a)
    columns = len(matrix_a[0])

    r = generate_m_by_n_matrix(rows,columns,False)
    for i in range(rows):
        for j in range(columns):
            a_j = get_column_vector_from_matrix(matrix, j)
            q_i = get_column_vector_from_matrix(q, i)
            r[i][j] = vector_dot_product(a_j, q_i)
    return q,r

def diag_sign(matrix_a):
    "Compute the signs of the diagonal of matrix A"
    diagonal_vector = matrix_diagonal_vector(matrix_a)
    signed_vector   = vector_sign(diagonal_vector)
    signed_matrix   = diag_vector_to_matrix(signed_vector)
    return signed_matrix

def adjust_sign(q, r):
    """
    Adjust the signs of the columns in Q and rows in R to
    impose positive diagonal of Q
    """
    d       = diag_sign(q)
    q       = matrix_product(q,d)
    r       = matrix_product(d,r)

    return q, r

## input validator to check whether entered input paramaters are +ve integers or not
def inputValidator(value):
    try:
        val = int(value)
    except ValueError:
        raise RuntimeError(f"{val} That's not an int!.  Please enter positve integer")

    if(val <=0) :
            raise RuntimeError(f" {val }That's not a positve integer. Please enter positve integer")
    return val


def total_operation_count(m,n):
    addition_Q = ((n * (m - 1) * m) / 2) + (n - 1) * m
    division_Q = m * n
    multiplication_Q = n * m ** 2
    total_operation_Q = addition_Q + division_Q + multiplication_Q
    print("\nTotal Number of Addition_Q: {}\n".format(int(addition_Q)))
    print("\nTotal Number of division_Q: {}\n".format(int(division_Q)))
    print("\nTotal Number of multiplication_Q: {}\n".format(int(multiplication_Q)))

    addition_R = (m * (m + 1) * (n - 1)) / 2
    multiplcation_R = (m * (m + 1) * n) / 2
    total_operation_R = addition_R + multiplcation_R
    print("\nTotal Number of addition_R: {}\n".format(int(addition_R)))
    print("\nTotal Number of multiplcation_R: {}\n".format(int(multiplcation_R)))

    total_operation = total_operation_Q + total_operation_R
    print("\nTotal Number of Addition, Division, Multiplication for Q & R are : {}\n".format(int(total_operation)))


## Driver
if __name__ == '__main__':
    rows       = inputValidator(input("please enter number of rows of a Matrix. It accepts only integer >0"))
    columns    = inputValidator(input("please enter number of columns of a Matrix. It accepts only integer >0"))
    if rows <= columns:
        raise RuntimeError(f"rows {rows} must be greater than columns {columns}")
    matrix = generate_m_by_n_matrix(rows,columns,True)
    #matrix = [[2.0,3.0],[4.0,3.0],[2.0,8.0]]
    fnorm  = frobenius_norm(matrix)
    rank   = rank_of_matrix(matrix)
    print("---------------------------Q1-part1 : Fnorm-----------------------------------")
    print(f"matrix=")
    display(matrix)
    print(f"Given matrix is {rows} X {columns} matrix and its Frobenius_norm = {fnorm}")

    print("---------------------------Q1-part2 :Can we apply Gram-Schmidt Algorithm?-----------------------------------")
    if(rank < columns):
        print(f"it appears that the columns are not linearly independent as number of columns \n"
              f" are {columns} and rank is {rank} and hence Gram-Schmidt Algorithm \n"
              f"cannot be applied")
    else:
        print(f"it appears that the columns are linearly independent as number of columns \n "
              f" are {columns} and rank is {rank} and \n hence Gram-Schmidt Algorithm can be "
              f"applied ")

    print("----------Q1-part3 Program is now calculating the Q and R matrices i.,e QR decomposition---------")
    q   = q_decomposition(matrix)
    q,r = qr_decomposition(matrix,q)
    q, r = adjust_sign(q, r)
    print(f"q matrix=\n")
    display(q)
    print(f"\n r=\n")
    display(r)
    print("----------Q1-part4 calculating norm of norm[Matrix - (q*r) ]---------")
    normOfAminsQR = norm_a_minus_qr(matrix,q, r)
    print(f"normOfAminsQR={normOfAminsQR}")
    print(f"Total number of additions ={g_additions}")
    print(f"Total number of multiplications ={g_multiplications}")
    print(f"Total number of divisions={g_divisions}")



    #--- operation Count
    import numpy as np
    m, n = np.array(matrix).shape
    total_operation_count(m, n)