import random

def matrix_transpose(matrix_a):
    a_rows = len(matrix_a)
    a_cols = len(matrix_a[0])
    result = generate_m_by_n_matrix(a_cols, a_rows)
    for i in range(a_cols):
        for j in range(a_rows):
            result[i][j] = matrix_a[j][i]
    return result

def matrix_product(matrix_a, matrix_b):
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
    else:
        raise Exception(f" To multiply matrix_a={matrix_a}  and matrix_b ={matrix_b} matrix_a's "
                        f" columns length must be equal to matrix_b rows size.  Here, matrix_a's "
                        f" dimensions are {a_rows}X{a_cols} and matrix_b dimensions are {b_rows}X{b_cols}"
                        f" a_cols= {a_cols} and b_rows={b_rows} are not matching" )
    return result


def matrix_add(matrix_a,matrix_b,is_subtract=False):
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
    return matrix_add(matrix_a,matrix_b,True)

def vector_add(vector_a, vector_b, is_subtract=False):
    a_len = len(vector_a)
    b_len = len(vector_b)
    if(a_len == b_len):
        r_vector = generate_vector(a_len)
        print(f"vector_a={vector_a} and vector_b = {vector_b}")
        for i in range(0,a_len):
            if(is_subtract):
                r_vector[i] = vector_a[i] - vector_b[i]
            else:
                r_vector[i] = vector_a[i] + vector_b[i]
        pass
    else:
        raise Exception(f" To add or subtract vector_a={vector_a} length {a_len} and "
                        f" vector_b={vector_b} length {b_len} should match "
                        f" they are not matching")
    return r_vector

def vector_subtract(vector_a,vector_b):
    return vector_add(vector_a,vector_b,True)

def get_column_vector_from_matrix(matrix_a,column_n):
    a_rows = len(matrix_a)
    a_cols = len(matrix_a[0])
    col_vector =  generate_vector(a_rows, False)
    if(column_n<a_cols):
        for i in range(a_rows):
            col_vector[i] = matrix_a[i][column_n]
    else:
        raise Exception(f"column_n {column_n} must be less than total number of columns{a_cols} for matrix {matrix_a}")
    return col_vector

def set_column_vector_from_matrix(matrix_a,column_n,vector_a):
    a_rows = len(matrix_a)
    a_cols = len(matrix_a[0])
    if(column_n<a_cols):
        for i in range(a_rows):
            print(f"for i={i} and matrix_a[i][column_n]  ={matrix_a[i][column_n]} and vector_a[i] ={vector_a[i]}")
            matrix_a[i][column_n] = vector_a[i]
    else:
        raise Exception(f"column_n {column_n} must be less than total number of columns{a_cols} for matrix {matrix_a}")
    return matrix_a

def matrix_diagonal_vector(matrix_a):
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
    a_len       = len(vector_a)
    s_vector    = generate_vector(a_len)
    for i in range(a_len):
        if(vector_a[i]>=0):
            s_vector[i] = 1
        else:
            s_vector[i] = -1
    return s_vector

def diag_vector_to_matrix(vector_a):
    a_len       = len(vector_a)
    a_matrix    = generate_m_by_n_matrix(a_len, a_len)
    for i in range(a_len):
            a_matrix[i][i] = vector_a[i]
    return a_matrix

def vector_dot_product(vector_a, vector_b):
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

def vector_scalar_add(vector_a, scalar):
    a_len = len(vector_a)
    for i in a_len:
        vector_a[i] += vector_a[i] + scalar
    return vector_a

def vector_scalar_product(vector_a,scalar):
    a_len    = len(vector_a)
    r_vector = generate_vector(a_len)
    for i in range(a_len):
        r_vector[i] = vector_a[i] * scalar
    return r_vector

def matrix_scalar_product(matrix_a, scalar):
    rows = len(matrix_a)
    cols = len(matrix_a[0])
    result_matrix = generate_m_by_n_matrix(rows,cols,False)
    for i in range(0,rows):
        for j in range(0, cols):
            result_matrix[i][j] = matrix_a[i][j] * scalar
    return result_matrix



## Generates m X n matrix
def generate_m_by_n_matrix(rows, cols, is_random=False):
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
    vector = []
    for i in range(rows):
        if (is_random):
            vector.append(round(random.uniform(10.50, 50.50), 4))
        else:
            vector.append(0.0)
    return vector

# Function to display matrix
def Display(matrix, row, col):
    for i in range(row):
        for j in range(col):
            print(" " + str(matrix[i][j]))
        print('\n')

# Function to swap needed during rank calculation for RREF pivoting
def swap(matrix, row1, row2, col):
    for i in range(col):
        temp = matrix[row1][i]
        matrix[row1][i] = matrix[row2][i]
        matrix[row2][i] = temp
    return matrix

def copy_matrix(matrix_a):
    rows   = len(matrix_a)
    cols   = len(matrix_a[0])
    result = generate_m_by_n_matrix(rows,cols)
    for i in range(rows):
       for j in range(cols):
           result[i][j] = matrix_a[i][j]
    return result

## calculates the rank of a matrix using RREF
def rank_of_matrix(matrix):
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
                    for i in range(rank):
                        lmatrix[col][i] -= (multiplier * lmatrix[row][i])

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
    rows = len(vector)
    fro = 0.0
    for i in range(rows):
            fro = fro + round((vector[i]**2),4)
    return fro**0.5


def frobenius_norm(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    fro = 0.0
    for i in range(rows):
        for j in range(cols):
            fro = fro + round((matrix[i][j]**2),4)
    return fro**0.5

def norm_a_minus_qr(matrix, q, r):
    qr = matrix_product(q,r)
    print(f" QR =\n{qr} and \n matrix=\n{matrix}")
    a_minus_qr = matrix_subtract(matrix,qr)
    # print(f" AminsQR is \n {AminsQR}")
    normOfAminsQR = frobenius_norm(a_minus_qr)
    return normOfAminsQR

def q_decomposition(matrix_a):
    rows    = len(matrix_a)
    columns = len(matrix_a[0])
    a = copy_matrix(matrix_a)
    print(f"copied matrix a={a}")
    q = generate_m_by_n_matrix(rows,rows,False) # initialize matrix Q
    u = generate_m_by_n_matrix(rows,rows,False) # initialize matrix u

    u_0 = get_column_vector_from_matrix(a,0) # assign first row to U

    u   = set_column_vector_from_matrix(u,0,u_0)
    q_0 = vector_scalar_product(u_0,  (1/vector_frobenius_norm(u_0)))
    q   = set_column_vector_from_matrix(q, 0, q_0)

    print(f" what is u_0 = {u_0}")
    print(f" what is q_0 = {q_0}")
    print(f" what is vector_frobenius_norm(u_0)) = {vector_frobenius_norm(u_0)}")

    for i in range(1,rows-1):
        u_i = get_column_vector_from_matrix(a, i)
        u   = set_column_vector_from_matrix(u,i,u_i)
        for j in range(i):
            # aq is a scalar
            a_i         = get_column_vector_from_matrix(a,i)
            q_j         = get_column_vector_from_matrix(q, j)
            print(f" what is u_i = {u_i}")
            print(f" what is a_i = {a_i}")
            print(f" what is q_j = {q_j}")
            aq          = vector_dot_product(a_i, q_j)
            print(f"aq={aq}")
            # q is vector and aq is scalar
            aqq_vector  = vector_scalar_product(q_j, aq)
            print(f"aqq_vector={aqq_vector}")
            u_i        = vector_subtract(u_i, aqq_vector)
            u = set_column_vector_from_matrix(u, i, u_i)
            print(f"before u_i={u_i}")
            print(f"before u={u}")
        u_i     = get_column_vector_from_matrix(u, i)
        print(f"after u_i={u_i}")
        q_i     = vector_scalar_product(u_i,  (1/vector_frobenius_norm(u_i)))
        q       = set_column_vector_from_matrix(q, i, q_i)
        print(f"q[i]={q[i]}")
    return q

def qr_decomposition(matrix_a,q):
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

## Driver
if __name__ == '__main__':
    rows       = inputValidator(input("please enter number of rows of a Matrix. It accepts only integer >0"))
    columns    = inputValidator(input("please enter number of columns of a Matrix. It accepts only integer >0"))
    if rows <= columns:
        raise RuntimeError(f"rows {rows} must be greater than columns {columns}")
    #matrix = generate_m_by_n_matrix(rows,columns,True)
    matrix = [[2.0,3.0],[4.0,3.0],[2.0,8.0]]
    fnorm  = frobenius_norm(matrix)
    rank   = rank_of_matrix(matrix)
    print("---------------------------Fnorm-----------------------------------")
    print(f"matrix={matrix}")
    print(f"Given matrix is {rows} X {columns} matrix and its Frobenius_norm = {fnorm}")

    print("---------------------------Can we apply Gram-Schmidt Algorithm?-----------------------------------")
    if(rank < columns):
        print(f"it appears that the columns are not linearly independent as number of columns \n"
              f" are {columns} and rank is {rank} and hence Gram-Schmidt Algorithm \n"
              f"cannot be applied")
    else:
        print(f"it appears that the columns are linearly independent as number of columns \n "
              f" are {columns} and rank is {rank} and \n hence Gram-Schmidt Algorithm can be "
              f"applied ")

    print("----------Program is now calculating the Q and R matrices i.,e QR decomposition---------")
    q   = q_decomposition(matrix)
    q,r = qr_decomposition(matrix,q)
    print(f"q matrix=\n{q} and \n r=\n{r}")
    q,r = adjust_sign(q,r)
    print(f"after applying the signs q matrix=\n{q} and \n r=\n{r}")



    normOfAminsQR = norm_a_minus_qr(matrix,q, r)
    print(f"normOfAminsQR={normOfAminsQR}")


