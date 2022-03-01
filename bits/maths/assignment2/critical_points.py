import math



class Polymomial:

    def zeroReplace(self,replaceDigit:str):
        mobileNumberStr = str(self.origMobileNumber)
        mobileNumberStr.replace("0",replaceDigit)
        return int(mobileNumberStr)

    def generatePolynomialEq(self):
        mobileNumArr = [(self.modifiedNumber//(10**i))%10 for i in range(math.ceil(math.log(self.modifiedNumber, 10))-1, -1, -1)]
        thrdDegPol   = ["x**3","x**2*y","x*y**2","y**3","x**2","x*y","y**2","x","y"]

        finalPolEq   = ""
        for i in range(0,len(mobileNumArr) - 1):
            op = (-1)**i
            opStr = '+'
            if(op==-1):
                opStr = "-"
            else:
                opStr = "+"
            if i ==0 :
                finalPolEq = finalPolEq + (str(mobileNumArr[i])+'*'+thrdDegPol[i])
            else:
                finalPolEq = finalPolEq + opStr  + (str(mobileNumArr[i]) + '*' + thrdDegPol[i])
        finalPolEq    =  finalPolEq + "-1*"+ str(mobileNumArr[i])
        return finalPolEq

    def findCriticalPoints(self):
        pass
    def findMinMaxSaddlePoints(self):
        pass
    def __init__(self, mobileNumber:int):
        self.origMobileNumber   = mobileNumber
        self.modifiedNumber     = self.zeroReplace("3")
        self.polynomialEq       = self.generatePolynomialEq()

def display(matrix):
    for i in matrix:
        print('\t'.join(map(str, i)))

def printSaddlePionts(matrix):
    """
    >> % values of the eigenvalues will determine the type of critical points
    >> % if the eigenvalues of Hessian matrix all of them are +ve at a critical point,
    >> % the function has local minimum there. If all are negative, the function has local maxim.
    >> % if they have mixed signs, the function has saddle point and if at least one of them
    >> % 0 then the critical point is degenrate
    :param matrix:
    :return:
    """



if __name__ == '__main__':
    polymomial = Polymomial(9959551699)
    p_equation = polymomial.polynomialEq
    m_equation = p_equation.replace("**","^")
    print(f"""python polymomial.polynomialEq={p_equation}""")
    print(f"""python matlabb equation ={m_equation}""")
    print(" use matlab to calcualte the critical points. critical points for 995-955-1699 are")



