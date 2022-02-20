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


if __name__ == '__main__':
    polymomial = Polymomial(9959551699)
    print(f"""polymomial.polynomialEq={polymomial.polynomialEq}""")


