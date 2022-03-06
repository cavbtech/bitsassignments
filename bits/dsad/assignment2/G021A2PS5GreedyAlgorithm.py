## Greedy algorithm to maximize the bonus by scheduling the problems
class Task:
    def __init__(self,name,deadline, bonus):
        self.name     = name
        self.deadline = deadline
        self.bonus    = bonus

    def __str__(self) -> str:
        return f"name={self.name} , deadline={self.deadline} bonus={self.bonus}"


class UsesCase:
    def __init__(self,use_case_name):
        self.use_case_name  = use_case_name
        self.tasks          = []
        self.dailyBonus     = []

    def __str__(self) -> str:
        return f"use_case_name={self.use_case_name} , tasks={self.tasks} dailyBonus={self.dailyBonus}"

    def findMaxDeadlinedTask(self):
        """
        This method finds the max dead line for the usecase
        :return:
        """
        maxDeadline = -1
        for task in self.tasks:
            if maxDeadline < task.deadline:
                maxDeadline = task.deadline
        return maxDeadline

    def maximizeBonus(self):
        ## create an array which store the bonus that we will get each day
        ## self.dailyBonus holds the total bonus for a usecase
        self.dailyBonus = [0]*(self.findMaxDeadlinedTask()+1)
        ## order the tasks by bonus
        self.tasks.sort(key=lambda task: (task.bonus,task.deadline), reverse=True)
        for task in self.tasks:
            #check for the availability of student
            i = task.deadline
            while i > 0:
                if self.dailyBonus[i] == 0:
                    #assign the bonus
                    self.dailyBonus[i] = task.bonus
                    break
                else:
                    i -=1




def createUseCase(usecase_counter, deadlines, bonusvalues):
    """
    This method taks deadlines and bonus and creates a Usecase
    :param deadlines:
    :param bonusvalues:
    :return:
    """
    deadlinesValues = deadlines.split()
    bonusValues     = bonusvalues.split()
    if(len(deadlinesValues)!=len(bonusValues)):
        raise RuntimeError(f""" Please make sure pass same number of deadlines and bonus elements. 
                                Deadline elements counts is {len(deadlinesValues)}ma{len(bonusValues)}""")

    usecase = UsesCase(f"""Usecase:{usecase_counter}""")
    for i in range(0,len(deadlinesValues)):
        ##TODO:Handle errors for other datatype. It should only accept integers
        deadline            = int(deadlinesValues[i].strip())
        bonusvalues         = int(bonusValues[i].strip())
        task                = Task(f"""Task:{i+1}""", deadline, bonusvalues)
        usecase.tasks.append(task)

    return usecase

def readFile(fileName):
    """
    This method reads input file and creates a list of
    QuestionProperties class which holds deadline and bonus

    The assumption is the first line is just number of use cases
    and will be ignored

    Then all even lines are deadlines and all odd lines are bonus values

    At each line, the values are separated by space

    :param fileName:
    :return:
    """
    file = open(fileName, "r")
    lineCounter     = 0
    list_use_cases    = []
    deadlines        = ""
    usecase_counter  = 0
    for lineraw in file:
        ## assuming the entered data prefixed with discription
        # i.e "No of use-cases: 2" ... the program needs only the value
        ## if the input doesnt have the description then pick only the value
        if(lineraw.find(":") >0 ):
            line = lineraw.split(":")[1]
        else:
            line = lineraw

        #Ignore the first line
        if lineCounter == 0:
            lineCounter +=1
            continue
        # This is for deadlines
        if lineCounter%2 == 1:
            deadlines    = line

        if lineCounter%2 ==0:
            bonusvalues = line
            usecase_counter = usecase_counter+1
            usecase = createUseCase(usecase_counter,deadlines,bonusvalues)
            usecase.maximizeBonus()
            list_use_cases.append(usecase)

        lineCounter += 1

    file.close()
    # return the usecases
    return list_use_cases

def writeToFile(list_use_cases, output_file):
    initialString       = ""
    resultCaseString    = ""
    orderString         = ""
    taskOrderString     = ""
    for usecase in list_use_cases:
        initialString = initialString + f"{sum(usecase.dailyBonus)} \n"
        resultCaseString    =  resultCaseString+"\n"+ \
            f"For the use case {usecase.use_case_name}, " \
            f" the maximum bonus earned is {sum(usecase.dailyBonus)}"

        orderString = "-->".join(list(map(lambda task: task.name, usecase.tasks)))
        taskOrderString = taskOrderString + "\n" + \
                           f"For the use case {usecase.use_case_name}, " \
                           f" the tasks were scheduled in {orderString}"

    file = open(output_file, "w")
    file.write(initialString)
    file.write(f"Total number of test cases are  {len(list_use_cases)} \n")
    file.write(f"{resultCaseString}")
    file.write(f"{taskOrderString}")

    file.close()

if __name__ == '__main__':
    list_use_cases = readFile("inputPS52.txt")
    writeToFile(list_use_cases,"output2.txt")