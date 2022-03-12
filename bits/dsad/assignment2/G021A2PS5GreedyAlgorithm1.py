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
        self.pickuptasks    = []

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
        """
        maximizeBonus identifies the max of tasks that can be picked up
        based on the max deadlines available. Thus max bonus can be calculated as sum of
        all tasks that were picked up.
        It first sorts all the tasks based on the weightage which is bonus / deadline
        :return:
        """

        ## order the tasks by bonus /  deadline
        self.tasks.sort(key=lambda task: (task.bonus), reverse=True)
        # calcualte the max deadlined task
        max_deadline = self.findMaxDeadlinedTask()
        total_bonus  = 0
        # remaining deadline is always inclusive of the last deadline hence it is max deadline + 1
        remaining_deadline = max_deadline+1
        i = 0
        for i in range(0,max_deadline):
            if remaining_deadline >0 :
                self.pickuptasks.append(self.tasks[i])
                total_bonus+=self.tasks[i].bonus
                remaining_deadline-=self.tasks[i].deadline




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
                                Deadline elements count is {len(deadlinesValues)} 
                                and bonus elments count is {len(bonusValues)}""")

    usecase = UsesCase(f"""Usecase:{usecase_counter}""")
    for i in range(0,len(deadlinesValues)):
        ##TODO:Handle errors for other datatype. It should only accept integers
        deadline    =0
        bonus       =0
        try:
            deadline           = int(deadlinesValues[i].strip())
            bonus              = int(bonusValues[i].strip())
            if deadline < 0 or bonus < 0:
                raise ValueError
        except ValueError:
            raise RuntimeError(f" values deadline {deadlinesValues[i]} and bonus {bonusValues[i]} must be postive integers ")
        task                = Task(f"""Task:{i+1}""", deadline, bonus)
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
    taskOrderString     = ""
    for usecase in list_use_cases:

        total_bonus = sum(list(map(lambda task: task.bonus, usecase.pickuptasks)))
        initialString = initialString + f"{total_bonus} \n"
        resultCaseString    =  resultCaseString+"\n"+ \
            f"For the use case {usecase.use_case_name}, " \
            f" the maximum bonus earned is {total_bonus}"

        orderString = "-->".join(list(map(lambda task: task.name, usecase.pickuptasks)))
        taskOrderString = taskOrderString + "\n" + \
                           f"For the use case {usecase.use_case_name}, " \
                           f" the tasks were scheduled in {orderString}"

    # print(initialString)
    # print(f"Total number of test cases are  {len(list_use_cases)} \n")
    # print(f"{resultCaseString}")
    # print(f"{taskOrderString}")

    file = open(output_file, "w")
    file.write(initialString)
    file.write(f"Total number of test cases are  {len(list_use_cases)} \n")
    file.write(f"{resultCaseString}")
    file.write(f"{taskOrderString}")

    file.close()

if __name__ == '__main__':
    list_use_cases = readFile("inputPS5.txt")
    #print(f"list_use_cases = {list_use_cases}")
    writeToFile(list_use_cases,"outputPS5.txt")