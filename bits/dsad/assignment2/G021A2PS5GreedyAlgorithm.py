def getAlphabet(i):
    """
    This method converts the input number to an alphabet.
    If the input number is under 26 then it adds suffix with 0 and
    if the input number is beyond 26 then it adds suffix as int(i/26)
    :param i:
    :return:
    """
    alphabet_A = 65
    alphabet_Z = 90
    A_Z_Span       = (alphabet_Z-alphabet_A)+1 ## it is anyway 26
    if(int(i/26)==0):
        input_alphabet = ""+chr(alphabet_A + (i % A_Z_Span))
    else:
        input_alphabet = "" + chr(alphabet_A + (i % A_Z_Span)) + str(int(i / 26))
    return input_alphabet

def merge_sort(tasks):
    """
    merge_sort, applies divide and conquer paradigm to sort recursively
    It will sorty by deadline
    :param arr:
    :return:
    """
    if len(tasks) <= 1:
        return
    # Identify the mid
    mid = len(tasks)//2
    # Identify the left subset
    left = tasks[:mid]
    # Identify the right subset
    right = tasks[mid:]

    # sort left array
    merge_sort(left)
    # sort right array
    merge_sort(right)
    # merge the sorted arrays
    merge_two_sorted_lists(left, right, tasks)

def merge_two_sorted_lists(leftTasks,rightTasks,orgTasks):
    """ merge_two_sorted_lists,two  indepedent arrays sort them and then merges them
    :param leftTasks:
    :param rightTasks:
    :param orgTasks:
    :return:
    """
    len_a = len(leftTasks)
    len_b = len(rightTasks)

    i = j = k = 0

    while i < len_a and j < len_b:
        if (leftTasks[i].bonus >= rightTasks[j].bonus):
            orgTasks[k] = leftTasks[i]
            # if(leftTasks[i].deadline==rightTasks[j].deadline):
            #     print(f"leftTasks[i]={leftTasks[i]} and rightTasks[j]={rightTasks[j]}")
            # # if(leftTasks[i].deadline < rightTasks[j].deadline and leftTasks[i].bonus < rightTasks[j].bonus):
            # #     orgTasks[k] = leftTasks[i]
            # # else:
            # #     orgTasks[k] = rightTasks[j]
            i+=1
        else:
            orgTasks[k] = rightTasks[j]
            j+=1
        k+=1

    while i < len_a:
        orgTasks[k] = leftTasks[i]
        i+=1
        k+=1

    while j < len_b:
        orgTasks[k] = rightTasks[j]
        j+=1
        k+=1

## Greedy algorithm to maximize the bonus by scheduling the problems
class Task:
    def __init__(self,name,deadline, bonus):
        self.name     = name
        self.deadline = deadline
        self.bonus    = bonus

    def __repr__(self):
        return f"name={self.name} , deadline={self.deadline} bonus={self.bonus}"
    def __str__(self):
        return f"name={self.name} , deadline={self.deadline}, bonus={self.bonus}"


class UsesCase:
    def __init__(self,use_case_name):
        self.use_case_name  = use_case_name
        self.tasks          = []
        self.assigned_tasks = []

    def __repr__(self):
        return f"use_case_name={self.use_case_name} , tasks={self.tasks}, assigned_tasks={self.assigned_tasks}"
    def __str__(self):
        return f"use_case_name={self.use_case_name} , tasks={self.tasks}, assigned_tasks={self.assigned_tasks}"

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
        self.assigned_tasks = [Task("na", 0, 0)] * (self.findMaxDeadlinedTask() + 1)
        ## order the tasks by bonus
        ## the complexity is n*log(n)
        merge_sort(self.tasks)
        for task in self.tasks:
            i = task.deadline
            # although this is a while loop, it is max of number of slots available. Hence it is
            # not considered n^2.  so, it is actually constant time.  Overall complexity is
            # n*log(n) + constant time(i.,e number of available slots).  Note: there is break applied
            while i > 0:
                if self.assigned_tasks[i].bonus == 0:
                    #assign the task
                    self.assigned_tasks[i] = task
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
                                Deadline elements count is {len(deadlinesValues)} 
                                and bonus elments count is {len(bonusValues)}""")

    usecase = UsesCase(f"""Usecase:{usecase_counter}""")
    for i in range(0,len(deadlinesValues)):
        try:
            deadline           = int(deadlinesValues[i].strip())
            bonus              = int(bonusValues[i].strip())
            if deadline < 0 or bonus < 0:
                raise ValueError
        except ValueError:
            raise RuntimeError(f" values deadline {deadlinesValues[i]} and bonus {bonusValues[i]} must be postive integers ")
        task                = Task(f"""{getAlphabet(i)}""", deadline, bonus)
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
    expected_usecases= 0
    for lineraw in file:
        ## assuming the entered data prefixed with discription
        # i.e "No of use-cases: 2" ... the program needs only the value
        ## if the input doesnt have the description then pick only the value
        if(lineraw.find(":") >0 ):
            line = (lineraw.split(":")[1]).strip()
        else:
            line = lineraw.strip()

        if(lineCounter==0 and line ==""):
            raise RuntimeError(" Please end valid number of cases as +ve Integer")
        if(line=="" and lineCounter > 1):
            continue

        #Ignore the first line
        if lineCounter == 0:
            if(line.isnumeric() and int(line)>0):
                expected_usecases = int(line)
            else:
                error_str = f"No of use-cases must be an integer.  Given {lineraw}"
                raise RuntimeError(error_str)
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
    if(len(list_use_cases) != expected_usecases):
        raise RuntimeError(f" Expected number of usecases are {expected_usecases} but actual are {len(list_use_cases)}")
    # return the usecases
    return list_use_cases


def writeToFile(list_use_cases, output_file):
    """
    writeToFile writes the list of usecases and the tasks in which they assigned
    :param list_use_cases:
    :param output_file:
    :return:
    """
    initialString       = ""
    resultCaseString    = ""
    taskOrderString     = ""

    for usecase in list_use_cases:
        total_bonus = sum(list(map(lambda task: task.bonus, usecase.assigned_tasks)))
        initialString = initialString + f"{total_bonus} \n"
        resultCaseString    =  resultCaseString+"\n"+ \
            f"For the use case {usecase.use_case_name}, " \
            f" the maximum bonus earned is {total_bonus}"

        orderString = "-->".join(list(filter(lambda task: task != "na",
                                             map(lambda task: task.name, usecase.assigned_tasks))))
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
    # for i in range(0,200):
    #     print(f"i={i} and getAlphabet(i) = {getAlphabet(i)}")
    list_use_cases = readFile("inputPS5.txt")
    writeToFile(list_use_cases,"outputPS5.txt")