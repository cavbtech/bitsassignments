class TaskHeapQueue:

    def __init__(self):
        self.H      = []
        self.size   = -1

    # Function to return the index of the
    # parent node of a given node
    def parent(self,i):
        return (i - 1) // 2


    # Function to return the index of the
    # left child of the given node
    def leftChild(self,i):
        return ((2 * i) + 1)


    # Function to return the index of the
    # right child of the given node
    def rightChild(self,i):
        return ((2 * i) + 2)


    # Function to shift up the
    # node in order to maintain
    # the heap property
    def shiftUp(self,i):
        while (i > 0 and self.H[self.parent(i)][0] < self.H[i][0]):
            # Swap parent and current node
            self.swap(self.parent(i), i)

            # Update i to parent of i
            i = self.parent(i)


    # Function to shift down the node in
    # order to maintain the heap property
    def shiftDown(self,i):
        maxIndex = i

        # Left Child
        l = self.leftChild(i)

        if (l <= self.size and self.H[l][0] > self.H[maxIndex][0]):
            maxIndex = l

        # Right Child
        r = self.rightChild(i)

        if (r <= self.size and self.H[r][0] > self.H[maxIndex][0]):
            maxIndex = r

        # If i not same as maxIndex
        if (i != maxIndex):
            self.swap(i, maxIndex)
            self.shiftDown(maxIndex)


    # Function to insert a
    # new element in
    # the Binary Heap
    def insert(self,p):
        self.size = self.size + 1
        self.H.append(p)

        # Shift Up to maintain
        # heap property
        self.shiftUp(self.size)
        print(f"self.H=${self.H}")


    # Function to extract
    # the element with
    # maximum priority
    def extractMax(self):
        result = self.H[0]

        # Replace the value
        # at the root with
        # the last leaf
        self.H[0] = self.H[self.size]
        self.size = self.size - 1

        # Shift down the replaced
        # element to maintain the
        # heap property
        self.shiftDown(0)
        return result


    # Function to change the priority
    # of an element
    def changePriority(self,i, p):
        oldp = self.H[i]
        y    = list(self.H[i])
        y[0] = p
        self.H[i] = tuple(y)

        if (p > oldp[0]):
            self.shiftUp(i)
        else:
            self.shiftDown(i)


    # Function to get value of
    # the current maximum element
    def getMax(self):
        return self.H[0]


    # Function to remove the element
    # located at given index
    def remove(self, i):
        self.H[i] = (self.getMax()[0] + 1,0,"na")

        # Shift the node to the root
        # of the heap
        self.shiftUp(i)

        # Extract the node
        return self.extractMax()


    def swap(self,i, j):
        temp = self.H[i]
        self.H[i] = self.H[j]
        self.H[j] = temp

class Task:
    def __init__(self,name,deadline, bonus):
        self.name     = name
        self.deadline = deadline
        self.bonus    = bonus

class UsesCase:
    def __init__(self,use_case_name):
        self.use_case_name  = use_case_name
        self.tasks          = []
        self.assignedTasks    = []

    def __str__(self) -> str:
        return f"use_case_name={self.use_case_name} , tasks={self.tasks} assignedTasks={self.assignedTasks}"

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
        maximizeBonus implemented using hash priority queue to get the complexity in n*logn complexity
        :return:
        """
        n                   = len(self.tasks)
        ## order the tasks by bonus /  deadline
        merge_sort(self.tasks)


        taskPQueue = TaskHeapQueue()
        result   = []
        #maxHeap  = []
        # starting the iteration from the end
        for i in range(n - 1, -1, -1):
            # calculate slots between two deadlines
            if i == 0:
                slots_available = self.tasks[i].deadline
            else:
                slots_available = self.tasks[i].deadline - self.tasks[i-1].deadline

            taskPQueue.insert((-1*self.tasks[i].bonus, self.tasks[i].deadline, self.tasks[i].name))
            while slots_available and taskPQueue.H:
                # get the task with max_profit
                profit, deadline, job_id = taskPQueue.extractMax()
                # reduce the slots
                slots_available -= 1

                # include the job in the result array
                result.append(Task(job_id,deadline,profit*-1))

        merge_sort(result)
        self.assignedTasks = result


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
        if leftTasks[i].deadline <= rightTasks[j].deadline:
            orgTasks[k] = leftTasks[i]
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
        total_bonus = sum(list(map(lambda task: task.bonus, usecase.assignedTasks)))
        initialString = initialString + f"{total_bonus} \n"
        resultCaseString    =  resultCaseString+"\n"+ \
            f"For the use case {usecase.use_case_name}, " \
            f" the maximum bonus earned is {total_bonus}"

        orderString = "-->".join(list(map(lambda task: task.name, usecase.assignedTasks)))
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
    list_use_cases = readFile("inputPS5.txt")
    writeToFile(list_use_cases,"outputPS5.txt")