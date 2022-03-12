class TaskHeapQueue:
    """
    A simple implementation of priority queue
    Can be implemented as static methods but this is better way of code separation
    """
    def shiftUp(self, heap, pos):
        """
        Traverse through smaller child till the last leaf node and compare the values
        It is called while deleting any element and it internally calls shiftDown
        :param heap:
        :param pos:
        :return:
        """
        endpos = len(heap)
        startpos = pos
        newitem = heap[pos]
        childpos = 2 * pos + 1  # leftmost child position
        while childpos < endpos:
            # Set childpos to index of smaller child.
            rightpos = childpos + 1
            if rightpos < endpos and not heap[childpos] < heap[rightpos]:
                childpos = rightpos
            # Move the smaller child up.
            heap[pos] = heap[childpos]
            pos = childpos
            childpos = 2 * pos + 1
        heap[pos] = newitem
        self.shiftDown(heap, startpos, pos)

    def shiftDown(self, heap, startpos, pos):
        """
        It is called while adding any element within the heap
        It is heapify algorithm where the parent node is compared to each child node till the last
        leaf node and swap the elements based on the priority
        :param heap:
        :param startpos:
        :param pos:
        :return:
        """
        newitem = heap[pos]

        while pos > startpos:
            parentpos = (pos - 1) >> 1
            parent = heap[parentpos]
            if newitem < parent:
                heap[pos] = parent
                pos = parentpos
                continue
            break
        heap[pos] = newitem

    def insert(self, heap, item):
        """Insert an element onto heap"""
        heap.append(item)
        self.shiftDown(heap, 0, len(heap) - 1)

    def removeMin(self, heap):
        """Remove the smallest element off the heap"""
        poppedItem = heap.pop()
        if heap:
            returnitem = heap[0]
            heap[0] = poppedItem
            self.shiftUp(heap, 0)
            return returnitem
        return poppedItem



class Task:
    def __init__(self,name,deadline, bonus):
        self.name     = name
        self.deadline = deadline
        self.bonus    = bonus

    def __str__(self):
        return f"name={self.name} , deadline={self.deadline} bonus={self.bonus}"


class UsesCase:
    def __init__(self,use_case_name):
        self.use_case_name  = use_case_name
        self.tasks          = []
        self.assignedTasks    = []

    def __str__(self):
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
        maxHeap  = []
        # starting the iteration from the end
        for i in range(n - 1, -1, -1):
            # calculate slots between two deadlines
            if i == 0:
                slots_available = self.tasks[i].deadline
            else:
                slots_available = self.tasks[i].deadline - self.tasks[i-1].deadline

            taskPQueue.insert(maxHeap, (-1 * self.tasks[i].bonus, self.tasks[i].deadline, self.tasks[i].name))
            while slots_available and maxHeap:
                # get the task with max_profit
                profit, deadline, job_id = taskPQueue.removeMin(maxHeap)
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
        if (leftTasks[i].deadline <= rightTasks[j].deadline):
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
    expected_usecases= 0
    for lineraw in file:
        ## assuming the entered data prefixed with discription
        # i.e "No of use-cases: 2" ... the program needs only the value
        ## if the input doesnt have the description then pick only the value
        if(lineraw.find(":") >0 ):
            line = (lineraw.split(":")[1]).strip()
        else:
            line = lineraw.strip()

        if(line==""):
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
    initialString       = ""
    resultCaseString    = ""
    taskOrderString     = ""

    for usecase in list_use_cases:
        total_bonus = sum(list(map(lambda task: task.bonus, usecase.assignedTasks)))
        initialString = initialString + f"{total_bonus} \n"
        resultCaseString    =  resultCaseString+"\n"+ \
            f"For the use case {usecase.use_case_name}, " \
            f" the maximum bonus earned is {total_bonus}"


        orderString = "-->".join(list(filter(lambda task: task!="na",
                                             map(lambda task: task.name, usecase.assignedTasks))))
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