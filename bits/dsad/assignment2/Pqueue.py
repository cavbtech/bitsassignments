class TaskHeap:

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
        while (i > 0 and self.H[self.parent(i)] < self.H[i]):
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

        if (l <= self.size and self.H[l] > self.H[maxIndex]):
            maxIndex = l

        # Right Child
        r = self.rightChild(i)

        if (r <= self.size and self.H[r] > self.H[maxIndex]):
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
        self.H[i] = p

        if (p > oldp):

            self.shiftUp(i)
        else:
            self.shiftDown(i)


    # Function to get value of
    # the current maximum element
    def getMax(self):
        return self.H[0]


    # Function to remove the element
    # located at given index
    def Remove(self,i):
        print(f"i={i} and self.getMax()={self.getMax()} and self.H[i]={self.H[i]}")
        self.H[i] = self.getMax() + 1
        print(f"i={i} and self.getMax()={self.getMax()} and self.H[i]={self.H[i]}")
        # Shift the node to the root
        # of the heap
        self.shiftUp(i)

        # Extract the node
        self.extractMax()


    def swap(self,i, j):
        temp = self.H[i]
        self.H[i] = self.H[j]
        self.H[j] = temp


if __name__ == '__main__':
    prioryQ = TaskHeap()
    # Insert the element to the
    # priority queue
    prioryQ.insert(45)
    prioryQ.insert(20)
    prioryQ.insert(14)
    prioryQ.insert(12)
    prioryQ.insert(31)
    prioryQ.insert(7)
    prioryQ.insert(11)
    prioryQ.insert(13)
    prioryQ.insert(7)

    i = 0

    # Priority queue before extracting max
    print("Priority Queue : ", end="")
    while (i <= prioryQ.size):
        print(prioryQ.H[i], end=" ")
        i += 1

    print()

    # Node with maximum priority
    print("Node with maximum priority :", prioryQ.extractMax())

    # Priority queue after extracting max
    print("Priority queue after extracting maximum : ", end="")
    j = 0
    while (j <= prioryQ.size):
        print(prioryQ.H[j], end=" ")
        j += 1

    print()

    # Change the priority of element
    # present at index 2 to 49
    prioryQ.changePriority(2, 49)
    print("Priority queue after priority change : ", end="")
    k = 0
    while (k <= prioryQ.size):
        print(prioryQ.H[k], end=" ")
        k += 1

    print()

    # Remove element at index 3
    prioryQ.Remove(3)
    print("Priority queue after removing the element : ", end="")
    l = 0
    while (l <= prioryQ.size):
        print(prioryQ.H[l], end=" ")
        l += 1

# This code is contributed by divyeshrabadiya07.
