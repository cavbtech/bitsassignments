# job sequence from a given array
# of jobs with deadlines and profits

# function to schedule the jobs take 2
# arguments array and no of jobs to schedule


def printJobScheduling(arr, t):
    # length of array
    n = len(arr)

    # Sort all jobs according to
    # decreasing order of profit
    for i in range(n):
        for j in range(n - 1 - i):
            if arr[j][2] < arr[j + 1][2]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

    # To keep track of free time slots
    result = [False] * t

    # To store result (Sequence of jobs)
    job = ['-1'] * t

    # Iterate through all given jobs
    for i in range(len(arr)):

        # Find a free slot for this job
        # (Note that we start from the
        # last possible slot)
        for j in range(min(t - 1, arr[i][1] - 1), -1, -1):

            # Free slot found
            if result[j] is False:
                result[j] = True
                job[j] = arr[i][0]
                break

    # print the sequence
    print(job)


# Driver COde
'''
arr = [['a', 1, 20],  # Job Array
       ['b', 2, 40],
       ['c', 3, 10],
       ['d', 1, 10],
       ['e', 4, 20]]
'''


# Driver COde
arr = [['a', 4, 20],  # Job Array
       ['b', 1, 30],
       ['c', 2, 10],
       ['d', 2, 40]]

print("Following is maximum profit sequence of jobs")

# Function Call
printJobScheduling(arr, 4)