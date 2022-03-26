import numpy as np
import collections
import matplotlib.pyplot as plt
import queue
from scipy.spatial.distance import cityblock

# Define label for differnt point group
NOISE = 0
UNASSIGNED = 0
core = -1
edge = -2


# function to find all neigbor points in radius
def neighbor_points(data, pointId, radius):
    points = []
    for i in range(len(data)):
        # Euclidian distance using L2 Norm
        mdist = cityblock(data[i],data[pointId])
        edist = np.linalg.norm(data[i] - data[pointId])

        if  mdist<= radius:
            points.append(i)
        print(f"neighours: from point {data[i]} to point {data[pointId]}  "
              f"with mdist={mdist} and radius={radius} and it is neighbor {mdist<= radius}")
    return points


# DB Scan algorithom
def dbscan(data, Eps, MinPt):
    # initilize all pointlable to unassign
    pointlabel = [UNASSIGNED] * len(data)
    pointcount = []
    # initilize list for core/noncore point
    corepoint = []
    noncore = []

    # Find all neigbor for all point
    for i in range(len(data)):
        pointcount.append(neighbor_points(train, i, Eps))

    print(f"neighbour for each point: pointcount={pointcount}")

    # Find all core point, edgepoint and noise
    for i in range(len(pointcount)):
        if (len(pointcount[i]) >= MinPt):
            pointlabel[i] = core
            corepoint.append(i)
        else:
            noncore.append(i)

    for i in noncore:
        for j in pointcount[i]:
            if j in corepoint:
                pointlabel[i] = edge

                break

    # start assigning point to luster
    cl = 1
    # Using a Queue to put all neigbor core point in queue and find neigboir's neigbor
    for i in range(len(pointlabel)):
        q = queue.Queue()
        if (pointlabel[i] == core):
            pointlabel[i] = cl
            for x in pointcount[i]:
                if (pointlabel[x] == core):
                    q.put(x)
                    pointlabel[x] = cl
                elif (pointlabel[x] == edge):
                    pointlabel[x] = cl
            # Stop when all point in Queue has been checked
            while not q.empty():
                neighbors = pointcount[q.get()]
                for y in neighbors:
                    if (pointlabel[y] == core):
                        pointlabel[y] = cl
                        q.put(y)
                    if (pointlabel[y] == edge):
                        pointlabel[y] = cl
            cl = cl + 1  # move to next cluster

    print(f"corepoints={corepoint} and noncore or edge points={noncore} ")
    return pointlabel, cl


# Function to plot final result
def plotRes(data, clusterRes, clusterNum):
    nPoints = len(data)
    scatterColors = ['black', 'green', 'brown', 'red', 'purple', 'orange', 'yellow']
    for i in range(clusterNum):
        if (i == 0):
            # Plot all noise point as blue
            color = 'blue'
        else:
            color = scatterColors[i % len(scatterColors)]
        x1 = [];
        y1 = []
        for j in range(nPoints):
            if clusterRes[j] == i:
                x1.append(data[j, 0])
                y1.append(data[j, 1])
        plt.scatter(x1, y1, c=color, alpha=1, marker='.')


# Load Data
#raw = spio.loadmat('DBSCAN.mat')
#"A":(1,2),"B":(1,4),"C":(2,4),"D":(3,2)
train = np.array([[1,2],[1,4],[2,4],[3,2],[3,5],[3,7],[4,5],[4,6]])

# Set EPS and Minpoint
epss = [2]
minptss = [3]
# Find ALl cluster, outliers in different setting and print resultsw
for eps in epss:
    for minpts in minptss:
        print('Set eps = ' + str(eps) + ', Minpoints = ' + str(minpts))
        pointlabel, cl = dbscan(train, eps, minpts)
        plotRes(train, pointlabel, cl)
        plt.show()
        print('number of cluster found: ' + str(cl - 1))
        counter = collections.Counter(pointlabel)
        print(counter)
        outliers = pointlabel.count(0)
        print('numbrer of outliers found: ' + str(outliers) + '\n')


