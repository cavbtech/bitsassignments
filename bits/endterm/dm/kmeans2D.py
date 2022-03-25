import numpy as np
import random
from collections import OrderedDict
import pandas as pd

def flatenMap(clustdict):
    dictbig = {}
    for item in clustdict.items():
        for point in item[1]:
             dictbig[point] = item[0]
    return dictbig

def calculate_dist(point1,point2):
    dist = ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)**0.5
    return dist


def printData(iteration,points,distdict, clustdict):
    print(f"---------------iteration-{iteration}-------------")
    distdataset = {
        'point':points
    }
    df1 = pd.DataFrame(distdataset)
    for item in distdict.items():
        df1[item[0]] = item[1]


    reverseddict =  flatenMap(clustdict)
    #reverseddict = {v: k for k, v in clustdict.items()}
    df3 = pd.DataFrame(dict([(k, pd.Series((k,v))) for k, v in reverseddict.items()]))\
        .T.set_axis(['point','cluster'],axis=1)

    df4 = pd.merge(df1,df3,on="point", how="left")
    print(df4)



def findRandomCentrods(points,k):
    centroids = ()
    for i in range(0,k):
        point = points[i]
        centroids = centroids+((point),)
    centroids = sorted(centroids)

    return centroids

def findCentroids(clusdict):
    centroids = ()
    for cluster_points in clusdict.values():
        counter = 0
        x =0
        y =0
        for datavalues in cluster_points:
            x = x + datavalues[0]
            y = y + datavalues[1]
            counter = counter+1
        if(counter>0):
            centroids = centroids+((round((x/counter),2),round((y/counter),2)),)
    centroids = sorted(centroids)
    return centroids


def checkForConvergence(centroids, prevcentroids):
    epsilon = 0.5
    isConverged = True;

    for i in range(0,len(centroids)):
        leftPoint = centroids[i]
        righPoint = prevcentroids[i]

        if((abs(round(leftPoint[0],2)-round(righPoint[0],2))<=epsilon
               and abs(round(leftPoint[1],2)-round(righPoint[1],2)<=epsilon ))==False):
            isConverged =  False
            break;
    return isConverged

def doKmeans(points, k):
    centroids =findRandomCentrods(points,k)
    print(f"Initial Assumed Random Centroids = {centroids}")
    iteration = 0
    # at this time forget about optimization. dont have time. Brut force
    # I can order the cluster data but the challenge is it needs to done for 2 dimensions too which is painful
    while(True):
        iteration = iteration+ 1

        clusdict = {}
        distdict = {}
        prevcentroids = centroids
        for i in range(0,len(points)):
            ## default cluster assignment
            cluster     = centroids[0]
            prevDist    = 9999999999999
            for j in range(0,len(centroids)):
                distance = calculate_dist(points[i],centroids[j])
                if centroids[j] in distdict:
                    distdict[centroids[j]] = np.append(distdict[centroids[j]],[distance])
                else:
                    distdict[centroids[j]] = np.array([distance])
                if distance < prevDist:
                    cluster = centroids[j]
                    prevDist = distance
            if cluster in clusdict:
                values = clusdict[cluster]
                values = values+(points[i],)
                clusdict[cluster] = values
            else:
                clusdict[cluster] = (points[i],)

        for centroid in centroids:
            if centroid in clusdict:
                pass
            else:
                clusdict[centroid] = (0,0)

            if centroid in distdict:
                pass
            else:
                distdict[centroid] = np.array([0])

        clusdict    = OrderedDict(sorted(clusdict.items()))
        distdict    = OrderedDict(sorted(distdict.items()))
        # print(f" centroids={centroids} and clusdict={clusdict} and distdict={distdict}")

        centroids = findCentroids(clusdict)
        # print(f" new centroids = {centroids}")
        printData(iteration,points,distdict,clusdict)


        isConverged = checkForConvergence(centroids,prevcentroids)
        print(f"current centroids={centroids} and prevcentroids={prevcentroids} "
              f"and are they converged ? ={isConverged}")
        if(isConverged):
            break

if __name__ == '__main__':
    points = (
              (185,72),(170,56),(168,60),
              (179,68),(182,82),(188,77),
                       (180,70),(183,84),)
    print(f"""points={points}""")
    doKmeans(points,2)