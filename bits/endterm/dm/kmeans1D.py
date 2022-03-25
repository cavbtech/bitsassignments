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



def findRandomCentrods(k):
    centroids = []
    for i in range(0,k):
        centroids.append(random.randint(min(points),max(points)))
    return centroids

def findCentroids(clusdict):
    centroids = []
    for cluster_points in clusdict.values():
        centroids.append(np.average(cluster_points))
    return centroids

def doKmeans(points, k):
    centroids =findRandomCentrods(k)
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
            cluster     = centroids[0]
            prevDist    = 9999999999999
            for j in range(0,len(centroids)):
                distance = abs(points[i] - centroids[j])
                if centroids[j] in distdict:
                    distdict[centroids[j]] = np.append(distdict[centroids[j]],[distance])
                else:
                    distdict[centroids[j]] = np.array([distance])
                if distance < prevDist:
                    cluster = centroids[j]
                    prevDist = distance
            if cluster in clusdict:
                values = clusdict[cluster]
                values = np.append(values,[points[i]])
                clusdict[cluster] = values
            else:
                clusdict[cluster] = [points[i]]

        for centroid in centroids:
            if centroid in clusdict:
                pass
            else:
                clusdict[centroid] = [0]

            if centroid in distdict:
                pass
            else:
                distdict[centroid] = [0]

        clusdict    = OrderedDict(sorted(clusdict.items()))
        distdict    = OrderedDict(sorted(distdict.items()))
        # print(f" centroids={centroids} and clusdict={clusdict} and distdict={distdict}")

        centroids = findCentroids(clusdict)
        # print(f" new centroids = {centroids}")
        printData(iteration,points,distdict,clusdict)
        isConverged = np.allclose(np.array(centroids),np.array(prevcentroids))
        if(isConverged):
            break

if __name__ == '__main__':
    points = np.asarray([185,170,168,179,182,188,180,183])
    print(f"""points={points}""")
    doKmeans(points,3)