import math

from bits.endterm.dm.maxdendo.MaxDendoD import dendo_max

if __name__ == '__main__':
    pointsDict = {"A":(185,72),"B":(170,56),"C":(168,60),
                  "D":(179,68),"E":(182,82),
                  "F":(188,77),"G":(180,70),"H":(183,84)}

    print(f"Given points {pointsDict}")
    dist = math.dist(pointsDict["A"],pointsDict["B"])
    distDict = {}
    for i in pointsDict:
        distDictJ = {}
        for j in pointsDict:
            dist = math.dist(pointsDict[i],pointsDict[j])
            distDictJ[j] = round(dist,2)
        distDict[i] = distDictJ
    print(f"{distDict}")
    df2 = dendo_max(distDict)


