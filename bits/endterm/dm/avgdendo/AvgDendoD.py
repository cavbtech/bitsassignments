import numpy as np
import pandas as pd
def avg_dist_dict(distDict, min_xcol_name, min_ycol_name):
    x_value =  distDict[min_xcol_name]
    y_value =  distDict[min_ycol_name]
    xy_value = {}
    for i in x_value:
        xy_value[i]=round(((x_value[i]+y_value[i])/2),2)
    xy_value[min_xcol_name+min_ycol_name] = 0.0
    distDict[min_xcol_name+min_ycol_name] = xy_value
    distDict.pop(min_xcol_name)
    distDict.pop(min_ycol_name)
    for rows in distDict:
        if min_ycol_name in distDict[rows]:
            distDict[rows].pop(min_ycol_name)
        if min_xcol_name in distDict[rows]:
            distDict[rows].pop(min_xcol_name)
        distDict[rows][min_xcol_name+min_ycol_name] = xy_value[rows]
    return distDict

def findMininDictFromBottomUp(distDict):
    min_x_index =0
    min_y_index =0
    min_value    = 99999999
    rowCount = 0;
    for i in distDict:
        row = distDict[i]
        if(rowCount==0):
            rowCount = rowCount+1
            continue
        colcount = 0
        for j in row:
            if(colcount<rowCount):
                if(min_value > distDict[i][j]):
                    min_value = distDict[i][j]
                    min_x_index = rowCount
                    min_y_index = colcount
            colcount = colcount +1
        rowCount = rowCount + 1
    return (min_x_index,min_y_index,min_value)

def dendo_avg(distDict):
    mindata = findMininDictFromBottomUp(distDict)
    print("-------------------------------------------------------")
    df = pd.DataFrame.from_dict(distDict)
    print(df)
    if df.size ==1 :
        return
    # get the last row
    coldict = {}
    for i in range (0,len(df.columns)):
        coldict[i] = df.columns[i]
    min_xcol_name = coldict[mindata[0]]
    min_ycol_name = coldict[mindata[1]]

    distDict      = avg_dist_dict(distDict,min_xcol_name,min_ycol_name)
    dendo_avg(distDict)

if __name__ == '__main__':
    #In case the question comes in single dimension
    distDict = {'A': {'A': 0.0, 'B': 21.93, 'C': 20.81, 'D': 7.21, 'E': 10.44, 'F': 5.83, 'G': 5.39, 'H': 12.17},
     'B': {'A': 21.93, 'B': 0.0, 'C': 4.47, 'D': 15.0, 'E': 28.64, 'F': 27.66, 'G': 17.2, 'H': 30.87},
     'C': {'A': 20.81, 'B': 4.47, 'C': 0.0, 'D': 13.6, 'E': 26.08, 'F': 26.25, 'G': 15.62, 'H': 28.3},
     'D': {'A': 7.21, 'B': 15.0, 'C': 13.6, 'D': 0.0, 'E': 14.32, 'F': 12.73, 'G': 2.24, 'H': 16.49},
     'E': {'A': 10.44, 'B': 28.64, 'C': 26.08, 'D': 14.32, 'E': 0.0, 'F': 7.81, 'G': 12.17, 'H': 2.24},
     'F': {'A': 5.83, 'B': 27.66, 'C': 26.25, 'D': 12.73, 'E': 7.81, 'F': 0.0, 'G': 10.63, 'H': 8.6},
     'G': {'A': 5.39, 'B': 17.2, 'C': 15.62, 'D': 2.24, 'E': 12.17, 'F': 10.63, 'G': 0.0, 'H': 14.32},
     'H': {'A': 12.17, 'B': 30.87, 'C': 28.3, 'D': 16.49, 'E': 2.24, 'F': 8.6, 'G': 14.32, 'H': 0.0}}
    dendo_avg(distDict)

    pass


