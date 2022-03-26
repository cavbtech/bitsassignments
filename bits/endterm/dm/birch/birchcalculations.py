import math

import numpy as np


def calculate_radius(n, ls, ss):
    ss_by_n    = np.array([round(ss[0]/n,2), round(ss[1]/n,2)])
    ls_by_n_ws = np.array([round((ls[0]/n)**2,2),round((ls[1]/n)**2,2)])
    print(f"ss_by_n={ss_by_n} and ls_by_n_ws={ls_by_n_ws}")
    dist = math.dist(ss_by_n,ls_by_n_ws)
    print(f"dist={dist}")
    return dist**0.5


def calculate_diameter(n, ls, ss):
    nss2 = np.array([round(2*n*ss[0],2), round(2*n*ss[1], 2)])
    lss2 = np.array([round((2*ls[0]) ** 2, 2), round((2*ls[1]) ** 2, 2)])
    print(f"nss2={nss2} and lss2={lss2}")
    e_dist = math.dist(nss2, lss2)
    print(f"e_dist={e_dist}")
    diametersqr = round(e_dist/(n*(n-1)),2)
    return diametersqr ** 0.5


class CF_data:
    def __init__(self, number_of_points, linear_sum, square_sum):
        self.number_of_points = number_of_points
        self.linear_sum = linear_sum
        self.square_sum = square_sum
        self.centroid   = (linear_sum[0]/number_of_points, linear_sum[1]/number_of_points, )
        self.radius     = calculate_radius(number_of_points,linear_sum, square_sum)
        self.diameter   = calculate_diameter(number_of_points,linear_sum, square_sum)

    def __repr__(self):
        return f""" cluster feature = ({self.number_of_points}, {self.linear_sum}, {self.square_sum})
             centroid = {self.centroid}
             radius   = {self.radius}
             diameter = {self.diameter}
        """
    def __str__(self):
        return f""" cluster feature = ({self.number_of_points}, {self.linear_sum}, {self.square_sum})
             centroid = {self.centroid}
             radius   = {self.radius}
             diameter = {self.diameter}
        """

def cluster_feature(pointsDict):
    ls_x =0
    ls_y =0
    ss_x = 0
    ss_y =0
    for i in pointsDict:
        ls_x = ls_x + pointsDict[i][0]
        ls_y = ls_y + pointsDict[i][1]
        ss_x = ss_x + (pointsDict[i][0]**2)
        ss_y = ss_y + (pointsDict[i][1] ** 2)

    cf = CF_data(len(pointsDict),(ls_x,ls_y),(ss_x,ss_y))
    return cf

def radius():
    pass

if __name__ == '__main__':
    pointsDict = {"A":(1,2),"B":(1,4),"C":(2,4),"D":(3,2)}
    cf = cluster_feature(pointsDict)
    print(f"cluster feature={cf}")
