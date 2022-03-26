import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#https://www.mlstack.cafe/blog/k-means-clustering-interview-questions
# function returns WSS score for k values from 1 to kmax
def calculate_WSS(points, kmax):
    sse = []
    for k in range(1, kmax + 1):
        kmeans = KMeans(n_clusters=k).fit(points)
        #print(f"for k={k} kmeans={kmeans}")
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0
        print(f"for k={k} kmeans={kmeans} centroids={centroids} and pred_clusters={pred_clusters}")

        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]

            curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2

        sse.append(curr_sse)
    return sse

if __name__ == '__main__':
    points = np.asarray([
              (185,72),(170,56),(168,60),
              (179,68),(182,82),(188,77),
                       (180,70),(183,84)
              # [185,170,168,179,182,188,180,183],
              # [72,56,60,68,82,77,70,84]
              ])
    results = calculate_WSS(points, 4)
    print(f"""results = {results}""")
    plt.plot( results, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method using Distortion')
    plt.show()