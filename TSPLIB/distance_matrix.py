import sys
import getopt
import random
import numpy as np

try:
    opts, args = getopt.getopt(sys.argv[1:], 'f:')
except getopt.GetoptError:
    sys.exit()

for opt, arg in opts:
    if opt == '-f':
        file_name = arg

try:
    with open(file_name, 'r') as file:
        lines = file.readlines()
except FileNotFoundError:
    print('an exception occurred')
    print(sys.exc_info()[0])
    sys.exit()

except NameError:
    print('a name exception has occurred')
    print(sys.exc_info()[0])
    sys.exit()
"""
File structure:

0 Name = ...
1 Comment = ...
2 Type = <TSP>
3 Dimension = ...
4 Edge weight type = <EUCL2D, GEO, ATT>
5 Node coord section
6 id longitude latitude
7 ...
. ...
. ...
. EOF
"""

# last line is 'EOF' -> should not be considered a coordinate
coords = [(float(city.split()[1]), float(city.split()[2])) for city in lines[6:-1]]


def euclidean_distance(point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**.5


def distance_matrix(city_array):
    n = len(city_array)
    matrix = np.zeros(shape=(n, n))
    for i in range(n):
        matrix[i][i] = np.inf

    for i in range(n-1):
        for j in range(i+1, n):
            matrix[i][j] = euclidean_distance(city_array[i], city_array[j])
    return matrix

d_matrix = distance_matrix(coords)

quit()


def k_means(data, k=10, it=100):

    if k > len(data):
        print('k larger than number of observations')
        return data
    # step 1: initialization by random choice
    centroids = random.sample(data, k)

    for iter in range(it):
        # step 2: assignment
        clusters_coords = [[] for i in range(k)]
        clusters_city_ids = [[] for i in range(k)]

        for city_id, city in enumerate(coords):
            distances = [euclidean_distance(city, centroid) for centroid in centroids]
            try:
                cluster_id = np.argmin(distances)[0]
            except IndexError:
                cluster_id = np.argmin(distances)
            except:
                raise
                sys.exit()

            clusters_coords[cluster_id].append(city)
            clusters_city_ids[cluster_id].append(city_id)

        new_centroids = centroids.copy()
        # step 3: update centroids
        for (i, cluster) in enumerate(clusters_coords):
            new_centroids[i] = tuple(np.sum(cluster, axis=0)/len(cluster))

        if new_centroids == centroids:
            print('K-means converged!')
            print('iteration: ', iter)
            break
        else:
            centroids = new_centroids
            continue

    return clusters_coords, clusters_city_ids

