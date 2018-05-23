import numpy as np


def euclidean_distance(point1, point2):
    return np.sqrt(sum([(point1[0] - point2[0])**2, (point1[1] - point2[1])**2]))


def distance_matrix(city_array):
    N = len(city_array)
    matrix = np.zeros(shape=(N, N))
    for i in range(N-1):
        for j in range(i+1, N):
            matrix[i, j] = euclidean_distance(city_array[i], city_array[j])
    return matrix


def get_pairs(array):
    pairs = []
    for i in range(len(array)):
        try:
            pairs.append((array[i], array[i+1]))
        except IndexError:
            print('something went wrong')
            pass
    return pairs


def get_distances(pairs, distance_matrix):
    distances = []
    for pair in pairs:
        # find the distance between the cities in the given distance matrix
        distances.append(distance_matrix[pair[0], pair[1]])
    return distances


def total_distance(solution, distance_matrix):

    return sum(get_distances(get_pairs(solution), distance_matrix))
