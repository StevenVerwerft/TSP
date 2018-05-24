import numpy as np


def euclidean_distance(point1, point2):
    return np.sqrt(sum([(point1[0] - point2[0])**2, (point1[1] - point2[1])**2]))


def distance_matrix(city_array):
    """

    :param city_array: Array containing coordinates for the different cities
    :return: matrix with euclidean distances between each city ( = right-triangular matrix)
    """
    N = len(city_array)

    # start with an matrix filled with zeros
    matrix = np.zeros(shape=(N, N))

    # append the matrix with right-triangular elements (assumption distance matrix is symmetrical)
    for i in range(N-1):
        matrix[i, i] = np.inf  # distance on diagonal (between same cities) to infinite
        for j in range(i+1, N):
            matrix[i, j] = euclidean_distance(city_array[i], city_array[j])

    return matrix


def get_pairs(array):
    pairs = []
    for i in range(len(array)):
        try:
            pairs.append(sorted([array[i], array[i+1]]))
        except IndexError:
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


def two_opt(route, move):

    assert move[0] < move[1], 'move pair not increasing: ({}, {})'.format(move[0], move[1])
    newRoute = route[:move[0]] + list(reversed(route[move[0]: move[1]+1])) + route[move[1]+1:]
    return newRoute


def swap(route, move):

    assert move[0] < move[1], 'move pair not increasing'
    newRoute = route[:move[0]] + route[move[1]: move[1]+1] + route[move[0]+1: move[1]] + route[move[0]: move[0]+1] \
               + route[move[1]+1:]
    return newRoute