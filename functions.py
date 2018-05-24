import numpy as np
import random


def euclidean_distance(point1, point2):
    return np.sqrt(sum([(point1[0] - point2[0]) ** 2, (point1[1] - point2[1]) ** 2]))


def distance_matrix(city_array):
    """

    :param city_array: Array containing coordinates for the different cities
    :return: matrix with euclidean distances between each city ( = right-triangular matrix)
    """
    N = len(city_array)

    # start with an matrix filled with zeros
    matrix = np.zeros(shape=(N, N))

    # append the matrix with right-triangular elements (assumption distance matrix is symmetrical)
    for i in range(N - 1):
        matrix[i, i] = np.inf  # distance on diagonal (between same cities) to infinite
        for j in range(i + 1, N):
            matrix[i, j] = euclidean_distance(city_array[i], city_array[j])

    return matrix


def get_pairs(array):
    pairs = []
    for i in range(len(array)):
        try:
            pairs.append(sorted([array[i], array[i + 1]]))
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
    newRoute = route[:move[0]] + list(reversed(route[move[0]: move[1] + 1])) + route[move[1] + 1:]
    return newRoute


def swap(route, move):
    assert move[0] < move[1], 'move pair not increasing'
    newRoute = route[:move[0]] + route[move[1]: move[1] + 1] + route[move[0] + 1: move[1]] + route[move[0]: move[0] + 1] \
               + route[move[1] + 1:]
    return newRoute


def greedy_search(coordinates, distance_matrix):
    """
    :param coordinates: Array containing the coordinates of the cities
    :param distance_matrix: Matrix holding the Euclidean distances between each of the cities
    :return:
    """
    # initialization
    n_cities = len(coordinates)

    # original distance matrix should remain unaffected after greedy search
    distance_matrix = distance_matrix.copy()
    # make a greedy solution
    route_indices = []
    route_coordinates = []

    # 1 pick random first city
    starting = random.choice(range(n_cities))

    route_indices.append(starting)

    # The city coordinates are adjusted simultanously with the route, alternatively this can be done in the end
    # CHECK IF THIS RESULTS INTO THE SAME SOLUTION8
    route_coordinates.append(coordinates[starting])

    while len(route_indices) < len(coordinates):
        # take the last occuring city in the route
        previous_city = route_indices[-1]

        # select all pairs between last city and other cities in distance matrix
        # assumption: distances are symmetrical, only L/U-triangle needed
        # distances are for one part found in a column and for the other part in a row, indexed by the previous city

        neighbours = np.append(distance_matrix[:previous_city, previous_city], distance_matrix[previous_city,
                                                                               previous_city:])

        # the nearest city to the last city in the route is chosen as the next city
        next_city = np.argmin(neighbours)  # position of nearest neighbour

        # update the current route indices and the coordinates
        route_indices.append(next_city)
        route_coordinates.append(coordinates[next_city])

        # CONSTRAINT: Each city allowed to be included in the route only once!
        # After choosing a city based on the nearest distance in the distance matrix we can do two things:
        # 1: Delete all occurrences of this city in the distance matrix (matrix becomes smaller in the end)
        # => faster evaluation of minimal distances => more efficient
        # 2: Set all the distances to and from this city on infinite
        # => matrix stays the same size, but the indices in the matrix keep referencing to the cities in the array

        # REMARK: assess the efficiency gains of the first method!

        # Method 2: infinite distance to visited cities.
        distance_matrix[:, previous_city], distance_matrix[previous_city, :] = np.inf, np.inf

    return route_indices, route_coordinates


def local_search(route, distance_matrix, coordinates, move_type='2-opt'):
    max_iter = 100000
    oldRoute = route.copy()
    print(oldRoute)
    coordinates = coordinates.copy()
    for i in range(max_iter):
        [a, b] = sorted(random.sample(range(len(route)), 2))

        # 2-opt: reverse order of route between move points
        if move_type == '2-opt':
            newRoute = two_opt(oldRoute, [a, b])
        # swap: swap cities on move points
        if move_type == 'swap':
            newRoute = route[:a] + route[b:b + 1] + route[a + 1:b] + route[a:a + 1] + route[b:]

        if total_distance(newRoute, distance_matrix) - total_distance(oldRoute, distance_matrix) < 0:
            print('better route found!')
            print('total distance: ', total_distance(newRoute, distance_matrix))
            oldRoute = newRoute
            coordinates = two_opt(coordinates, [a, b])

    return oldRoute, coordinates
