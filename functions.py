import numpy as np
import random
import matplotlib.pyplot as plt
import time
np.set_printoptions(linewidth=150)


def readfile(filename):
    # Open input file
    infile = open(filename, 'r')

    # Read instance header

    Name = infile.readline().strip().split()[1]  # NAME
    FileType = infile.readline().strip().split()[1]  # TYPE
    Comment = infile.readline().strip().split()[1]  # COMMENT
    Dimension = infile.readline().strip().split()[1]  # DIMENSION
    EdgeWeightType = infile.readline().strip().split()[1]  # EDGE_WEIGHT_TYPE
    infile.readline()

    # Read node list
    nodelist = []
    N = int(Dimension)
    min_x, min_y = 0.0, 0.0

    for i in range(0, N):
        x, y = infile.readline().strip().split()[1:]
        nodelist.append((float(x), float(y)))

        if not EdgeWeightType == 'GEO':
            if float(x) < min_x:
                min_x = float(x)
            if float(y) < min_y:
                min_y = float(y)
    nodelist = np.array(nodelist)

    if not EdgeWeightType == 'GEO':

        nodelist[:, 0] = nodelist[:, 0] - min_x
        nodelist[:, 1] = nodelist[:, 1] - min_y

    # Close input file

    infile.close()

    return nodelist.tolist(), EdgeWeightType


def euclidean_distance(point1, point2):
    return np.sqrt(sum([(point1[0] - point2[0]) ** 2, (point1[1] - point2[1]) ** 2]))


def distance_from_geo(point1, point2, measure='km'):

    lon1, lon2 = to_degrees(point1[1]), to_degrees(point2[1])
    lat1, lat2 = to_degrees(point1[0]), to_degrees(point2[0])

    if measure == 'km':
        radius = 6378.388
    elif measure == 'miles':
        radius = 3959

    dtr = np.pi / 180.
    phi_1 = lat1 * dtr
    phi_2 = lat2 * dtr
    lambda_1 = lon1 * dtr
    lambda_2 = lon2 * dtr

    d_phi = phi_2 - phi_1
    d_lambda = lambda_2 - lambda_1

    a = haversine(d_phi + np.cos(phi_1)*np.cos(phi_2)*haversine(d_lambda))
    d = 2 * radius * np.arcsin(np.sqrt(a))

    return d


def to_degrees(point):

    deg = int(point + .5)
    min = point - deg
    new_point = deg + (5./3. * min)
    return new_point


def haversine(angle):
    return np.sin(angle/2.)**2


def distance_matrix(city_array, distance_type=None, measure='km'):
    """

    :param city_array: Array containing coordinates for the different cities
    :return: matrix with euclidean distances between each city ( = right-triangular matrix)
    """
    N = len(city_array)

    # start with an matrix filled with zeros
    matrix = np.zeros(shape=(N, N))

    # append the matrix with right-triangular elements (assumption distance matrix is symmetrical)

    if not distance_type == 'GEO':
        for i in range(N - 1):
            matrix[i, i] = np.inf  # distance on diagonal (between same cities) to infinite
            for j in range(i + 1, N):
                matrix[i, j] = round(euclidean_distance(city_array[i], city_array[j]))

        matrix[N-1, N-1] = np.inf

    if distance_type == 'GEO':
        for i in range(N - 1):
            matrix[i, i] = np.inf
            for j in range(i +1, N):
                matrix[i, j] = round(distance_from_geo(city_array[i], city_array[j], measure=measure))

        matrix[N-1, N-1] = np.inf

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
    print('starting city: ', starting)
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


def local_search(route, distance_matrix, coordinates, move_type='2-opt', max_iter=300000, max_time=None,
                 first_x=None):

    # auxiliary variables
    n_cities = len(route)  # the amount of different cities in the TSP
    oldRouteDistance = total_distance(route, distance_matrix)
    BestGoalfunctionValues = [oldRouteDistance]
    AllGoalfunctionValues = [oldRouteDistance]


    # memories
    if first_x:
        firstXMemory = []

    # stopping criterion = time
    if max_time:
        starttime = time.time()
        max_iter = np.inf
        time_interval = range(1, int(max_time)+1)
        print(time_interval)

    # to be checked if these variables stay 'local' or if deepcopy is necessary
    oldRoute = route.copy()
    coordinates = coordinates.copy()
    i = 1
    while i < max_iter:
        i += 1
        if max_time:
            if (time.time() - starttime) > max_time:
                break

        # pick two random cities in the existing route
        # each position in the route list refers to a unique city
        move = sorted(random.sample(range(n_cities), 2))

        # 2-opt: reverse order of route between move points
        if move_type == '2-opt':
            # make a temporary new route by two-opting the two randomly chosen cities
            newRoute = two_opt(oldRoute, move)

        # swap: swap cities on move points
        elif move_type == 'swap':
            # make a temporary new route by swapping the two randomly chosen cities
            a, b = move[0], move[1]
            newRoute = route[:a] + route[b:b + 1] + route[a + 1:b] + route[a:a + 1] + route[b:]

        # check if the goalfunction of the new route is smaller than the goalfunction of the oldroute
        newRouteDistance = total_distance(newRoute, distance_matrix)
        AllGoalfunctionValues.append(newRouteDistance)  # memory for all visitied solutions

        if newRouteDistance - oldRouteDistance < 0:
            if not first_x:
                # local search performs best improvement strategy
                oldRoute = newRoute
                coordinates = two_opt(coordinates, move)
                oldRouteDistance = newRouteDistance
                BestGoalfunctionValues.append(oldRouteDistance)

            if first_x:
                # local search performs first x improvement strategy
                firstXMemory.append((move, newRouteDistance))
                if len(firstXMemory) == first_x:
                    # pick the best solution from the memory
                    # sort the moves in the first x memory by ascending goalfunction value
                    firstXMemory = sorted(firstXMemory, key=lambda x: x[1])
                    best_move = firstXMemory[0][0]
                    oldRoute = two_opt(oldRoute, best_move)
                    oldRouteDistance = firstXMemory[0][1]
                    BestGoalfunctionValues.append(oldRouteDistance)

    if max_time:
        print('calculation time (seconds): {}s'.format(round(time.time() - starttime), 4))
        print('total iterations: {:.2E}'.format(i))

    print('final goalfunction value: ', BestGoalfunctionValues[-1])
    return oldRoute, coordinates, BestGoalfunctionValues, AllGoalfunctionValues


def plot_coordinates(coordinate_array):
    # 1 check if first and last point is the same city
    if coordinate_array[0] != coordinate_array[-1]:
        coordinate_array.append(coordinate_array[0])

    # make seperate arrays for x and y coordinates
    x, y = np.array([city[0] for city in coordinate_array]), np.array([city[1] for city in coordinate_array])

    # dotted style for the cities
    plt.plot(x, y, 'o')

    # show the path with arrows
    plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1],
               y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color='#1F77B4')

    # give the first city a different color
    plt.scatter(x[0], y[0], c='black', s=100)
