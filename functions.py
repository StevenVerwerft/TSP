import numpy as np
import random
import matplotlib.pyplot as plt
import time
from math import radians, sqrt, sin, cos, atan2

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


def geodistance(point1, point2):

    lon1, lon2 = radians(point1[1]), radians(point2[1])
    lat1, lat2 = radians(point1[0]), radians(point2[0])

    dlon = lon1 - lon2

    radius = 6372.833

    y = sqrt(
        (cos(lat2) * sin(dlon)) ** 2
        + (cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)) ** 2
        )
    x = sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(dlon)
    c = atan2(y, x)
    return radius * c


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
                matrix[i, j] = round(geodistance(city_array[i], city_array[j]))

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

    # initial goal function value
    print("starting goalfunction value: {}".format(total_distance(route, distance_matrix)))
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


def local_search2(route, distance_matrix, coordinates, move_type='2-opt', max_iter=300000, max_time=None,
                 first_x=None):
    # auxiliary variables
    n_cities = len(route)  # the amount of different cities in the TSP
    oldRouteDistance = total_distance(route, distance_matrix)
    BestGoalfunctionValues = [oldRouteDistance]
    AllGoalfunctionValues = [oldRouteDistance]
    movepool = [(i, j) for i in range(n_cities-1) for j in range(i+1, n_cities)]
    random.shuffle(movepool)

    # initial goal function value
    print("starting goalfunction value: {}".format(total_distance(route, distance_matrix)))
    # memories

    if first_x:
        firstXMemory = []

    # stopping criterion = time
    if max_time:
        starttime = time.time()
        max_iter = np.inf
        time_interval = range(1, int(max_time) + 1)
        print(time_interval)


    # to be checked if these variables stay 'local' or if deepcopy is necessary
    oldRoute = route.copy()
    coordinates = coordinates.copy()
    i = 1
    j = 1
    while i < max_iter:
        i += 1
        if max_time:
            if (time.time() - starttime) > max_time:
                break

        # pick two random cities in the existing route
        # each position in the route list refers to a unique city
        base = random.randint(0, len(movepool)-1)
        for move in movepool[base:] + movepool[:base]:

            # 2-opt: reverse order of route between move points
            if move_type == '2-opt':
                # make a temporary new route by two-opting the two randomly chosen cities
                newRoute = two_opt(oldRoute, move)

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
                    break

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


def tabu_search(route, distance_matrix, coordinates, max_iter, max_time=None, random_order=True, tabu_tenure=100):

    # auxiliary variables
    nCities = len(route)
    currentRouteDistance = total_distance(route, distance_matrix)
    bestGoalfunctionValues = [currentRouteDistance]
    allGoalfunctionValues = [currentRouteDistance]

    # intitial value
    print('starting value: {}'.format(currentRouteDistance))

    # initialize tabu memory
    tabuMemory = []

    # initialize movepool
    movePool = [(i, j) for i in range(nCities-1) for j in range(i+1, nCities)]
    random.shuffle(movePool)

    # stopping criterium = time
    if max_time:
        starttime = time.time()
        max_iter = np.inf
        time_interval = range(1, int(max_time)+1)

    currentRoute = route.copy()
    coordinates = coordinates.copy()
    i = 1
    j = 1
    while i < max_iter:
        i += 1
        localOptimum = True
        if max_time:
            ts = time.time()
            if (ts - starttime) > max_time:
                break

            # random order implementation
            if random_order:
                base = random.randint(0, len(movePool)-1)
            else:
                base = 0

            for move in movePool[base:] + movePool[:base]:
                j += 1
                newRoute = two_opt(currentRoute, move)
                newrouteDistance = total_distance(newRoute, distance_matrix)
                allGoalfunctionValues.append(newrouteDistance)

                if (newrouteDistance < currentRouteDistance) and move not in tabuMemory:
                    print('better solution found')
                    tabuMemory.append(move)
                    if len(tabuMemory) > tabu_tenure:
                        tabuMemory.pop(0)

                    bestGoalfunctionValues.append(newrouteDistance)

                    currentRouteDistance = newrouteDistance
                    currentRoute = newRoute
                    coordinates = two_opt(coordinates, move)

                    localOptimum = False
                    break

            if localOptimum:
                # perform some random moves as perturbation and restart the search
                perturbationIntensity = 1  # number of random moves selected in perturbation
                moves = random.sample(movePool, perturbationIntensity)
                for move in moves:
                    currentRoute = two_opt(currentRoute, move)
                    coordinates = two_opt(coordinates, move)
                    tabuMemory.append(move)
                    if len(tabuMemory) > tabu_tenure:
                        tabuMemory.pop(0)

                currentRouteDistance = total_distance(currentRoute, distance_matrix)
                bestGoalfunctionValues.append(currentRouteDistance)

    if max_time:
        print('Calculation time (seconds): {}s'.format(round(time.time() - starttime), 2))
        print('Total iterations: {:.2E}'.format(j))
        print('Final goalfunction value: {}'.format(min(bestGoalfunctionValues)))

    return currentRoute, coordinates, bestGoalfunctionValues, allGoalfunctionValues
