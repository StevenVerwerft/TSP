from functions import *
import random
#random.seed(10)
import matplotlib.pyplot as plt
import sys
import copy
import getopt


def main():
    """
        Usage:

    """

    # Default values:
    n_cities = 100
    cities = [random.sample(range(100), 2) for x in range(n_cities)]
    iters = 30e3
    max_time = None
    first_x = None
    edgeweighttype = None

    # otpion parser (only short options)
    try:
        opts, args = getopt.getopt(sys.argv[1:], "n:i:r:t:x:h")
        print(opts)
    except getopt.GetoptError as err:
            print(err)
            sys.exit(2)

    for o, a in opts:
        if o == '-n':
            n_cities = int(a)

        elif o == '-i':
            filename = a
            cities, edgeweighttype = readfile(filename)

        elif o == '-r':
            iters = int(a)

        elif o == '-t':
            max_time = float(a)

        elif o == '-x':
            first_x = int(a)

        elif o == '-h':
            print('-i [inputfile] -n [num cities (optional)] -r [iterations (default=30e3)] '
                  '-t [max time (default=None)] -x [first x (default=None)]')
            sys.exit(2)

    # generate distance matrix
    d_matrix = distance_matrix(cities, distance_type=edgeweighttype, measure='km')  # fix modification by greedy search
    distances = copy.deepcopy(d_matrix)
    # GREEDY SEARCH
    solution, solution_coordinates = greedy_search(cities, d_matrix)
    solution_coordinates.append(solution_coordinates[0])

    plt.figure(1)
    plt.subplot(221)
    plot_coordinates(coordinate_array=solution_coordinates)

    # LOCAL SEARCH
    solution_coordinates.pop(-1)
    newRoute, newCoordinates, BestGoalfunctionValues, AllGoalfunctionValues = \
        local_search(solution, distances, solution_coordinates, move_type="2-opt",
                                            max_iter=iters, max_time=max_time, first_x=first_x)

    plt.subplot(222)
    plot_coordinates(coordinate_array=newCoordinates)

    plt.subplot(223)
    plt.plot(range(len(BestGoalfunctionValues)), BestGoalfunctionValues)

    plt.subplot(224)
    plt.plot(range(len(AllGoalfunctionValues)), AllGoalfunctionValues, zorder=1, lw=.5)
    plt.scatter(0, BestGoalfunctionValues[0], c='red', zorder=2)
    plt.show()


if __name__ == '__main__':
    main()
