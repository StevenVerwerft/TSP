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

    # otpion parser (only short options)
    try:
        opts, args = getopt.getopt(sys.argv[1:], "n:i:")
    except getopt.GetoptError as err:
            print(err)
            sys.exit(2)

    for o, a in opts:
        if o == '-n':
            n_cities = int(a)
        else:
            n_cities = 100
        if o == '-i':
            filename = a
            cities = readfile(filename)
        else:
            print('random generated TSP will be used')
            cities = [random.sample(range(100), 2) for x in range(n_cities)]

    # generate distance matrix
    d_matrix = distance_matrix(cities)  # fix modification by greedy search
    distances = copy.deepcopy(d_matrix)
    # GREEDY SEARCH
    solution, solution_coordinates = greedy_search(cities, d_matrix)
    solution_coordinates.append(solution_coordinates[0])

    plt.figure(1)
    plt.subplot(121)
    plot_coordinates(coordinate_array=solution_coordinates)

    # LOCAL SEARCH
    solution_coordinates.pop(-1)
    newRoute, newCoordinates = local_search(solution, distances, solution_coordinates, move_type="2-opt",
                                            max_iter=50000, max_time=10)

    plt.subplot(122)
    plot_coordinates(coordinate_array=newCoordinates)
    plt.show()


if __name__ == '__main__':
    main()
