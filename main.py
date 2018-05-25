from functions import *
import random
#random.seed(10)
import matplotlib.pyplot as plt
import sys
import copy


def main():

    # write an arg parser
    if sys.argv[1]:
        n_cities = int(sys.argv[1])
    else:
        n_cities = 100

    if sys.argv[2]:
        filename = sys.argv[2]
        cities = readfile(filename)
    else:
        print('random generated sample will be used')
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
