from functions import *
import random
import matplotlib.pyplot as plt
import sys
import copy


def main():

    # write an arg parser
    if sys.argv[1]:
        n_cities = int(sys.argv[1])
    else:
        n_cities = 100
    # generate cities (15 cities to start)
    cities = [random.sample(range(100), 2) for x in range(n_cities)]

    # generate distance matrix
    d_matrix = distance_matrix(cities)  # fix modification by greedy search
    distances = copy.deepcopy(d_matrix)
    print(d_matrix)
    # GREEDY SEARCH
    solution, solution_coordinates = greedy_search(cities, d_matrix)
    solution_coordinates.append(solution_coordinates[0])
    print('total distance = ', total_distance(solution, distances))
    print(sorted(solution))
    quit()
    plt.figure(1)

    plt.subplot(121)
    plt.plot([city[0] for city in solution_coordinates], [city[1] for city in solution_coordinates], 'o-')

    # LOCAL SEARCH
    solution_coordinates.pop(-1)
    newRoute, newCoordinates = local_search(solution, distances, solution_coordinates, move_type="2-opt")
    print(newRoute)
    print(distances)
    newCoordinates.append(newCoordinates[0])
    plt.subplot(122)
    plt.plot([city[0] for city in newCoordinates], [city[1] for city in newCoordinates], 'o-')
    plt.show()


if __name__ == '__main__':
    main()