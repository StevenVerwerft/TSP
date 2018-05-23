from functions import *
import random
import matplotlib.pyplot as plt
import sys
import copy


def main():
    if sys.argv[1]:
        n_cities = int(sys.argv[1])
    else:
        n_cities = 100
    # generate cities (15 cities to start)
    cities = [random.sample(range(100), 2) for x in range(n_cities)]

    # generate distance matrix
    d_matrix = distance_matrix(cities)  # will be modified by greedy algorithm
    distances = copy.deepcopy(d_matrix)
    # make diagonal infinity
    for i in range(n_cities):
        d_matrix[i, i] = np.inf

    # get nearest city
    def greedy_search(cities, distance_matrix):

        # make a greedy solution
        solution = []
        solution_coordinates = []
        # 1 pick random first city
        starting = random.choice(range(n_cities))
        solution.append(starting)
        # distance_matrix[:, starting], distance_matrix[starting, :] = np.inf, np.inf
        solution_coordinates.append(cities[starting])

        while len(solution) < len(cities):

            previous = solution[-1]
            neighbours = np.append(d_matrix[:previous,previous], d_matrix[previous, previous:])
            next = np.argmin(neighbours)  # position of nearest neighbour
            print('next: ', next)
            solution.append(next)
            solution_coordinates.append(cities[next])
            distance_matrix[:, previous], distance_matrix[previous, :] = np.inf, np.inf

        return solution, solution_coordinates

    def local_search(route, distance_matrix, move_type='2-opt'):
        pass

    solution, solution_coordinates = greedy_search(cities, d_matrix)
    solution_coordinates.append(solution_coordinates[0])
    del d_matrix  # delete garbage
    print('total distance = ', total_distance(solution, distances))
    plt.plot([city[0] for city in solution_coordinates], [city[1] for city in solution_coordinates], 'o-')
    plt.show()


if __name__ == '__main__':
    main()