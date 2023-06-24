import csv
import math
import random
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import fractional_matrix_power
from sklearn.datasets import make_moons, make_circles

def read_csv_file(file_name, num_nodes):
    coordinates = []
    binary_labels = []
    with open(file_name, 'r') as csv_file:
        ROWS = list(csv.reader(csv_file))[1:num_nodes + 1]
        random.shuffle(ROWS)
        coordinates = np.array(ROWS)[:,:2].astype(float)
        binary_labels = np.array(ROWS)[:,2].astype(int)
    return np.array(coordinates), np.array(binary_labels)

def write_csv_file(coordinates, binary_labels, file_name):
    ROWS = ['x,y,l']
    for i in range(len(binary_labels)):
        coordinates_string = ','.join(map(str, coordinates[i]))
        ROWS.append(f'{coordinates_string},{binary_labels[i]}')
        
    with open(file_name, 'w') as csv_file:
        for ROW in ROWS:
            csv_file.write(ROW + '\n')
    
def propagate_labels(num_nodes, num_samples_labeled, num_iterations, dataset):
    coordinates, labels = dataset

    samples_labeled = np.concatenate(((labels[:num_samples_labeled, None] == np.arange(2)).astype(float), np.zeros((num_nodes - num_samples_labeled, 2))))
    
    sigma = 0.1
    
    distance_matrix = cdist(coordinates, coordinates, 'euclidean')
    
    rbf_kernel = lambda x, sigma: math.exp((-x)/(2*(math.pow(sigma,2))))
    vectorized_rbf = np.vectorize(rbf_kernel)
    
    weight_matrix = vectorized_rbf(distance_matrix, sigma)
    
    np.fill_diagonal(weight_matrix, 0)

    sum_lines = np.sum(weight_matrix, axis=1)
    
    diagonal_matrix = np.diag(sum_lines)
    diagonal_matrix = fractional_matrix_power(diagonal_matrix, -0.5)
    
    scale_matrix = np.dot(np.dot(diagonal_matrix, weight_matrix), diagonal_matrix)

    alpha = 0.99

    propagated_labels = np.dot(scale_matrix, samples_labeled) * alpha + (1 - alpha) * samples_labeled

    for _ in range(num_iterations):
        propagated_labels = np.dot(scale_matrix, propagated_labels) * alpha + (1 - alpha) * samples_labeled

    final_labels = np.zeros_like(propagated_labels)
    final_labels [np.arange(len(propagated_labels)), propagated_labels.argmax(1)] = 1

    binary_labels = [1 if x == 0 else 0 for x in final_labels[0:, 0]]
    
    return coordinates, binary_labels
    
def main():
    num_nodes = 1000
    num_samples_labeled = int(num_nodes * 0.2)
    num_iterations = 100

    sklearn_moons_coordinates, sklearn_moons_labels = make_moons(num_nodes, shuffle=True, noise=0.1, random_state=None)
    write_csv_file(sklearn_moons_coordinates, sklearn_moons_labels, 'sklearn_moons.csv')
    
    sklearn_circles_coordinates, sklearn_circles_labels = make_circles(num_nodes, shuffle=True, noise=0.1, random_state=None, factor=0.5)
    write_csv_file(sklearn_circles_coordinates, sklearn_circles_labels, 'sklearn_circles.csv')
    
    statsim_spirals_coordinates, statsim_spirals_labels = propagate_labels(num_nodes, num_samples_labeled, num_iterations, read_csv_file('statsim_spirals.csv', num_nodes))
    write_csv_file(statsim_spirals_coordinates, statsim_spirals_labels, 'propagated_statsim_spirals.csv')

    statsim_moons_coordinates, statsim_moons_labels = propagate_labels(num_nodes, num_samples_labeled, num_iterations, read_csv_file('statsim_moons.csv', num_nodes))
    write_csv_file(statsim_moons_coordinates, statsim_moons_labels, 'propagated_statsim_moons.csv')
    
    make_moons_coordinates, make_moons_labels = propagate_labels(num_nodes, num_samples_labeled, num_iterations, (sklearn_moons_coordinates, sklearn_moons_labels))
    write_csv_file(make_moons_coordinates, make_moons_labels, 'propagated_sklearn_moons.csv')
    
    make_circles_coordinates, make_circles_labels = propagate_labels(num_nodes, num_samples_labeled, num_iterations, (sklearn_circles_coordinates, sklearn_circles_labels))
    write_csv_file(make_circles_coordinates, make_circles_labels, 'propagated_sklearn_circles.csv')
    
main()