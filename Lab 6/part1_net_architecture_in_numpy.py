import numpy as np
from data import LinearlySeparableClasses, NonlinearlySeparableClasses
from visualization_utils import inspect_data, plot_data, x_data_from_grid, visualize_activation_function, \
    plot_two_layer_activations
from random import random
import os
import sys

np.random.seed(int.from_bytes(os.urandom(4), 'little'))

# Przykładowe funkcje aktywacji

def relu(logits):
    return np.maximum(logits, 0)


def sigmoid(logits):
    return 1. / (1. + np.exp(-logits))
    # return np.exp(-np.logaddexp(0, -logits))     # to samo co wyżej, ale stabilne numerycznie


def hardlim(logits):
    return (logits > 0).astype(np.float32)
    # return np.round(sigmoid(logits))             # to samo co wyżej, bez wykorzystywania porównań i rzutowań


def linear(logits):
    return logits


def zad1_single_neuron(student_id):
    gen = LinearlySeparableClasses()
    x, y = gen.generate_data(seed=student_id)
    n_samples, n_features = x.shape

    # zakomentuj, jak juz nie potrzebujesz
    # inspect_data(x, y)
    # plot_data(x, y, plot_xy_range=[-1, 2])

    # model pojedynczego neuronu
    class SingleNeuron:
        def __init__(self, n_in, f_act):
            self.W = 0.01 * np.random.randn(n_in, 1)  # rozmiar W: [n_in, 1]
            self.b = 0.01 * np.random.randn(1)  # rozmiar b: [1]
            self.f_act = f_act

        def forward(self, x_data):
            """
            :param x_data: wejście neuronu: np.array o rozmiarze [n_samples, n_in]
            :return: wyjście neuronu: np.array o rozmiarze [n_samples, 1]
            """
            # TODO (0.5 point)
            return self.f_act(np.dot(x_data, self.W) + self.b)

    # neuron zainicjowany losowymi wagami
    model = SingleNeuron(n_in=n_features, f_act=hardlim)

    # TODO: ustawienie właściwych wag (0.5 point)
    # model.W[:, 0] = [0.8676, 0.0782]
    # model.b[:] = [-0.7943]

    
    def generate_random_individual():
        np.random.seed(int.from_bytes(os.urandom(4), 'little'))
        return [np.random.uniform(-1, 1, size=2), np.random.uniform(-1, 1)]


    def fitness(individual):
        model.W[:, 0] = individual[0]
        model.b[:] = individual[1]

        y_pred = model.forward(x)
        return np.mean(y == y_pred)
    
    # implement training
    n_steps = 1000
    best_solution = None
    best_fitness = 0  
    population = [generate_random_individual() for _ in range(100)]

    for step in range(n_steps):
        # evaluate fitness
        fitnesses = [fitness(individual) for individual in population]
        best_individual = population[np.argmax(fitnesses)]
        best_individual_fitness = np.max(fitnesses)
        if best_individual_fitness > best_fitness:
            best_solution = best_individual
            best_fitness = best_individual_fitness
            if best_fitness == 1:
                break
        print(f'Step={step}, best_fitness={best_fitness:.6f}')

        # selection
        selected = np.random.choice(range(len(population)), size=20, replace=False)
        selected_individuals = [population[i] for i in selected]

        # crossover
        new_population = []
        for i in range(100):
            parent1 = selected_individuals[np.random.randint(20)]
            parent2 = selected_individuals[np.random.randint(20)]
            child = [[np.random.choice([parent1[0][0], parent2[0][0]]),np.random.choice([parent1[0][1], parent2[0][1]])]]
            child.append(np.random.choice([parent1[1], parent2[1]]))
            new_population.append(child)

        # mutation
        for i in range(100):
            if np.random.rand() < 0.1:
                new_population[i] = generate_random_individual()

        population = new_population

    print(f'Best solution: {best_solution}')
    print(f'Best fitness: {best_fitness}')
    

    model.W[:, 0] = best_solution[0]
    model.b[:] = best_solution[1]




    # działanie i ocena modelu
    y_pred = model.forward(x)
    print(f'Accuracy = {np.mean(y == y_pred) * 100}%')

    # test na całej przestrzeni wejść (z wizualizacją)
    x_grid = x_data_from_grid(min_xy=-1, max_xy=2, grid_size=1000)
    y_pred_grid = model.forward(x_grid)
    plot_data(x, y, plot_xy_range=[-1, 2], x_grid=x_grid,
              y_grid=y_pred_grid, title='Linia decyzyjna neuronu')


def zad2_two_layer_net(student_id):
    gen = NonlinearlySeparableClasses()
    x, y = gen.generate_data(seed=student_id)
    n_samples, n_features = x.shape

    # zakomentuj, jak juz nie potrzebujesz
    # inspect_data(x, y)
    # plot_data(x, y, plot_xy_range=[-1, 2])

    # warstwa czyli n_out pojedynczych, niezależnych neuronów operujących na tym samym wejściu\
    # (i-ty neuron ma swoje parametry w i-tej kolumnie macierzy W i na i-tej pozycji wektora b)
    class DenseLayer:
        def __init__(self, n_in, n_out, f_act):
            # rozmiar W: ([n_in, n_out])
            self.W = 0.01 * np.random.randn(n_in, n_out)
            self.b = 0.01 * np.random.randn(n_out)  # rozmiar b  ([n_out])
            self.f_act = f_act

        def forward(self, x_data):
            # TODO
            return self.f_act(np.dot(x_data, self.W) + self.b)

    # TODO: warstwy mozna składać w wiekszy model
    class SimpleTwoLayerNetwork:
        def __init__(self, n_in, n_hidden, n_out):
            self.hidden_layer = DenseLayer(n_in, n_hidden, relu)
            self.output_layer = DenseLayer(n_hidden, n_out, hardlim)

        def forward(self, x_data):
            # TODO
            return self.output_layer.forward(self.hidden_layer.forward(x_data))

    # model zainicjowany losowymi wagami
    model = SimpleTwoLayerNetwork(n_in=n_features, n_hidden=2, n_out=1)

    # TODO: ustawienie właściwych wag

    # model.hidden_layer.W[:, 0] = [0.9668, -0.6539]
    # model.hidden_layer.W[:, 1] = [-0.9947, 0.0912]
    # model.hidden_layer.b[:] = [-0.5995, 0.5665]
    # model.output_layer.W[:, 0] = [-0.8510, 04-0.3192]
    # model.output_layer.b[:] = [0.1090]

    def generate_random_individual():
        np.random.seed(int.from_bytes(os.urandom(4), 'little'))
        return [np.random.uniform(-1, 1, size=2) for _ in range(4)]+[np.random.uniform(-1, 1)]


    def fitness(individual):
        model.hidden_layer.W[:, 0] = individual[0]
        model.hidden_layer.W[:, 1] = individual[1]
        model.hidden_layer.b[:] = individual[2]
        model.output_layer.W[:, 0] = individual[3]
        model.output_layer.b[:] = individual[4]

        y_pred = model.forward(x)
        return np.mean(y == y_pred)
    
    # implement training
    n_steps = 1000
    best_solution = None
    best_fitness = 0  
    population = [generate_random_individual() for _ in range(100)]

    for step in range(n_steps):
        # evaluate fitness
        fitnesses = [fitness(individual) for individual in population]
        best_individual = population[np.argmax(fitnesses)]
        best_individual_fitness = np.max(fitnesses)
        if best_individual_fitness > best_fitness:
            best_solution = best_individual
            best_fitness = best_individual_fitness
            if best_fitness == 1:
                break
        print(f'Step={step}, best_fitness={best_fitness:.6f}')

        # selection
        selected = np.random.choice(range(len(population)), size=20, replace=False)
        selected_individuals = [population[i] for i in selected]

        # crossover
        new_population = []
        for i in range(100):
            parent1 = selected_individuals[np.random.randint(20)]
            parent2 = selected_individuals[np.random.randint(20)]
            child = [[np.random.choice([parent1[j][0], parent2[j][0]]),np.random.choice([parent1[j][1], parent2[j][1]])] for j in range(4)]
            child.append(np.random.choice([parent1[4], parent2[4]]))
            new_population.append(child)

        # mutation
        for i in range(100):
            if np.random.rand() < 0.1:
                new_population[i] = generate_random_individual()

        population = new_population

    print(f'Best solution: {best_solution}')
    print(f'Best fitness: {best_fitness}')
    

    model.hidden_layer.W[:, 0] = best_solution[0]
    model.hidden_layer.W[:, 1] = best_solution[1]
    model.hidden_layer.b[:] = best_solution[2]
    model.output_layer.W[:, 0] = best_solution[3]
    model.output_layer.b[:] = best_solution[4]

    # działanie i ocena modelu
    y_pred = model.forward(x)
    print(f'Accuracy = {np.mean(y == y_pred) * 100}%')

    plot_two_layer_activations(model, x, y)


if __name__ == '__main__':
    # visualize_activation_function(hardlim)

    student_id = 196766       # Twój numer indeksu, np. 102247

    zad1_single_neuron(student_id)
    zad2_two_layer_net(student_id)
