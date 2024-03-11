from itertools import compress
import random
import time
import matplotlib.pyplot as plt
import numpy as np

from data import *


def initial_population(individual_size, population_size):
    return [[random.choice([True, False]) for _ in range(individual_size)] for _ in range(population_size)]


def fitness(items, knapsack_max_capacity, individual):
    total_weight = sum(compress(items['Weight'], individual))
    if total_weight > knapsack_max_capacity:
        return 0
    return sum(compress(items['Value'], individual))


def population_best(items, knapsack_max_capacity, population):
    best_individual = None
    best_individual_fitness = -1
    for individual in population:
        individual_fitness = fitness(items, knapsack_max_capacity, individual)
        if individual_fitness > best_individual_fitness:
            best_individual = individual
            best_individual_fitness = individual_fitness
    return best_individual, best_individual_fitness


items, knapsack_max_capacity = get_big()
print(items)

population_size = 100
generations = 200
n_selection = 20
n_elite = 1

start_time = time.time()
best_solution = None
best_fitness = 0
population_history = []
best_history = []
population = initial_population(len(items), population_size)
for _ in range(generations):
    population_history.append(population.copy())

    # TODO: implement genetic algorithm

    # Selection using roulette wheel selection without repetition
    fitnesses = [fitness(items, knapsack_max_capacity, individual)
                 for individual in population]
    total_fitness = sum(fitnesses)
    selected = set()

    while len(selected) < n_selection:
        r = random.uniform(0, total_fitness)
        acc = 0
        for i, individual in enumerate(population):
            acc += fitnesses[i]
            if acc >= r:
                selected.add(i)
                break

    selected = list(selected)
    selected = [population[i] for i in selected]

    # for _ in range(n_selection):
    #     r = random.uniform(0, total_fitness)
    #     acc = 0
    #     for i, individual in enumerate(population):
    #         acc += fitnesses[i]
    #         if acc >= r:
    #             selected.append(individual)
    #             break


    # Crossover
    children = []
    for i in range(0, n_selection, 2):
        parent1 = selected[i]
        parent2 = selected[i+1]
        crossover_point = random.randint(0, len(parent1))
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        children.append(child1)
        children.append(child2)

    # Mutation
    for child in children:
        index = random.randint(0, len(child)-1)
        child[index] = not child[index]

    # Elitism
    # select n_elite best individuals
    index = np.argsort(fitnesses, axis=0)[::-1].tolist()
    elite = []
    for i in range(n_elite):
        elite.append(population[index[i]])

    for i in range(n_selection):
        index = random.randint(0, len(population)-1)
        population[index] = children[i]
    for i in range(n_elite):
        index = random.randint(0, len(population)-1)
        population[index] = elite[i]

    best_individual, best_individual_fitness = population_best(
        items, knapsack_max_capacity, population)
    if best_individual_fitness > best_fitness:
        best_solution = best_individual
        best_fitness = best_individual_fitness
    best_history.append(best_fitness)

end_time = time.time()
total_time = end_time - start_time
print('Best solution:', list(compress(items['Name'], best_solution)))
print('Best solution value:', best_fitness)
print('Time: ', total_time)

# plot generations
x = []
y = []
top_best = 10
for i, population in enumerate(population_history):
    plotted_individuals = min(len(population), top_best)
    x.extend([i] * plotted_individuals)
    population_fitnesses = [
        fitness(items, knapsack_max_capacity, individual) for individual in population]
    population_fitnesses.sort(reverse=True)
    y.extend(population_fitnesses[:plotted_individuals])
plt.scatter(x, y, marker='.')
plt.plot(best_history, 'r')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()
