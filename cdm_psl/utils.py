import numpy as np
from scipy.spatial.distance import cdist


def sbx(sorted_pop, eta=15):
    n_pop, n_var = sorted_pop.shape
    new_pop = np.empty_like(sorted_pop)

    for i in range(0, n_pop, 2):
        parent1, parent2 = (
            sorted_pop[np.random.choice(n_pop)],
            sorted_pop[np.random.choice(n_pop)],
        )
        rand = np.random.random(n_var)
        gamma = np.empty_like(rand)
        mask = rand <= 0.5
        gamma[mask] = (2 * rand[mask]) ** (1 / (eta + 1))
        gamma[~mask] = (1 / (2 * (1 - rand[~mask]))) ** (1 / (eta + 1))

        offspring1 = 0.5 * ((1 + gamma) * parent1 + (1 - gamma) * parent2)
        offspring2 = 0.5 * ((1 - gamma) * parent1 + (1 + gamma) * parent2)

        new_pop[i] = offspring1
        if i + 1 < n_pop:
            new_pop[i + 1] = offspring2

    return new_pop


def environment_selection(population, n):
    """
    environmental selection in SPEA-2
    :param population: current population
    :param n: number of selected individuals
    :return: next generation population
    """
    fitness = cal_fit(population)
    index = np.nonzero(fitness < 1)[0]
    if len(index) < n:
        rank = np.argsort(fitness)
        index = rank[:n]
    elif len(index) > n:
        del_no = trunc(population[index, :], len(index) - n)
        index = np.setdiff1d(index, index[del_no])

    population = population[index, :]
    return population, index


def trunc(pop_obj, k):
    n, m = np.shape(pop_obj)
    distance = cdist(pop_obj, pop_obj)
    distance[np.eye(n) > 0] = np.inf
    del_no = np.ones(n) < 0
    while np.sum(del_no) < k:
        remain = np.nonzero(np.logical_not(del_no))[0]
        temp = np.sort(distance[remain, :][:, remain], axis=1)
        rank = np.argsort(temp[:, 0])
        del_no[remain[rank[0]]] = True
    return del_no


def cal_fit(pop_obj):
    n, m = np.shape(pop_obj)
    dominance = np.ones((n, n)) < 0
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            k = int(np.any(pop_obj[i, :] < pop_obj[j, :])) - int(
                np.any(pop_obj[i, :] > pop_obj[j, :])
            )
            if k == 1:
                dominance[i, j] = True
            elif k == -1:
                dominance[j, i] = True

    s = np.sum(dominance, axis=1, keepdims=True)

    r = np.zeros(n)
    for i in range(n):
        r[i] = np.sum(s[dominance[:, i]])

    distance = cdist(pop_obj, pop_obj)
    distance[np.eye(n) > 0] = np.inf
    distance = np.sort(distance, axis=1)
    d = 1 / (distance[:, int(np.sqrt(n))] + 2)

    fitness = r + d
    return fitness


def pm_mutation(pop_dec, boundary):

    pro_m = 1
    dis_m = 20
    pop_dec = pop_dec[: (len(pop_dec) // 2) * 2, :]
    n, d = np.shape(pop_dec)

    site = np.random.random((n, d)) < pro_m / d
    mu = np.random.random((n, d))
    temp = site & (mu <= 0.5)
    lower, upper = np.tile(boundary[0], (n, 1)), np.tile(boundary[1], (n, 1))
    pop_dec = np.minimum(np.maximum(pop_dec, lower), upper)
    norm = (pop_dec[temp] - lower[temp]) / (upper[temp] - lower[temp])
    pop_dec[temp] += (upper[temp] - lower[temp]) * (
        np.power(
            2.0 * mu[temp] + (1.0 - 2.0 * mu[temp]) * np.power(1.0 - norm, dis_m + 1.0),
            1.0 / (dis_m + 1),
        )
        - 1.0
    )
    temp = site & (mu > 0.5)
    norm = (upper[temp] - pop_dec[temp]) / (upper[temp] - lower[temp])
    pop_dec[temp] += (upper[temp] - lower[temp]) * (
        1.0
        - np.power(
            2.0 * (1.0 - mu[temp])
            + 2.0 * (mu[temp] - 0.5) * np.power(1.0 - norm, dis_m + 1.0),
            1.0 / (dis_m + 1.0),
        )
    )
    offspring_dec = np.maximum(np.minimum(pop_dec, upper), lower)
    return offspring_dec


def sort_population(pop, label_matrix, conf_matrix):
    size = len(pop)
    domination_counts = []
    avg_confidences = []
    for i in range(size):
        count = sum(label_matrix[j, i] == 2 for j in range(size))
        domination_counts.append(count)
        confidence = sum(
            conf_matrix[j, i] for j in range(size) if label_matrix[j, i] == 2
        )
        avg_confidences.append(confidence / (count if count > 0 else 1))

    sorted_pop = sorted(
        zip(pop, domination_counts, avg_confidences),
        key=lambda x: (x[1], -x[2]),
    )
    sorted_pop = [x[0] for x in sorted_pop]

    sorted_pop_array = np.array(sorted_pop)

    return sorted_pop_array
