"""
NSGA-II (Non-dominated Sorting Genetic Algorithm II) Implementation
Multi-objective evolutionary algorithm for architecture search.

Reference: Deb et al., "A Fast and Elitist Multiobjective Genetic Algorithm:
NSGA-II", IEEE TEC, 2002.
"""

import numpy as np
import random
import copy
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from green_nas.search_space import (
    Genome, SEARCH_SPACE, MAX_DEPTH, SEED_GENOMES
)


def dominates(p, q):
    """
    Return True if p dominates q (minimization on all objectives).
    """
    p_fit = p.fitness
    q_fit = q.fitness

    if p_fit is None or q_fit is None:
        return False

    return (all(pv <= qv for pv, qv in zip(p_fit, q_fit)) and
            any(pv < qv for pv, qv in zip(p_fit, q_fit)))


def fast_non_dominated_sort(population):
    """
    Fast non-dominated sorting algorithm (O(MN²)).
    Returns fronts: list of lists, where fronts[0] is Pareto optimal.
    """
    fronts = [[]]

    for p in population:
        p.dominated_solutions = []
        p.domination_count = 0

        for q in population:
            if p is q:
                continue
            if dominates(p, q):
                p.dominated_solutions.append(q)
            elif dominates(q, p):
                p.domination_count += 1

        if p.domination_count == 0:
            p.rank = 0
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in p.dominated_solutions:
                q.domination_count -= 1
                if q.domination_count == 0:
                    q.rank = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    # Remove empty last front
    if len(fronts[-1]) == 0:
        fronts.pop()

    return fronts


def crowding_distance_assignment(front):
    """
    Assign crowding distance to individuals in a front.
    Higher distance = more diversity = preferred during selection.
    """
    n = len(front)
    if n == 0:
        return

    for ind in front:
        ind.crowding_distance = 0.0

    if n <= 2:
        for ind in front:
            ind.crowding_distance = float('inf')
        return

    num_objectives = len(front[0].fitness)

    for m in range(num_objectives):
        front.sort(key=lambda x: x.fitness[m])

        front[0].crowding_distance = float('inf')
        front[-1].crowding_distance = float('inf')

        obj_min = front[0].fitness[m]
        obj_max = front[-1].fitness[m]
        obj_range = obj_max - obj_min

        if obj_range == 0:
            continue

        for i in range(1, n - 1):
            front[i].crowding_distance += (
                (front[i + 1].fitness[m] - front[i - 1].fitness[m]) / obj_range
            )


def tournament_selection(population, tournament_size=2):
    """
    Binary tournament selection.
    Prefer lower rank, then higher crowding distance.
    """
    candidates = random.sample(population, tournament_size)
    candidates.sort(key=lambda x: (x.rank if x.rank is not None else 999,
                                   -x.crowding_distance))
    return candidates[0]


def crossover(parent1, parent2):
    """
    Uniform crossover: for each gene, randomly pick from either parent.
    """
    child_genes = {}
    for key in SEARCH_SPACE.keys():
        child_genes[key] = random.choice([
            parent1.genes[key], parent2.genes[key]
        ])

    # Constraint: bidirectional only for LSTM
    if child_genes['model_type'] != 'lstm':
        child_genes['bidirectional'] = False

    return Genome(genes=child_genes)


def create_offspring(population, offspring_size,
                     crossover_prob=0.8, mutation_prob=0.1):
    """
    Generate offspring via selection, crossover, and mutation.
    """
    offspring = []

    while len(offspring) < offspring_size:
        parent1 = tournament_selection(population)
        parent2 = tournament_selection(population)

        # Crossover
        if random.random() < crossover_prob:
            child = crossover(parent1, parent2)
        else:
            child = Genome(genes=copy.deepcopy(parent1.genes))

        # Mutation
        if random.random() < mutation_prob:
            child.mutate()

        offspring.append(child)

    return offspring[:offspring_size]


def environmental_selection(population, pop_size):
    """
    Elitist selection using non-dominated sorting and crowding distance.
    """
    fronts = fast_non_dominated_sort(population)

    for front in fronts:
        crowding_distance_assignment(front)

    next_gen = []
    for front in fronts:
        if len(next_gen) + len(front) <= pop_size:
            next_gen.extend(front)
        else:
            front.sort(key=lambda x: x.crowding_distance, reverse=True)
            remaining = pop_size - len(next_gen)
            next_gen.extend(front[:remaining])
            break

    return next_gen


def extract_pareto_front(population):
    """Extract the Pareto optimal front (rank 0) from population."""
    fronts = fast_non_dominated_sort(population)
    if len(fronts) > 0:
        return fronts[0]
    return []
