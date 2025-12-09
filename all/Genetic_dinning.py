import random

NUM_PHILOSOPHERS = 5
POP_SIZE = 10
GENERATIONS = 30
MUTATION_RATE = 0.1

def simulate_dining(strategy):
    """
    Simulate a dining session given a strategy.
    strategy[i] = 0 (pick left first) or 1 (pick right first)
    Return fitness based on how many can eat without deadlock.
    """
    forks = [True] * NUM_PHILOSOPHERS  # True = fork is free
    ate = [False] * NUM_PHILOSOPHERS   # Whether philosopher ate
    
    for i in range(NUM_PHILOSOPHERS):
        left = i
        right = (i + 1) % NUM_PHILOSOPHERS
        
        first, second = (left, right) if strategy[i] == 0 else (right, left)
        
        # Try to pick first fork
        if forks[first]:
            forks[first] = False
            # Try to pick second fork
            if forks[second]:
                forks[second] = False
                ate[i] = True
                # Put both forks back
                forks[first] = True
                forks[second] = True
            else:
                # Put back first fork if second not available
                forks[first] = True
        # If can't pick first fork, skip

    fitness_score = sum(ate)
    return fitness_score

def initial_population():
    return [[random.randint(0, 1) for _ in range(NUM_PHILOSOPHERS)] for _ in range(POP_SIZE)]

def fitness(pop):
    return [simulate_dining(ind) for ind in pop]

def selection(pop, fit):
    candidates = random.sample(list(zip(pop, fit)), k=3)
    return max(candidates, key=lambda x: x[1])[0]

def crossover(parent1, parent2):
    point = random.randint(1, NUM_PHILOSOPHERS - 1)
    child = parent1[:point] + parent2[point:]
    return child

def mutate(ind):
    if random.random() < MUTATION_RATE:
        idx = random.randint(0, NUM_PHILOSOPHERS - 1)
        ind[idx] = 1 - ind[idx]  # flip 0↔1
    return ind

def genetic_algorithm_dining():
    population = initial_population()
    best_strategy = None
    best_score = -1

    for gen in range(GENERATIONS):
        fit = fitness(population)
        current_best_idx = fit.index(max(fit))
        if fit[current_best_idx] > best_score:
            best_score = fit[current_best_idx]
            best_strategy = population[current_best_idx]
        
        print(f"Generation {gen}: Best Fitness = {best_score}")

        new_pop = [best_strategy]  # elitism
        while len(new_pop) < POP_SIZE:
            p1 = selection(population, fit)
            p2 = selection(population, fit)
            child = crossover(p1, p2)
            child = mutate(child)
            new_pop.append(child)
        population = new_pop

    print("\nBest Strategy Found:", best_strategy)
    print("Max Philosophers Eating Simultaneously:", best_score)

genetic_algorithm_dining()