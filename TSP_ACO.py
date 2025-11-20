import numpy as np
import matplotlib.pyplot as plt
import time
import random
from math import sqrt

# ---------------- Utility Functions ---------------- #
def euclidean_distance(a, b):
    return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def create_distance_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                D[i, j] = np.inf
            else:
                D[i, j] = euclidean_distance(coords[i], coords[j])
    return D

def initialize_pheromone_matrix(n, tau0):
    return np.full((n, n), tau0)

def choose_next_city(probabilities):
    r = random.random()
    cum = 0.0
    for i, p in enumerate(probabilities):
        cum += p
        if r <= cum:
            return i
    return len(probabilities) - 1

def build_solution(pheromone, distance, alpha, beta):
    n = pheromone.shape[0]
    tour = []
    start = random.randrange(n)
    tour.append(start)
    visited = set(tour)
    current = start
    while len(tour) < n:
        denom = 0.0
        probs = np.zeros(n)
        for j in range(n):
            if j in visited:
                probs[j] = 0.0
            else:
                probs[j] = (pheromone[current, j]**alpha) * ((1.0 / distance[current, j])**beta)
                denom += probs[j]
        if denom == 0:
            choices = [j for j in range(n) if j not in visited]
            next_city = random.choice(choices)
        else:
            probs = probs / denom
            next_city = choose_next_city(probs)
        tour.append(next_city)
        visited.add(next_city)
        current = next_city
    return tour

def tour_length(tour, distance):
    L = 0.0
    for i in range(len(tour) - 1):
        L += distance[tour[i], tour[i + 1]]
    L += distance[tour[-1], tour[0]]  # Return to start
    return L

# ---------------- ACO Core Algorithm ---------------- #
def aco_tsp(coords, n_ants=20, n_iterations=200, alpha=1.0, beta=5.0, rho=0.5, Q=100.0, tau0=None, verbose=True):
    n = len(coords)
    D = create_distance_matrix(coords)
    if tau0 is None:
        tau0 = 1.0 / (n * np.mean(D[np.isfinite(D)]))
    pheromone = initialize_pheromone_matrix(n, tau0)
    best_tour = None
    best_length = float('inf')
    history = []

    for it in range(n_iterations):
        all_tours = []
        all_lengths = []
        for ant in range(n_ants):
            tour = build_solution(pheromone, D, alpha, beta)
            L = tour_length(tour, D)
            all_tours.append(tour)
            all_lengths.append(L)
            if L < best_length:
                best_length = L
                best_tour = tour.copy()

        pheromone *= (1 - rho)  # Evaporation
        for tour, L in zip(all_tours, all_lengths):  # Deposit
            deposit = Q / L
            for i in range(len(tour) - 1):
                a, b = tour[i], tour[i + 1]
                pheromone[a, b] += deposit
                pheromone[b, a] += deposit
            pheromone[tour[-1], tour[0]] += deposit
            pheromone[tour[0], tour[-1]] += deposit

        history.append(best_length)
        if verbose and (it % max(1, n_iterations // 10) == 0 or it == n_iterations - 1):
            print(f"Iteration {it + 1}/{n_iterations} — Best length: {best_length:.4f}")

    return {
        "best_tour": best_tour,
        "best_length": best_length,
        "history": history,
        "distance_matrix": D
    }

# ---------------- Run Example ---------------- #
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    n_cities = 20
    coords = np.random.rand(n_cities, 2) * 100  # Random cities

    start_time = time.time()
    result = aco_tsp(coords, n_ants=40, n_iterations=200, alpha=1.0, beta=5.0, rho=0.45, Q=100.0, verbose=True)
    end_time = time.time()

    best_tour = result["best_tour"]
    best_length = result["best_length"]
    history = result["history"]

    print("\nACO Results:")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Best tour length: {best_length:.4f}")
    print("Best tour:", best_tour)

    # Plot the best tour
    plt.figure(figsize=(8, 6))
    xs, ys = coords[:, 0], coords[:, 1]
    plt.scatter(xs, ys)
    for i, (x, y) in enumerate(coords):
        plt.text(x + 0.5, y + 0.5, str(i), fontsize=9)
    path = best_tour + [best_tour[0]]
    plt.plot(coords[path, 0], coords[path, 1], 'b-', lw=1.5)
    plt.title("Best Tour Found by ACO")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    # Plot convergence curve
    plt.figure(figsize=(8, 4))
    plt.plot(history, 'r')
    plt.title("Convergence Curve (Best Length vs Iterations)")
    plt.xlabel("Iteration")
    plt.ylabel("Best Tour Length")
    plt.grid(True)
    plt.show()
