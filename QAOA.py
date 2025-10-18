
import itertools
from math import pi
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np


import networkx as nx
from neal import SimulatedAnnealingSampler

np.random.seed(42)


def bitstring_to_int(bit_string_sample):
    return int(2 ** np.arange(len(bit_string_sample)) @ bit_string_sample[::-1])

def U_B(beta):
    for wire in range(n_wires):
        qml.RX(2 * beta, wires=wire)

def U_C(gamma):
    for edge in graph:
        qml.CNOT(wires=edge)
        qml.RZ(gamma, wires=edge[1])
        qml.CNOT(wires=edge)
        # Could also do
        # IsingZZ(gamma, wires=edge)

def cut_value(bitstring, edges):
    if isinstance(bitstring, str):
        b = [int(x) for x in bitstring]
    else:
        b = list(bitstring)
    val = 0
    for (i, j) in edges:
        # bitstring jest w kolejności [q0, q1, ...]
        if b[i] != b[j]:
            val += 1
    return val

def maxcut_bruteforce(n, edges):
    best = -1
    best_strings = []
    for bits in itertools.product([0,1], repeat=n):
        v = cut_value(bits, edges)
        if v > best:
            best = v
            best_strings = [bits]
        elif v == best:
            best_strings.append(bits)
    return best, best_strings

n_wires = 5
graph = [(0,1), (0,2), (0,3), (0,4)]

# Klasyczne optimum (dla porównania)
best_classic_val, best_classic_strings = maxcut_bruteforce(n_wires, graph)
print("Klasyczne optimum (bruteforce):", best_classic_val, "\nrozwiązania:", [tuple(b) for b in best_classic_strings])


dev = qml.device("lightning.qubit", wires=n_wires, shots=20)

@qml.qnode(dev)
def circuit(gammas, betas, return_samples=False):
    for wire in range(n_wires):
        qml.Hadamard(wires=wire)
    # p instances of unitary operators
    for gamma, beta in zip(gammas, betas):
        U_C(gamma)
        U_B(beta)

    if return_samples:
        # sample bitstrings to obtain cuts
        return qml.sample()
    # during the optimization phase we are evaluating the objective using expval
    C = qml.sum(*(qml.Z(w1) @ qml.Z(w2) for w1, w2 in graph))
    return qml.expval(C)


def objective(params):
    """Minimize the negative of the objective function C by postprocessing the QNnode output."""
    return -0.5 * (len(graph) - circuit(*params))

def qaoa_maxcut(n_layers=1):
    print(f"\np={n_layers:d}")

    # initialize the parameters near zero
    init_params = 0.01 * np.random.rand(2, n_layers, requires_grad=True)

    # initialize optimizer: Adagrad works well empirically
    opt = qml.AdagradOptimizer(stepsize=0.5)

    # optimize parameters in objective
    params = init_params.copy()
    steps = 30
    for i in range(steps):
        params = opt.step(objective, params)
        if (i + 1) % 5 == 0:
            print(f"Objective after step {i+1:3d}: {-objective(params): .7f}")

    # sample 100 bitstrings by setting return_samples=True and the QNode shot count to 100
    bitstrings = circuit(*params, return_samples=True, shots=100)
    # convert the samples bitstrings to integers
    sampled_ints = [bitstring_to_int(string) for string in bitstrings]

    # print optimal parameters and most frequently sampled bitstring
    counts = np.bincount(np.array(sampled_ints))
    most_freq_bit_string = np.argmax(counts)
    print(f"Optimized parameter vectors:\ngamma: {params[0]}\nbeta:  {params[1]}")
    print(f"Most frequently sampled bit string is: {most_freq_bit_string:04b}")

    return -objective(params), sampled_ints


# perform QAOA on our graph with p=1,2 and keep the lists of sampled integers
int_samples1 = qaoa_maxcut(n_layers=1)[1]
int_samples2 = qaoa_maxcut(n_layers=2)[1]

xticks = range(0, 16)
xtick_labels = list(map(lambda x: format(x, "04b"), xticks))
bins = np.arange(0, 17) - 0.5

fig, _ = plt.subplots(1, 2, figsize=(8, 4))
for i, samples in enumerate([int_samples1, int_samples2], start=1):
    plt.subplot(1, 2, i)
    plt.title(f"n_layers={i}")
    plt.xlabel("bitstrings")
    plt.ylabel("freq.")
    plt.xticks(xticks, xtick_labels, rotation="vertical")
    plt.hist(samples, bins=bins)
plt.tight_layout()
plt.show()

########################################################################
########################################################################
########################################################################

G = nx.Graph()
# Krawędzie
edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 2), (3, 5)]
G.add_edges_from(edges)
N = G.number_of_nodes()

def max_cut_to_qubo(graph):
    """Tworzy macierz Q dla problemu Max-Cut (słownik QUBO)."""
    Q = {}
    
    # Koszty kwadratowe (sprzężenia)
    for u, v in graph.edges():
        # QUBO = Max-Cut (max_edges)
        # Przekształcenie: Max(C) <=> Min(-C)
        # Dla Max-Cut koszt krawędzi (u, v) wynosi 1, gdy x_u != x_v
        # W modelu QUBO/Ising to wymaga manipulacji na kosztach polowych i sprzężeń.
        # W standardowym mapowaniu QUBO dla Max-Cut: Q[u, v]=1 i Q[u,u]=-deg(u)
        
        # Koszty sprzężeń
        Q[(u, v)] = Q.get((u, v), 0) + 1
        
    # Koszty liniowe (pola)
    for i in graph.nodes():
        # Koszt liniowy: -degree(i)
        Q[(i, i)] = -graph.degree(i)
        
    return Q

Q_qubo = max_cut_to_qubo(G)

sampler = SimulatedAnnealingSampler()

# Uruchomienie symulacji
response = sampler.sample_qubo(Q_qubo, num_reads=200, seed=42) # Użycie seed dla powtarzalności

# Pobieramy najlepsze znalezione rozwiązanie (najniższa energia)
best_sample = response.first.sample
best_energy = response.first.energy

# Najlepszy podział
partition_A = [k for k, v in best_sample.items() if v == 0]
partition_B = [k for k, v in best_sample.items() if v == 1]


def calculate_cut_value(graph, partition_A, partition_B):
    cut_value = 0
    set_A = set(partition_A)
    for u, v in graph.edges():
        # Krawędź jest w cięciu, jeśli wierzchołki są w różnych partycjach
        if (u in set_A) != (v in set_A):
            cut_value += 1
    return cut_value

max_cut_value = calculate_cut_value(G, partition_A, partition_B)

print("==============================================")
print("Wyniki Max-Cut (Simulated Annealing)")
print("==============================================")
print(f"Partycja A (Grupa 0): {partition_A}")
print(f"Partycja B (Grupa 1): {partition_B}")
print(f"Maksymalna wartość cięcia: {max_cut_value}")
print(f"Minimalna energia QUBO (powinna być -MaxCut): {best_energy}")

# Ustalenie pozycji wierzchołków
pos = nx.spring_layout(G, seed=42) 

# Definicja kolorów wierzchołków
node_colors = []
for node in G.nodes():
    if node in partition_A:
        node_colors.append('skyblue') # Kolor dla partycji A
    else:
        node_colors.append('salmon')  # Kolor dla partycji B

# Definicja kolorów krawędzi (te w cięciu będą czerwone)
edge_colors = []
cut_edges = []
for u, v in G.edges():
    # Sprawdzamy, czy krawędź jest w cięciu (tj. w różnych partycjach)
    u_in_A = u in partition_A
    v_in_A = v in partition_A
    
    if u_in_A != v_in_A:
        edge_colors.append('red') # Krawędź cięcia
        cut_edges.append((u, v))
    else:
        edge_colors.append('gray') # Krawędź wewnątrzgrupowa

plt.figure(figsize=(10, 7))
plt.title(f"Rozwiązanie Max-Cut (Wartość: {max_cut_value})")

# Rysowanie wierzchołków
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700)

# Rysowanie krawędzi cięcia (czerwone, grubsze)
nx.draw_networkx_edges(G, pos, 
                       edgelist=cut_edges, 
                       edge_color='red', 
                       width=3.0, 
                       label='Cut Edges')

# Rysowanie krawędzi wewnątrzgrupowych (szare, cieńsze)
intra_edges = [e for e in G.edges() if e not in cut_edges]
nx.draw_networkx_edges(G, pos, 
                       edgelist=intra_edges, 
                       edge_color='gray', 
                       width=1.5, 
                       alpha=0.6)

# Rysowanie etykiet wierzchołków
nx.draw_networkx_labels(G, pos, font_color='black', font_weight='bold')

plt.axis('off')
plt.show()

def generate_random_graph(num_nodes, edge_probability, seed=None):
    """
    Generuje losowy graf w modelu Erdos-Renyi (G(n, p)).

    Args:
        num_nodes (int): Liczba wierzchołków (n).
        edge_probability (float): Prawdopodobieństwo utworzenia każdej krawędzi (p),
                                  powinno być w zakresie [0.0, 1.0].
        seed (int, optional): Ziarno dla generatora liczb losowych, 
                              zapewniające powtarzalność. Domyślnie None.

    Returns:
        networkx.Graph: Wygenerowany losowy graf.
    """
    if not (0.0 <= edge_probability <= 1.0):
        raise ValueError("Prawdopodobieństwo krawędzi musi być między 0.0 a 1.0.")
        
    # Używamy funkcji erdos_renyi_graph z networkx
    G = nx.fast_gnp_random_graph(
        n=num_nodes, 
        p=edge_probability, 
        seed=seed
    )
    
    return G

# --- Przykład użycia ---
print("Generowanie losowego grafu G(10, 0.3)...")
# Graf z 10 wierzchołkami, gdzie każda krawędź ma 30% szans na pojawienie się.
random_graph = generate_random_graph(
    num_nodes=10, 
    edge_probability=0.3, 
    seed=42 # Użycie seed zapewnia, że za każdym razem otrzymasz ten sam graf
)

print(f"Liczba wierzchołków: {random_graph.number_of_nodes()}")
print(f"Liczba krawędzi: {random_graph.number_of_edges()}")

# Opcjonalna wizualizacja wygenerowanego grafu
plt.figure(figsize=(8, 6))
plt.title("Losowy Graf G(10, 0.3)")
pos = nx.spring_layout(random_graph, seed=42)
nx.draw(random_graph, pos, 
        with_labels=True, 
        node_color='lightgreen', 
        node_size=600, 
        edge_color='gray', 
        font_weight='bold')
plt.show()