import numpy as np
from typing import List, Union, Callable
from copy import deepcopy
from .utils import bin_to_float, decode_multidim


class Qubit():
    superposition_amplitude = 1 / np.sqrt(2)

    def __init__(self, measurement: bool) -> None:
        self.measurement: bool = measurement
        self.alpha: float = self.superposition_amplitude
        self.beta: float = self.superposition_amplitude
        self.superposition: bool = False
        
    def apply_hadamard(self) -> 'Qubit':
        superposition_amplitude = 1 / np.sqrt(2)

        self.alpha = superposition_amplitude
        self.beta = superposition_amplitude
        self.superposition = True
        
        return self
        
    def apply_xgate(self) -> None:
        self.measurement = not self.measurement
        
    def measure(self) -> bool:
        if not self.superposition:
            return self.measurement
        
        random_var = np.random.uniform(0, 1)
        self.measurement = np.power(self.beta, 2) >= random_var
        self.superposition = False
        
        return self.measurement
    
    def __str__(self) -> str:
        return f'Qubit[superposition: {self.superposition}, [measurement: {self.measurement}]]'



def single_qubit_measurement(qubit: Qubit) -> bool:
    return qubit.measure()

def full_measurement(quantum_chromosome: List[Qubit]) -> str:
    measured_bits = []
    
    for gene in quantum_chromosome:
        measured_value = gene.measure()
        measured_bits.append(str(int(measured_value)))
        
    return ''.join(measured_bits)

def crossover(
        queen: List[Qubit],
        male: List[Qubit],
        num_swap_qubits: int
    ) -> str:

    quantum_chromosome = deepcopy(queen)
    random_ind = np.random.randint(0, len(queen) - num_swap_qubits)
    
    for i in range(random_ind, random_ind + num_swap_qubits):
        if queen[i].measure() == male[i].measure():
            quantum_chromosome[i].apply_hadamard()
        else:
            quantum_chromosome[i].apply_xgate()
    
    return full_measurement(quantum_chromosome)



def evolution(
        population_size: int,
        fitness: Callable[[Union[float, List[float]]], float],
        dimensions: int = 1,
        qubits_per_dim: int = 16,
        num_males: int = 20,
        num_elites: int = 20,
        max_iteration: int = 500,
        crossover_size: int = 3,
        maximize: bool = False
    ) -> (Union[float, List[float]], dict, List[str]):
    
    qubits_in_indiv = qubits_per_dim * dimensions
    population = []

    # Initialize the population (each individual is a quantum chromosome)
    for i in range(population_size):
        quantum_chromosome = [Qubit(False).apply_hadamard() for _ in range(qubits_in_indiv)]
        quantum_chromosome_measured = [str(int(qubit.measure())) for qubit in quantum_chromosome]
        new_chromosome = ''.join(quantum_chromosome_measured)
        population.append(new_chromosome)
    
    if dimensions == 1:
        fitness_key = lambda x: fitness(bin_to_float(x))
    else:
        fitness_key = lambda x: fitness(decode_multidim(x, dimensions))
    
    population.sort(key=fitness_key, reverse=maximize)
    
    queen = population[0]
    males = population[1:num_males]
    
    # --- History recording ---
    history = {"iteration": [], "queen_fitness": [], "queen_value": []}
    if dimensions == 1:
        queen_value = bin_to_float(queen)
    else:
        queen_value = decode_multidim(queen, dimensions)
    
    history["iteration"].append(0)
    history["queen_fitness"].append(fitness(queen_value))
    history["queen_value"].append(queen_value)
    # --- End of history recording ---
    
    for iteration in range(max_iteration):
        for i in range(population_size - num_elites, population_size):
            selected_male = males[np.random.randint(0, num_males - 1)]
            population[i] = crossover(
                [Qubit(bool(int(digit))) for digit in queen],
                [Qubit(bool(int(digit))) for digit in selected_male],
                crossover_size
            )
        
        population.sort(key=fitness_key, reverse=maximize)
        queen = population[0]
        males = population[1:num_males]
        
        if dimensions == 1:
            queen_value = bin_to_float(queen)
            current_fitness = fitness(queen_value)
            print(f'Iteration: {iteration + 1}, Queen: {queen_value}, Fitness: {current_fitness}')
        else:
            queen_value = decode_multidim(queen, dimensions)
            current_fitness = fitness(queen_value)
            print(f'Iteration: {iteration + 1}, Queen: {queen_value}, Fitness: {current_fitness}')
        
        # Record the current iteration info into history
        history["iteration"].append(iteration + 1)
        history["queen_fitness"].append(current_fitness)
        history["queen_value"].append(queen_value)
    
    if dimensions == 1:
        return bin_to_float(queen), history, population
    else:
        return decode_multidim(queen, dimensions), history, population
