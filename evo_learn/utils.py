import numpy as np
from typing import List


def bin_to_float(bitstring: str) -> float:
    if len(bitstring) != 16 or not all(digit in '01' for digit in bitstring):
        raise ValueError("Invalid bitstring length")
    
    int_val = int(bitstring, 2)
    bytes_val = int_val.to_bytes(2, byteorder='big')
    return np.frombuffer(bytes_val, dtype=np.float16)[0]  # Use np.float16
    


def decode_multidim(chromosome: str, dimensions: int) -> List[float]:
    
    if len(chromosome) % dimensions != 0:
        raise ValueError(f"Chromosome length must be divisible by dimensions ({dimensions})")
    
    bits_per_dim = len(chromosome) // dimensions
    
    if bits_per_dim != 16:
        raise ValueError("Each dimension must use 16 bits")
    
    values = []
    for i in range(dimensions):
        start_idx = i * bits_per_dim
        end_idx = start_idx + bits_per_dim
        dim_bits = chromosome[start_idx:end_idx]
        values.append(bin_to_float(dim_bits))
    
    return values
