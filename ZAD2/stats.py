import numpy as np
import pandas as df
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from util import random_matrix, random_vector_T
from benchmark import *
from gauss import *
from inverse import *
from lu import *


def create_statistics(n: int, m:int, k: int, func, name, mul):
    results = []
    for i in range(n,m,10):
        A = random_matrix(i)
        b = random_vector_T(i)
        for j in range(k):
                if name == "gauss":
                    _, stats = func(A,b, mul)
                _, stats = func(A, mul)
                stats['size'] = i
                stats['method'] = name
                results.append(stats)

    full_df = df.concat(results, ignore_index=True)
    avg_df = full_df.groupby(["size", "method"], as_index=False).mean(numeric_only=True)
    return avg_df


def plot_metric(df, metric, functionName, mul):
    plt.figure(figsize=(8,5))

    for method, group in df.groupby('method'):
        plt.plot(group['size'], group[metric], label=method)

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) 
    plt.xlabel("Size (n)")
    plt.ylabel(metric)
    plt.title(f"{metric} for matrix {functionName} of size n using {mul} multiplication" )
    plt.legend()
    plt.grid(True)
    plt.show()

@benchmark()
def gauss_elimination_benchmark(A: np.ndarray, b: np.ndarray, mul):
     return gauss_elimination(A,b, mul)

@benchmark()
def inverse_benchmark(A: np.ndarray, mul):
     return inverse(A, mul)

@benchmark()
def lu_factorization_benchmark(A: np.ndarray, mul):
     return lu_factorization(A, mul)