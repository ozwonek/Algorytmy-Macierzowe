from Binet import *
from Strassen import *
from helpers import *
import numpy as np
import pandas as df
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from Ai import Ai, benchmark_Ai

def random_matrix(n: int, m: int):
    matrix = np.random.random((n, m)) + 0.00000001
    return np.frompyfunc(Float, 1, 1)(matrix)

functions = [binet_two_split_benchmark, binet_without_padding_benchmark, strassen_without_padding_benchmark, numpy_benchmark]
names = ["Binet dzielenie na 2", "Binet bez paddingu", "Strassen bez paddingu", "Wbudowane"]


def create_statistics(n: int, m:int, k: int, functions, names):
    results = []
    for i in range(n,m,10):
        A = random_matrix(i, i)
        B = random_matrix(i, i)
        for j in range(k):
            for func, name in zip(functions, names):
                _, stats = func(A, B)
                stats['size'] = i
                stats['method'] = name
                results.append(stats)

    full_df = df.concat(results, ignore_index=True)
    avg_df = full_df.groupby(["size", "method"], as_index=False).mean(numeric_only=True)
    return avg_df

def create_statistics_for_ai(m: int, k: int):
    results = []
    a, b = 4, 5
    while a * b <= m:
        A = random_matrix(a, b)
        B = random_matrix(b, b)
        for j in range(k):
            _, stats = benchmark_Ai(A, B)
            stats['size'] = f"{a * b}"
            stats['method'] = "Ai"
            results.append(stats)
        a *= 4
        b *= 5
    
    full_df = df.concat(results, ignore_index=True)
    avg_df = full_df.groupby(["size", "method"], as_index=False).mean(numeric_only=True)
    return avg_df


def plot_metric(df, metric):
    plt.figure(figsize=(8,5))

    for method, group in df.groupby('method'):
        plt.plot(group['size'], group[metric], label=method)

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) 
    plt.xlabel("Size (n)")
    plt.ylabel(metric)
    plt.title(f"{metric} for matrix multiplication of size n" )
    plt.legend()
    plt.grid(True)
    plt.show()
