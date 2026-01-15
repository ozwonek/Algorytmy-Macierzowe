import numpy as np
import pandas as df
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from util import random_matrix, random_vector_T
from benchmark import *
from gauss import *
from inverse import *
from lu import *
from scipy.optimize import curve_fit


def create_statistics(n: int, m:int, k: int,  func, name):
    results = []
    for i in range(n,m,10):
        A = random_matrix(i)
        b = random_vector_T(i)
        for j in range(k):
                if name == "gauss":
                    _, stats = func(A,b)
                else:     
                    _, stats = func(A)
                stats['size'] = i
                stats['method'] = name
                results.append(stats)

    full_df = df.concat(results, ignore_index=True)
    avg_df = full_df.groupby(["size", "method"], as_index=False).mean(numeric_only=True)
    return avg_df


def plot_metric(df, metric, functionName):
    plt.figure(figsize=(8,5))

    for method, group in df.groupby('method'):
        plt.plot(group['size'], group[metric], label=method)

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) 
    plt.xlabel("Size (n)")
    plt.ylabel(metric)
    plt.title(f"{metric} for matrix {functionName} of size n" )
    plt.legend()
    plt.grid(True)
    plt.show()

def fit_curve(name1, name2, dataframe):
    x_data = [i for i in dataframe[name1]]
    y_data = dataframe[name2]
    def power_law(x, a, k):
        return a*x**k

    params, covariance = curve_fit(power_law, x_data, y_data, p0=[1.0, 1.0])
    a, k = params
    print(f"Exponent k is approximately: {k}")
    print(f"contant a is approximately: {a}")
    return a, k

@benchmark()
def gauss_elimination_benchmark(A: np.ndarray, b: np.ndarray):
     return gauss_elimination(A,b)

@benchmark()
def inverse_benchmark(A: np.ndarray):
     return inverse(A)

@benchmark()
def lu_factorization_benchmark(A: np.ndarray):
     return lu_factorization(A)