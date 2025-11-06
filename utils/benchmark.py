
import time
import tracemalloc
from Float import Float 
import numpy as np
from functools import wraps
import pandas as pd

def benchmark(reference_func = None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            A, B = args[:2]
            Float.reset()
            tracemalloc.start()
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            exec_time = end_time - start_time
            mem_usage = peak / (1024 * 1024) 
            add_ops = Float.add_counter
            sub_ops = Float.sub_counter
            mul_ops = Float.mul_counter
            abs_error = None
            if reference_func is not None:
                ref_result = reference_func(A, B)
                abs_error = np.max(np.abs(result - ref_result)) 


            df = pd.DataFrame([{
                "time": exec_time,
                "memory": mem_usage,
                "error": abs_error,
                "add": add_ops,
                "sub": sub_ops,
                "mul": mul_ops,
            }])

            return result, df
        return wrapper
    return decorator