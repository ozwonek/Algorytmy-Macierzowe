import time
import tracemalloc
from Float import Float 
import numpy as np
from functools import wraps
import pandas as pd

def benchmark():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
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
            div_ops = Float.div_counter
            abs_error = None
            


            df = pd.DataFrame([{
                "time": exec_time,
                "memory": mem_usage,
                "error": abs_error,
                "add": add_ops,
                "sub": sub_ops,
                "mul": mul_ops,
                "div": div_ops,
                "flops": (add_ops + sub_ops + mul_ops +div_ops)/exec_time
            }])

            return result, df
        return wrapper
    return decorator