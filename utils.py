import numpy as np

def calculate_array_stats(arr):
    return {
        "Size": len(arr),
        "Min Value": min(arr),
        "Max Value": max(arr),
        "Range": max(arr) - min(arr),
        "Mean": round(np.mean(arr), 2),
        "Std Dev": round(np.std(arr), 2),
    }
