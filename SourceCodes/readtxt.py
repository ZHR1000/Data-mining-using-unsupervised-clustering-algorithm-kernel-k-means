import numpy as np
import pandas as pd
results = np.loadtxt("test1_data.txt", delimiter=" ")
print("Variance",np.var(results))
print("Standard Deviation",np.std(results))
