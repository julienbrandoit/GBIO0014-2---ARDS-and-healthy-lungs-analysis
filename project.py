import scipy.io
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import pandas as pd

"""
    ### UTILS ###
"""

def load_data(file_path):
    data = scipy.io.loadmat(file_path)
    data = {key: data[key][0] for key in ["P", "Q", "V", "t"]}
    return data


def extract_cycles(data):
    cycles = []

    t = data['t']
    P = data['P']
    Q = data['Q']
    V = data['V']

    zero_V_indices = np.argwhere(V == 0).flatten()
    filtered_V_indices = []

    if len(zero_V_indices) == 0:
        print("Not a proper curve ! Never cross V = 0")
        exit(-1)

    filtered_V_indices.append(zero_V_indices[0])

    for i in range(1, len(zero_V_indices)):
        if zero_V_indices[i] != zero_V_indices[i-1] + 1:
            filtered_V_indices.append(zero_V_indices[i])

    for i in range(0, len(filtered_V_indices) - 1):
        cycles.append({'t' : t[filtered_V_indices[i]:filtered_V_indices[i+1]+1] - t[filtered_V_indices[i]],
                       'P' : P[filtered_V_indices[i]:filtered_V_indices[i+1]+1],
                       'Q' : Q[filtered_V_indices[i]:filtered_V_indices[i+1]+1],
                       'V' : V[filtered_V_indices[i]:filtered_V_indices[i+1]+1]})

    return cycles

"""
    ### Q1
"""

"""
    ### Q2
"""

from sklearn.linear_model import LinearRegression

def model(t, Ers, V, Rrs, Q, P0):
    return Ers*V(t) + Rrs * Q(t) + P0

def estimate_parameters(data):
    V = data['V']
    P = data['P']
    Q = data['Q']

    X = np.array([V, Q]).T
    Y = np.array(P)

    model = LinearRegression().fit(X, Y)
    
    Ers = model.coef_[0]
    Rrs = model.coef_[1]
    P0 = model.intercept_

    R_squared = model.score(X, Y)

    return Ers, Rrs, P0, R_squared


def get_lung_properties(data):
    cycles = extract_cycles(data)
    properties = pd.DataFrame(columns=['Ers', 'Rrs', 'P0', 'R_squared'])
    for c, d in enumerate(cycles):
        Ers, Rrs, P0, R_squared = estimate_parameters(d)
        properties.loc[c] = [Ers, Rrs, P0, R_squared]
    return properties

"""
    ### main
"""
# Configuration of the important parameters and variables for the pipeline
data_path = "./data/"

# Loading the data
data_ARDS = load_data(data_path + "Ards.mat")
data_Control = load_data(data_path + "Control.mat")

# For Q2b we extract the compartment properties for each cycle of both patient and we compute Q25, Q50 and Q75

properties_control = get_lung_properties(data_Control)
properties_control_q = properties_control.quantile([0.25, 0.5, 0.75])

properties_ards = get_lung_properties(data_ARDS)
properties_ards_q = properties_ards.quantile([0.25, 0.5, 0.75])

## We print the result
print("# Q2b")
print("CONTROL PATIENT")
print(properties_control_q.head(3))
print(f"R^2 are all the range : [{min(properties_control['R_squared'])}, {max(properties_control['R_squared'])}]")
print("\n")
print("ARDS PATIENT")
print(properties_ards_q.head(3))
print(f"R^2 are all the range : [{min(properties_ards['R_squared'])}, {max(properties_ards['R_squared'])}]")