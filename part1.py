import scipy.io
from scipy.interpolate import griddata, interp1d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

data_path = "./data/"

ARDS_data = scipy.io.loadmat(data_path + "Ards.mat")
Control_data = scipy.io.loadmat(data_path + "Control.mat")

ARDS_data = {key: ARDS_data[key][0] for key in ["P", "Q", "V", "t"]}
Control_data = {key: Control_data[key][0] for key in ["P", "Q", "V", "t"]}

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

    V_sequence = [zero_V_indices[0]]

    for i in range(1, len(zero_V_indices)):
        
        if zero_V_indices[i] != zero_V_indices[i-1] + 1:
            filtered_V_indices.append(V_sequence[len(V_sequence)//2])
            V_sequence = []

        V_sequence.append(zero_V_indices[i])
    
    filtered_V_indices.append(V_sequence[len(V_sequence)//2])
        
    for i in range(0, len(filtered_V_indices) - 1):
        cycles.append({'t' : t[filtered_V_indices[i]:filtered_V_indices[i+1]+1] - t[filtered_V_indices[i]],
                       'P' : P[filtered_V_indices[i]:filtered_V_indices[i+1]+1],
                       'Q' : Q[filtered_V_indices[i]:filtered_V_indices[i+1]+1],
                       'V' : V[filtered_V_indices[i]:filtered_V_indices[i+1]+1]})

    return cycles 

def get_works(data, title="Data",alpha_value = 0.5, x_label = "X", y_label ="Y", grid = True):
    P = data['P']
    V = data['V']
    Q = data['Q']
    t = data['t'] - data['t'][0]
    #Create the P-V loop
    plt.figure(1)
    print("---PV loop---")
    # Create a colormap that represents the gradient of time
    cmap = plt.cm.jet  # You can choose any colormap you prefer
    norm = plt.Normalize(min(t), max(t))
    ncolors = len(t)  # Number of colors should match the number of time points
    colors = cmap(norm(t))
    points = np.array([P, V]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a LineCollection object with a color gradient
    lc = LineCollection(segments, cmap=ListedColormap(colors), norm=BoundaryNorm(t, ncolors), linewidth=2, alpha=alpha_value)
    lc.set_array(t)

    # Plot the trajectory
    fig, ax = plt.subplots()
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)
    plt.colorbar(lc, label='Time')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    minP = np.argmin(P)
    minV = 0
    index_inspi = np.where(Q >= 0)
    index_inspi = index_inspi[0]
    V_Q_0 = V[index_inspi[-1]]
    P_Q_0 = P[index_inspi[-1]]

    P_inspi = P[index_inspi]
    V_inspi = V[index_inspi]

    a,b = np.polyfit([P[minP], P_Q_0],  [0, V_Q_0], deg = 1)
    px = V_inspi/a - b
    vy = P_inspi*a +b

    equation_label = f"y = {a:.3f}x + {b:.3f}"
    ax.plot([P[minP], P_Q_0],  [0, V_Q_0], label = equation_label, color ="deepskyblue")
    ax.legend(loc = 'lower right')

    RIW = np.trapz(P_inspi-px, V_inspi)
    REW = np.trapz(P,V) - RIW
    plt.plot(P_inspi, V_inspi,color = "mediumpurple")
    plt.fill_between(P,V, color = "lavender")
    #plt.fill_between(P_inspi, V_inspi, color = "mediumpurple", label = 'RIW')
    plt.legend(loc = 'lower right')
    plt.show()
    plt.figure(2)
    print("---VQ loop---")
    # Create a colormap that represents the gradient of time
    cmap = plt.cm.jet  # You can choose any colormap you prefer
    norm = plt.Normalize(min(t), max(t))
    ncolors = len(t)  # Number of colors should match the number of time points
    colors = cmap(norm(t))
    points = np.array([V, Q]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a LineCollection object with a color gradient
    lc = LineCollection(segments, cmap=ListedColormap(colors), norm=BoundaryNorm(t, ncolors), linewidth=2, alpha=alpha_value)
    lc.set_array(t)

    # Plot the trajectory
    fig, ax = plt.subplots()
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)
    plt.colorbar(lc, label='Time')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    ax.plot([0,V_Q_0],  [0, 0], color ="deepskyblue")
    plt.show()
    

    return RIW, REW

cycles = extract_cycles(Control_data)

RIW, REW = get_works(cycles[3])
print(RIW, REW)