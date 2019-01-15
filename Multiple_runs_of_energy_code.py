import numpy as np
from Feynman_research_prob_density_graph import create_paths
import matplotlib.pyplot as pyplot
import time

start = time.time()
#set constants
N = 200
#T = 75
a = 1.4
m = 1 #mass is equal to 1
omega = 1
#h-bar is taken as 1 throughout
paths = 1000 #number of paths to run
keep = 23 #frequency at which paths are stored
discard = 50 #number of intial paths to discard
offset = discard/keep +1
n_keep = int((paths-discard)/keep)
#create arrays
x_array = np.zeros(N) #array where paths are perturbed
path_array = np.zeros((n_keep,N)) #array where paths are stored

def run_energy_code(x_array):
    """Runs energy code from prob density code in order to 
    average the values this produces over 10 runs"""
    runs = 10
    accept_rate = []
    x = np.arange(1,runs+1,1)
    for i in range(0,runs):
        cycle = create_paths(x_array)
        accept_rate.append(cycle[4])
    pyplot.plot(x, accept_rate)
    pyplot.xlabel('Run number')
    pyplot.ylabel('Acceptance rate')
    pyplot.legend()
    pyplot.show()
    
run_energy_code(x_array)   
end = time.time()
print("--- %s seconds ---" % (end - start))