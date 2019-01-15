import numpy as np
import matplotlib.pyplot as pyplot
import random
import time

start = time.time()
#set constants
#N = 200
N = 60
#T = 30
a = 0.5
#a = 1.4
m = 1 #mass is equal to 1
omega = 1
#h-bar is taken as 1 throughout
paths = 1000 #number of paths to run
keep = 4 #frequency at which paths are stored
#keep = 23
discard = 40 #number of intial paths to discard
#discard = 50
offset = discard/keep +1
n_keep = int((paths-discard)/keep)
#create arrays
x_array = np.zeros(N) #array where paths are perturbed
path_array = np.zeros((n_keep,N)) #array where paths are stored
    
def potential(x):
    """Calculates the potential for a given value of x"""
    v = (m*omega**2)*0.5*(x**2)
    return v
    
def potential_differential(x):
    #differentiated with respect to x
    dv = (m*omega**2)*x 
    return dv

def energy(x1, x2, a):
    energy = 0.5*(m*(x2-x1)/a)**2 + potential((x2+x1)/2)
    return energy
    
def create_paths(x_array):
    """Uses the metropolis algorithm to make a Monte Carlo selection of paths 
    taking into account their weight"""
    eps = 1.4
    #eps = 1.2
    for j in range(paths):
        x_array[-1] = x_array[0]
        if j>discard and j%keep == 0:
            path_array[int((j/keep)-offset):] = x_array
        for i in range(0, (N-1), 1): 
            x_new = np.copy(x_array)
            random_perturb = random.uniform(-eps, eps)
            random_number = random.uniform(0,1)
            x_new[i] = x_array[i] + random_perturb 
            delta_energy = energy(x_array[i-1], x_new[i], a) + energy(x_new[i], x_array[i+1], a) - energy(x_array[i-1], x_array[i], a) - energy(x_array[i], x_array[i+1], a)
            if delta_energy < 0:
                x_array[i] = x_new[i]
            elif random_number < np.exp(-a*delta_energy):
                x_array[i] = x_new[i]
            else:
                x_array[i] = x_array[i]
    all_paths = path_array
    values = path_array.flatten()
    mean = np.mean(values)
    var = np.var(values)
    sigma = np.sqrt(var)
    #print (mean, sigma, len(values))
    #return values, mean, sigma, all_paths
    return all_paths
    
def quantum_method(x):
    """Computes the expected groundstate wave function probability density
    using the conventional QM method."""
    denominator = np.pi**0.25
    numerator = np.exp((-x**2)/2)
    y = (numerator/denominator)**2
    return y
    
def gaussian(x, mean, sigma):
    """Gaussian function given mean and standard deviation"""
    y = (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(x-mean)**2/(2*sigma**2))
    return y
    
def H_expectation(x):
    H_expec = potential(x) + 0.5*x*potential_differential(x)
    return H_expec
    
def groundstate_energy(all_paths):
    total = np.zeros(n_keep)
    add = 0
    for i,row in enumerate(all_paths):
        for item in row:
    # for i in range(len(all_paths)):
    #     for j in range(len(all_paths[i])):
            E = H_expectation(item)
            add += E
        add = add/N
        total[i] = add      
    average = sum(total)/(n_keep)
    theoretical = omega*0.5
    return total, average
    #print('calculated groundstate energy =', average)
    #print('theoretical groundstate energy =', theoretical) 

def bootstrap(all_paths):
    """creates a bootstrap of the paths in order to calculate the groundstate
    energy from a different sample"""
    bootstrap = [] # new ensemble
    for i in range(0,n_keep):
        alpha = int(random.uniform(0,n_keep)) # choose random config
        bootstrap.append(all_paths[alpha]) # keep all_paths[alpha]
    return bootstrap

def uncertainty_calc(path_array):
    """calculates the uncertainty using the bootstrap method"""
    energies = []
    for i in range(50):
       sample = bootstrap(path_array)
       energies.append(sample)
    mean = np.mean(energies)
    uncertainty = np.std(energies)
    perc_error = ((0.5-mean)/0.5)*100
    return mean, uncertainty, perc_error
    
paths = create_paths(x_array)
energy_array = groundstate_energy(paths)[0]
print(uncertainty_calc(energy_array))    


end = time.time()
print("--- %s seconds ---" % (end - start))
