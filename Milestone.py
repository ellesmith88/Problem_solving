import numpy as np
import matplotlib.pyplot as pyplot
import random
import time


start = time.time()
#set constants
N = 14
T = 7
a = T/N
m = 1 #mass is equal to 1
omega = 1
#h-bar is taken as 1 throughout
paths = 30000 #number of paths to run
keep = 4 #frequency at which paths are stored
discard = 40 #number of intial paths to discard
offset = discard/keep +1
n_keep = int((paths-discard)/keep)
#create arrays
x_array = np.zeros(N) #array where paths are perturbed
path_array = np.zeros((n_keep,N)) #array where paths are stored
    
def potential(x):
    """Calculates the potential for a given value of x"""
    v = (m*omega**2)*0.5*(x**2)
    return v

def energy(x1, x2, a):
    energy = 0.5*(m*(x2-x1)/a)**2 + potential((x2+x1)/2)
    return energy
    
def create_paths(x_array):
    """Uses the metropolis algorithm to make a Monte Carlo selection of paths 
    taking into account their weight"""
    eps = 1.4
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
    values = path_array.flatten()
    mean = np.mean(values)
    var = np.var(values)
    sigma = np.sqrt(var)
    print (mean, sigma, len(values))
    return values, mean, sigma
    
    
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
    
def plot_and_values(x_array):
    """Takes the paths created and plots them in a histogram to produce the 
    probability density and compares to QM method. Finds the values of prob
    density for each method at a given x"""
    #plot histogram: path integral method
    x_find = 1 #x at which to find find PD values
    data = create_paths(x_array)
    n, bins, patches = pyplot.hist(data[0], 300, normed=True)#, align='right')
    x_range = bins[:-1].max() - bins[:-1].min()
    middle_bin_values = bins[:-1] + x_range/(2*len(n))
    pyplot.plot(middle_bin_values,n, label = 'Path Integral Method')
    #plot expected from quantum method
    x = np.arange(-2.2,2.2,0.1)
    y_qm = np.zeros(len(x))
    y_gaussian = np.zeros(len(x))
    for i in range(len(x)):
        y_qm[i] = quantum_method(x[i])
        y_gaussian[i] = gaussian(x[i], data[1], data[2])
    pyplot.plot(x,y_qm, label = 'Quantum Method')
    #plot histogram fit
    pyplot.plot(x, y_gaussian, label = 'Histogram fit')
    #pyplot.plot(x, norm.pdf(x,data[1],data[2]), label = 'Histogram Fit stats')
    pyplot.xlim([-2,2])
    pyplot.xlabel('x')
    pyplot.ylabel('Probability density')
    pyplot.legend()
    pyplot.show()
    #find values
    QM_value = quantum_method(x_find)
    fit_value = gaussian(x_find, data[1], data[2])
    error = (QM_value - fit_value)/QM_value
    print ('QM_value =', QM_value,', fit_value =', fit_value,', Percentage error =', error*100,'%')

plot_and_values(x_array)



end = time.time()
print("--- %s seconds ---" % (end - start))