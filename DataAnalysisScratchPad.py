# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 12:36:05 2021

@author: rache
"""
import numpy as np
from matplotlib import pyplot as plt
import darthabrader as da
from numpy import genfromtxt
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes 
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib import colors
from matplotlib import cm
from matplotlib.colors import ListedColormap

####fitting velocity data to Dietrich settling curve
rho_sediment = 2.65
rho_fluid = 1
diameter_array1 = genfromtxt('./outputs/diam5turbulent2021-02-11.csv', delimiter=',')
diameter_array2 = genfromtxt('./outputs2/diam5turbulent2021-02-13.csv', delimiter=',')
diameter_array3 = genfromtxt('./outputs2/diam5turbulent2021-02-14.csv', delimiter=',')
scallop_length = 5
VelocityAvg1 = genfromtxt('./outputs/VelocityAvg5turbulent2021-02-11.csv', delimiter=',')
VelocityAvg2 = genfromtxt('./outputs2/VelocityAvg5turbulent2021-02-13.csv', delimiter=',')
VelocityAvg3 = genfromtxt('./outputs2/VelocityAvg5turbulent2021-02-14.csv', delimiter=',')

diameter_array = np.append(diameter_array2, diameter_array3, axis=0)
VelocityAvg = np.append(VelocityAvg2, VelocityAvg3, axis = 0)
    
fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
g = 981 # cm*s^-2
nu = 0.01307  # g*cm^-1*s^-1
not_nan_idx = np.where(~np.isnan(VelocityAvg1))
diameter_array=diameter_array1[not_nan_idx]
VelocityAvg=VelocityAvg1[not_nan_idx]
D_star = ((rho_sediment-rho_fluid)*g*(diameter_array)**3)/(rho_fluid*nu)
W_star = (rho_fluid*VelocityAvg**3)/((rho_sediment-rho_fluid)*g*nu)
W_star_Dietrich = (1.71 * 10**-4 * D_star**2)
axs.scatter(diameter_array*10, W_star, label = 'simulated impact velocity')
axs.plot(diameter_array*10, W_star_Dietrich, c = 'g', label = 'settling velocity (Dietrich, 1982)')

def settling_velocity(D_star, r, s):
    return r * D_star**s

pars, cov = curve_fit(f=settling_velocity, xdata=D_star, ydata=W_star, p0=[1.71 * 10**-4, 2], bounds=(-np.inf, np.inf))
# Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
stdevs = np.sqrt(np.diag(cov))
# Calculate the residuals
res = W_star - settling_velocity(D_star, *pars)
axs.plot(diameter_array*10, settling_velocity(D_star, *pars), linestyle='--', linewidth=2, color='black', label = 'fitted to Equation (13), with r= '+str(round(pars[0], 2))+' and s= '+str(round(pars[1], 2)))

plt.legend()
plt.semilogy()
axs.set_xlabel('diameter (mm)')
axs.set_ylabel('dimensionless settling velocity, Dstar') 
axs.set_title('Particle velocities over '+str(scallop_length)+' cm scallops, fit to Settling Velocity of Natural Particles (Dietrich, 1982)')

plt.show()






#####abrasion v dissolution 
cb_max = 0.02
cb_tiny = 4 * 10**-5
cb_old = 0.01
cb = np.linspace(cb_tiny, cb_max, 9)
fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
NEA5_old1 = genfromtxt('./outputs2/NormErosionAvg5turbulent2021-02-13.csv', delimiter=',')
NEA5_old2 = genfromtxt('./outputs2/NormErosionAvg5turbulent2021-02-14.csv', delimiter=',')
NEA5_old = np.append(NEA5_old1, NEA5_old2, axis=0)

diameter_array = np.append(diameter_array2, diameter_array3, axis=0)
for i in range(len(cb)):
    NEA5_new = NEA5_old * cb[i]/cb_old
    cb_array = np.ones(len(NEA5_new)) * cb[i]
    findColors = np.ones_like(NEA5_new)
    col = findColors.tolist()
    for j in range(len(NEA5_new)):
        if NEA5_new[j] < 5.66:
            col[j] = 'b'
        elif NEA5_new[j] > 5.66 and NEA5_new[j] <=12.175:
            col[j] = 'c'
        elif NEA5_new[j] > 12.175:
            col[j] = 'g'
    axs.scatter((diameter_array*10), (cb_array), c = col)

axs.set_title('Abrasion Rate Normalized by Number of Impacts on 5 cm Scallops')
axs.set_xlabel('particle grainsize (mm)')
axs.set_ylabel('bedload sediment concentration')
plt.show()


####number of impacts
l32 = 5
numScal = 8
dx0 = 0.05/l32
xScal = np.arange(0, numScal+dx0,dx0)  #x-array for scallop field
uScal = np.arange(0,1+dx0,dx0)  #x-array for a single scallop

x0, z0 = da.scallop_array(xScal, uScal, numScal, l32)   #initial scallop profile, dimensions in centimeters
fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
Diam5 = genfromtxt('./outputs/diam5turbulent2021-02-11.csv', delimiter=',')
EAI5 = genfromtxt('./outputs/ErosionAtImpact5turbulent2021-02-11.csv', delimiter=',')
ErosionSum = np.zeros_like(Diam5)
NormErosionAvg = np.zeros_like(Diam5)
NumberOfImpactsByGS = np.zeros_like(Diam5)
for r in range(len(Diam5)):
    ErosionSum[r] = -np.sum(EAI5[r, int(len(x0)/2):int(len(x0)/2+len(uScal))][EAI5[r, int(len(x0)/2):int(len(x0)/2+len(uScal))]<0]*1000*36*24*365.25)
    NumberPositives = len(EAI5[r, int(len(x0)/2):int(len(x0)/2+len(uScal))][EAI5[r, int(len(x0)/2):int(len(x0)/2+len(uScal))]<0])
    NumberOfImpactsByGS[r] = NumberPositives
    if NumberPositives > 0:
        NormErosionAvg[r] = ErosionSum[r]/NumberPositives
    else:
        NormErosionAvg[r] = 0
axs.scatter(Diam5*10, NormErosionAvg, label = 'impacts on 5 cm scallop', zorder = 0)
plt.title('Number of particle impacts on one scallop')
axs.set_xlabel('particle grainsize (mm)')
axs.set_ylabel('number of impacts')

plt.legend(loc='upper left')
axs.grid(True, which = 'both', axis = 'x')
plt.show()


fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
Diam5 = genfromtxt('./outputs/diam5turbulent2021-02-11.csv', delimiter=',')
Diam2_5 = genfromtxt('./outputs/diam2.5turbulent2021-02-15.csv', delimiter=',')
IEA5 = genfromtxt('./outputs/ImpactEnergyAvg5turbulent2021-02-11.csv', delimiter=',')
IEA2_5 = genfromtxt('./outputs/ImpactEnergyAvg2.5turbulent2021-02-15.csv', delimiter=',')

axs.scatter(Diam5*10, IEA5, label = 'KE on 5 cm scallop')
axs.scatter(Diam2_5*10, IEA2_5, label = 'KE on 2.5 cm scallop')

plt.title('Average kinetic energy at impact')
axs.set_xlabel('particle grainsize (mm)')
axs.set_ylabel('kinetic eregy (ergs)')

plt.legend(loc='upper left')
plt.show()
