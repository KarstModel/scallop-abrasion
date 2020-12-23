#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
import SimpleAbrader as sa

# ## assumptions
# 
# 1. sediment concentration uniform in x, one grain begins at each x-index location at height of bedload
# 2. particle velocity influenced by water velocity, gravity, and viscosity
# 

# In[2]:
plt.close('all')
# In[5]:
# building the initial scallop array

xmax = 6  #number of scallops
dx = 0.01
xScal = np.arange(0, xmax+dx,dx)  #x-array for scallop field
uScal = np.arange(0,1+dx,dx)  #x-array for a single scallop


x0, z0 = sa.scallop_array(xScal, uScal, xmax)   #initial scallop profile, dimensions in centimeters
dzdx = np.gradient(z0, x0)
theta2 = np.arctan(dzdx)  #slope angle at each point along scalloped profile


# In[6]:
# definitions and parameters
# =============================================================================
# This is where the grainsize is selected by user
# =============================================================================
D = 2
xi = np.linspace(0, 1, 5)
delta = 0.7 + (0.5 + 3.5 * xi)*D
Hf = delta[1]

if D < 0.0063:
    grain = 'silt'
elif D >= 0.0063 and D < 0.2:
    grain = 'sand'
elif D >= 0.2 and D < 6.4:
    grain = 'pebbles'
elif D >= 6.4:
    grain = 'cobbles'

rho_quartz = 2.65  # g*cm^-3
rho_water = 1
Re = 23300     #Reynold's number from scallop formation experiments (Blumberg and Curl, 1974)
mu_water = 0.01307  # g*cm^-1*s^-1  #because we are in cgs, value of kinematic viscosity of water = dynamic
L = 5    # cm, crest-to-crest scallop length
# In[9]:
u_w0 = (Re * mu_water) / (L * rho_water)   # cm/s, assume constant downstream, x-directed velocity equal to average velocity of water as in Curl (1974)

w_s = sa.settling_velocity(rho_quartz, rho_water, D, Hf) 
# In[10]:

impact_data, loc_data = sa.sediment_saltation(x0, z0, u_w0, w_s, D, 0.05, theta2, mu_water/rho_water)

fig, axs = plt.subplots(nrows = 3, ncols = 1, figsize = (11,26))    
axs[0].plot (x0, z0, 'grey')
axs[0].scatter (impact_data[:, 1], impact_data[:, 6])
axs[0].set_ylabel('KE (ergs)')
axs[0].set_xlabel('x (cm)')
axs[0].set_title('Kinetic energy of impacting ' + str(round(D*10, 3)) + ' mm ' + grain + ' on floor scallops, D/L = ' + str(round(D/5, 2)))
axs[1].plot (x0, z0, 'grey')
axs[1].scatter(impact_data[:, 1], impact_data[:, 2], (impact_data[:, 6])/10**(np.sqrt(D**3)))
axs[1].set_ylabel('z (cm)')
axs[1].set_xlabel('x (cm)')
axs[1].set_title('Kinetic energy of impacting ' + str(round(D*10, 3)) + ' mm '+ grain +' on floor scallops, dot size scales with energy, D/L = ' + str(round(D/5, 2)))
axs[2].plot (x0, z0*100, 'grey')
axs[2].scatter (impact_data[:, 1], impact_data[:, 5])
axs[2].set_title (str(round(D*10, 3)) + ' mm '+ grain +' velocity normal to surface at point of impact (cm/s)')
axs[2].set_xlabel ('x (cm)')
axs[2].set_ylabel ('w_i (cm/s)');

# trajectory figure
fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))    
axs.set_xlim(0, 30)
#axs.set_ylim(-1, 8)
axs.plot (x0, z0, 'grey')
ld = np.array(loc_data, dtype=object)
for p in ld[(np.random.randint(len(loc_data),size=100)).astype(int)]:
    axs.plot(p[:,1], p[:,2], 2, 'blue')
axs.set_ylabel('z (cm)')
axs.set_xlabel('x (cm)')
axs.set_title('Trajectories of randomly selected ' + str(round(D*10, 3)) + ' mm '+ grain +' on floor scallops, D/L = ' + str(round(D/5, 2)))

# velocity exploration
###histogram of last recorded velocities of all particles
fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))    
axs.hist(impact_data[:, 5], 20)
axs.set_xlabel('w_i (cm/s)')
axs.set_title('Histogram of impact velocities of ' + str(round(D*10, 3)) + ' mm ' + grain + ' on 5 cm floor scallops. Fall height = ' + str(round(Hf, 3)) + ' cm.')

 ###velocities with time
fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))    
for p in ld[(np.random.randint(len(loc_data),size=100)).astype(int)]:
     axs.plot(p[:,0], np.sqrt(p[:,4]**2 + p[:,3]**2), 2, 'blue')
axs.set_ylabel('v_s (cm)')
axs.set_xlabel('t (sec)')
axs.set_title('Velocity magnitudes of randomly selected ' + str(round(D*10, 3)) + ' mm '+ grain +' in flow over 5 cm scallops')
 

### vertical velocities with time
fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))    
for p in ld[(np.random.randint(len(loc_data),size=100)).astype(int)]:
    axs.plot(p[:,0], p[:,4], 2, 'blue')
axs.set_ylabel('v_s (cm)')
axs.set_xlabel('t (sec)')
axs.set_title('Vertical velocities of randomly selected ' + str(round(D*10, 3)) + ' mm '+ grain +' in flow over 5 cm scallops')

### erosion rate calculation and comparison with dissolution

fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))    

w = -np.mean(impact_data[:,4])
B = 9.4075*10**-12
A = np.linspace(0.01, 1, 101)
F = np.linspace(0.01, 1, 101)
E = np.empty([101, 101])
cpsTOmpy = 3155760

for k in range(len(A)):
    for l in range(len(F)):
        E[k, l] = cpsTOmpy * A[k] * F[l] * B * w**3  #derived from Lamb et al. (2008)--see markdown below

# #solve for Dreybrodt et al. (2005) agreement

e_min = 0.06
e_max = 0.47  #erosion rate in mm/year
f_min = e_min / ( 3155760*A* B * w**3)
f_max = e_max / (3155760* A* B * w**3)

ImpactRate = 0.41
IR = ImpactRate * np.ones_like(A)

fig = plt.figure(figsize=(11, 7), dpi=100)
plt.contourf(A, F, E, alpha = 0.5, cmap = 'nipy_spectral')
plt.colorbar()
plt.ylim(0, 1)
plt.plot(IR, F)
plt.plot(A, f_max, '--', color = 'grey', label = 'dissolution')
plt.plot(A, f_min, '--', color = 'grey')
plt.fill_between(A, f_min, f_max, alpha = 0.8, color = 'grey')
plt.title('Erosion rate due to mechanical abrasion by 20 mm grains, dissolutional domain shaded in gray')
plt.xlabel('Fraction of grains impacting surface (not advected away)')
plt.ylabel('Fraction of surface exposed (not alluviated)')
fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                    wspace=0.4, hspace=0.1)
ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
ax.set_ylabel('Erosion rate (mm/year)');

