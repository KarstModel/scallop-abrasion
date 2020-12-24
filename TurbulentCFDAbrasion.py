# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 14:10:15 2020

@author: rachelbosch
"""


# In[1]:


import numpy as np
from matplotlib import pyplot as plt
import darthabrader as da
from dragcoeff import dragcoeff
from numpy import genfromtxt

# ## assumptions
# 
# 1. sediment concentration uniform in x, one grain begins at each x-index location at height of bedload
# 2. particle velocity influenced by water velocity, gravity, and viscosity
# 

# In[2]:

plt.close('all')

TurbVel = genfromtxt('TurbulentFlowfield.csv', delimiter=',')

# variable declarations
nx = 101
ny = 101
nt = 10
nit = 50 
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
new_x = np.linspace(0, 5, nx)
new_z = np.linspace(0, 5, ny)
new_X, new_Z = np.meshgrid(new_x, new_z)
new_u = np.zeros((ny, nx))
new_w = np.zeros((ny, nx))

#restructure STAR-CCM+ turbulent flow data set
for i in range(len(TurbVel)):
    x_index = np.int(TurbVel[i, 0])
    z_index = np.int(TurbVel[i, 1])
    new_u[z_index, x_index] = TurbVel[i, 2]
    new_w[z_index, x_index] = TurbVel[i, 3]
    
#identify holes in data set and patch them    
holes_and_wall_u = np.where(new_u == 0)
ux_zero = np.array(holes_and_wall_u[1])
uz_zero = np.array(holes_and_wall_u[0])

for j in range(len(ux_zero)):
    if (ux_zero[j] > 0 & ux_zero[j] <= 40):
        if (ux_zero[j] - ux_zero[j-1]) > 1:   #this is a hole in the lee-side data set
            hole_z = np.int(uz_zero[j])
            hole_x = np.int(ux_zero[j])
            new_u[hole_z, hole_x] = ((new_u[hole_z + 1, hole_x] + new_u[hole_z + 1, hole_x + 1] + new_u[hole_z, hole_x + 1] + new_u[hole_z - 1, hole_x + 1] + new_u[hole_z - 1, hole_x] + new_u[hole_z - 1, hole_x - 1] + new_u[hole_z, hole_x - 1] + new_u[hole_z + 1, hole_x - 1])/8)
            new_w[hole_z, hole_x] = ((new_w[hole_z + 1, hole_x] + new_w[hole_z + 1, hole_x + 1] + new_w[hole_z, hole_x + 1] + new_w[hole_z - 1, hole_x + 1] + new_w[hole_z - 1, hole_x] + new_w[hole_z - 1, hole_x - 1] + new_w[hole_z, hole_x - 1] + new_w[hole_z + 1, hole_x - 1])/8) 
        else:
            continue        #this is a wall boundary, no action needed
    elif (ux_zero[j] > 40 & ux_zero[j] < 100):
        if (ux_zero[j + 1] - ux_zero[j]) > 1:   #this is a hole in the stoss-side data set
            hole_z = np.int(uz_zero[j])
            hole_x = np.int(ux_zero[j])
            new_u[hole_z, hole_x] = ((new_u[hole_z + 1, hole_x] + new_u[hole_z + 1, hole_x + 1] + new_u[hole_z, hole_x + 1] + new_u[hole_z - 1, hole_x + 1] + new_u[hole_z - 1, hole_x] + new_u[hole_z - 1, hole_x - 1] + new_u[hole_z, hole_x - 1] + new_u[hole_z + 1, hole_x - 1])/8)
            new_w[hole_z, hole_x] = ((new_w[hole_z + 1, hole_x] + new_w[hole_z + 1, hole_x + 1] + new_w[hole_z, hole_x + 1] + new_w[hole_z - 1, hole_x + 1] + new_w[hole_z - 1, hole_x] + new_w[hole_z - 1, hole_x - 1] + new_w[hole_z, hole_x - 1] + new_w[hole_z + 1, hole_x - 1])/8) 
        else:
            continue        #this is a wall boundary, no action needed
    else:
        continue      # this is also inside the wall   
    

fig = plt.figure(figsize=(11, 7), dpi=100)
plt.contourf(new_X, new_Z, np.sqrt(new_u**2 + new_w**2), alpha = 0.5)
plt.colorbar()
plt.title('Velocity magnitude, turbulent flow')
plt.xlabel('X')
plt.ylabel('Z');

# In[4]:


# six-scallop long velocity matrix

u_water = np.empty(shape=(101,601))
w_water = np.empty(shape=(101,601))

a = 6
b = len(new_u) - 1
c = len(new_u) - 1

for i in range(a):
    for j in range(b):   
        u_water[:, i*b + j] = new_u[:, j]
        w_water[:, i*b + j] = new_w[:, j]
        
# In[5]:


# building the initial scallop array

xmax = 6  #number of scallops
dx = 0.01
xScal = np.arange(0, xmax+dx,dx)  #x-array for scallop field
uScal = np.arange(0,1+dx,dx)  #x-array for a single scallop


x0, z0 = da.scallop_array(xScal, uScal, xmax)   #initial scallop profile, dimensions in centimeters
dzdx = np.gradient(z0, x0)
theta2 = np.arctan(dzdx)  #slope angle at each point along scalloped profile


# In[6]:


# definitions and parameters
# =============================================================================
# This is where the grainsize is selected by user
# =============================================================================
grain_diam_max = 2.5  # cm
diam = grain_diam_max * np.logspace(-3, 0, 9)
EnergyAtImpact = np.empty(shape = (len(diam), len(x0)))
XAtImpact = np.empty(shape = (len(diam), len(x0)))
ZAtImpact = np.empty(shape = (len(diam), len(x0)))
ErosionAtImpact = np.empty(shape = (len(diam), len(x0)))

i = 0
for D in diam:
    xi = np.linspace(0, 1, 5)
    delta = 0.7 + (0.5 + 3.5 * xi)*D
    Hf = delta[1]
    
    if D < 0.0063:
        grain = 'silt'
    elif D >= 0.0063 and D < 0.2:
        grain = 'sand'
    elif D >= 0.2:
        grain = 'gravel'

    
    rho_quartz = 2.65  # g*cm^-3
    rho_ls = 2.55
    rho_water = 1
    Re = 23300     #Reynold's number from scallop formation experiments (Blumberg and Curl, 1974)
    mu_water = 0.01307  # g*cm^-1*s^-1  #because we are in cgs, value of kinematic viscosity of water = dynamic
    L = 5    # cm, crest-to-crest scallop length
    
    
    
    
    # In[9]:
    
    
    u_w0 = (Re * mu_water) / (L * rho_water)   # cm/s, assume constant downstream, x-directed velocity equal to average velocity of water as in Curl (1974)
    w_w0 = 1
    
    Re_p = da.particle_reynolds_number(D, w_w0, mu_water/rho_water)
    drag_coef = dragcoeff(Re_p)
    print('drag_coef', drag_coef)
    
    w_s = da.settling_velocity(rho_quartz, rho_water, drag_coef, D, Hf) # DW 12/8: I think this calculation might assume things no longer true about the settling code
                                # RB 12/9: @DW, w_s is only used to calculate the trajectories in the upper fall where flow is assumed to be uniform
    
    # In[10]:
    
    impact_data, loc_data = da.sediment_saltation(x0, z0, w_water, u_water, u_w0, w_s, D, 0.05, theta2, mu_water/rho_water)
    
    ImpactEnergyAvg = np.empty_like(diam)
    TotalImpactEnergy = np.empty_like(diam)
    ImpactEnergyTotalAvg = np.average(impact_data[:, 6])
    NumberImpacts = np.count_nonzero(impact_data[:, 6])
    ImpactEnergyAvg[i] = ImpactEnergyTotalAvg/NumberImpacts 
    TotalImpactEnergy[i] = np.sum(impact_data[300:401, 6])
      
    EnergyAtImpact[i, :] = impact_data[:, 6]
    XAtImpact[i, :] = impact_data[:, 1]
    ZAtImpact[i, :] = impact_data[:, 2]
    
    B = 9.4075*10**-12  # s**2·cm**-2
    ErosionAtImpact[i, :] = B * (impact_data[:, 5])**3    ##Lamb et al., 2008
    
    i += 1
    
    # trajectory figure
    fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))    
    axs.set_xlim(15, 25)
    axs.set_aspect('equal')
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
    
### average energy plot
fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
axs.semilogy((diam * 10), ImpactEnergyAvg)
axs.set_xlabel('Grain diameter (mm)')
axs.set_ylabel('Average partticle kinetic energy per impact (ergs)') 

### total energy plot 
fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
axs.semilogy((diam * 10), TotalImpactEnergy*10**-7)
axs.set_xlabel('Grain diameter (mm)')
axs.set_ylabel('Total impact energy over length of one scallop (Joules)') 

# impact location & energy-at-location plot



from matplotlib import colors
from matplotlib import cm

GetMaxEnergies = EnergyAtImpact[-1, :][EnergyAtImpact[-1, :] != 0]
ColorScheme = np.log10(GetMaxEnergies)  ## define color scheme to be consistent for every plot
ColorNumbers = ColorScheme[np.logical_not(np.isnan(ColorScheme))] 
ColorMax = np.ceil(np.max(ColorNumbers))

my_colors = cm.get_cmap('cool', 256)
fig, axs = plt.subplots(nrows = len(diam), ncols = 1, figsize = (11,26))

for j in range(len(diam)):
    
    if diam[j] < 0.0063:
        grain = 'silt'
    elif diam[j] >= 0.0063 and D < 0.2:
        grain = 'sand'
    elif diam[j] >= 0.2:
        grain = 'gravel'
    axs[j].set_xlim(15, 25)
    axs[j].set_ylim(-0.5, 1.5)
    axs[j].set_aspect('equal')
    axs[j].plot(x0, z0, 'grey')
    EnergyAtImpact[j, :][EnergyAtImpact[j, :]==0] = np.nan
    findColors = (np.log10(EnergyAtImpact[j, :]))/ColorMax 
    impact_dots = axs[j].scatter(XAtImpact[j, :], ZAtImpact[j, :], c = my_colors(findColors) )
    
    

    axs[j].set_ylabel('z (cm)')
    
    axs[j].set_title('Locations of impact, ' + str(round(diam[j]*10, 3)) + ' mm '+ grain +' on floor scallops, color indicates particle kinetic energy')
#legend
fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                    wspace=0.4, hspace=0.1)
cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
norm = colors.Normalize(vmin = 0, vmax = ColorMax)
cbar = plt.colorbar(cm.ScalarMappable(norm = norm, cmap='cool'), cax = cb_ax)
cb_ax.set_ylabel('log10 of Kinetic energy of impact (ergs)')
axs[-1].set_xlabel('x (cm)')
plt.show()

# =============================================================================
# Two different ways to get erosion rate due to abrasion:
    # 1. Work-energy theorem: work done on limestone = change in kinetic energy of sandstone grain
    # 2. Erosion rate due to abrasion expression from Lamb et al. (2008). 
# =============================================================================




# ### 1. WORK-ENERGY THEOREM (these results are very much the wrong order of magnitude)
# CoR = 0.4    #coefficient of restitution for sandstone impinging on limestone
# WorkDoneAtImpact = EnergyAtImpact * (1 - CoR)   #work done on limestone, work-energy theorem

# for j in range(len(diam)):
#     fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
    
#     mlr = (np.pi * rho_ls * diam[j]**3)/6    # mass limestone removed
#     dv_dt = WorkDoneAtImpact/(mlr * diam[j])
#     E_we = dv_dt * rho_ls * 24 * 3600  # convert cm*s**-1 to g*cm**-2*day*-1
    
#     if diam[j] < 0.0063:
#         grain = 'silt'
#     elif diam[j] >= 0.0063 and D < 0.2:
#         grain = 'sand'
#     elif diam[j] >= 0.2:
#         grain = 'gravel'
#     axs.set_xlim(15, 25)

#     #axs.set_aspect('equal')
#     axs.plot(x0, (E_we[j]))
   
    
#     axs.set_xlim(15, 25)
#     axs.set_ylabel('Erosion rate (g/(cm^2*day))')
#     axs.set_xlabel('x (cm)')
    
#     axs.set_title('Erosion rate by work-energy theorem, ' + str(round(diam[j]*10, 3)) + ' mm '+ grain +' on floor scallops')

# plt.show()

### 2. BUFFALO WILD WINGS AND WECK

for j in range(len(diam)):
    fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
    
    E_bw3 = -(ErosionAtImpact[j, :] * rho_ls * 24 * 3600 * 1000)  # convert Lamb et al. (2008) units to Hammer et al. (2011) units
    for k in range(len(E_bw3)):                                  # cm*s**-1 to mg*cm**-2*day**-1
        if E_bw3[k] < 0:
            E_bw3[k] = 0
    
    if diam[j] < 0.0063:
        grain = 'silt'
    elif diam[j] >= 0.0063 and D < 0.2:
        grain = 'sand'
    elif diam[j] >= 0.2:
        grain = 'gravel'
    axs.set_xlim(15, 25)

    #axs.set_aspect('equal')
    axs.scatter(x0, E_bw3)
   
    
    axs.set_xlim(15, 25)
    axs.set_ylabel('Erosion rate (mg/(cm^2*day))')
    axs.set_xlabel('x (cm)')
    
    axs.set_title('Erosion rate ala BW3, ' + str(round(diam[j]*10, 3)) + ' mm '+ grain +' on floor scallops')

plt.show()


### Plot erosion rate (BW3 approach) v. scallop surface slope    
for j in range(len(diam)):
    fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
    
    E_bw3 = -(ErosionAtImpact[j, :] * rho_ls * 24 * 3600 * 1000)  # convert Lamb et al. (2008) units to Hammer et al. (2011) units
    for k in range(len(E_bw3)):                                  # cm*s**-1 to mg*cm**-2*day**-1
        if E_bw3[k] < 0:
            E_bw3[k] = 0
    
    if diam[j] < 0.0063:
        grain = 'silt'
    elif diam[j] >= 0.0063 and D < 0.2:
        grain = 'sand'
    elif diam[j] >= 0.2:
        grain = 'gravel'

    axs.scatter(theta2, E_bw3)
   

    axs.set_ylabel('Erosion rate (mg/(cm^2*day))')
    axs.set_xlabel('dz/dx')
    
    axs.set_title('Erosion rate ala BW3, ' + str(round(diam[j]*10, 3)) + ' mm '+ grain +' v. floor scallop local slope')

plt.show()