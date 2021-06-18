# -*- coding: utf-8 -*-
"""
DKARST.py --- Does Karst Abrasion Result in Scalloped Tunnels?

Created on Fri Jan 15 06:20:38 2021

@author: rachelbosch

User inputs crest-to-crest scallop length, number of grainsizes to iterate over, and number of scallops in bedrock profile. 
Program outputs number and location of particle impacts by sediment on bedrock profile, drag coefficent and partile Reynold's number
    of each particle at impact, velocity and kinetic energy of each particle at impact, and resulting erosion rate
    of bedrock surface due to those impacts.

This code accompanies 
    Bosch, Rachel, and Dylan Ward, 2021, “Numerical modeling comparison of mechanical and chemical erosion of sculpted limestone surfaces,” 
    in preparation to submit to JGR: Earth Surface.
    
Other files required for execution include darthabrader.py, scallopplotlib.py, dragcoeff.py, TurbulentFlowfield1.csv, TurbulentFlowfield2.5.csv, and TurbulentFlowfield5.csv, TurbulentFlowfield10.csv.
"""


# In[1]:


import numpy as np
from matplotlib import pyplot as plt
import darthabrader as da
import scallopplotlib as spl
from os.path import join

# =============================================================================


# ## assumptions
# 
# 1. sediment concentration uniform in x, one grain begins at each x-index location at height of bedload
# 2. particle velocity influenced by water velocity, gravity, and viscosity
# 

# In[2]:

plt.close('all')


# ### user input: 
# =============================================================================
scallop_lengths = [1, 2.5, 5, 10]
for j in range(len(scallop_lengths)):
    l32 = scallop_lengths[j]
    outfolder='./outputs'  # make sure this exists first
    n = 100  #number of grainsizes to simulate in diameter array
    numScal = int(5000/l32) #number of scallops
    numPrtkl = 200 # number of particles to release for each grainsize, for now, must use fewer than (l32 * numScal / 0.05)
    flow_regime = 'turbulent'    ### choose 'laminar' or 'turbulent'
    if flow_regime == 'laminar':
        l32 = 5
    
    rho_water = 1
    Re = 23300     #Reynold's number from scallop formation experiments (Blumberg and Curl, 1974)
    mu_water = 0.01307  # g*cm^-1*s^-1  #because we are in cgs, value of kinematic viscosity of water = dynamic
    u_w0 = (Re * mu_water) / (l32 * rho_water)   # cm/s, assume constant downstream, x-directed velocity equal to average velocity of water as in Curl (1974)
    
    grain_diam_max = (2**(np.log2(0.00055249*u_w0**2)+3))/10 
    grain_diam_min = 0.0177
    if grain_diam_max > 10:
        grain_diam_max = 10
    
    max_time = 20  #seconds
    abrasion_start_location = 500   # only particles that travel > 2 m will contribute to abrasion
    # =============================================================================
    
    #build the bedrock scallop array
    dx = 0.05
    dx0 = dx/l32
    xScal = np.arange(0, numScal+dx0,dx0)  #x-array for scallop field
    uScal = np.arange(0,1+dx0,dx0)  #x-array for a single scallop
    
    x0, z0 = da.scallop_array(xScal, uScal, int(numScal), l32)   #initial scallop profile, dimensions in centimeters
    z0 = z0 - np.min(z0)
    dzdx = np.gradient(z0, x0)
    theta2 = np.arctan(dzdx)  #slope angle at each point along scalloped profile
    
    #build the flowfield matrix
    nx = l32*20 + 1
    ny = l32*20 + 1
    new_x = np.linspace(0, l32/numScal, int(nx))
    new_z = np.linspace(0, l32/numScal, int(ny))
    new_X, new_Z = np.meshgrid(new_x, new_z)
    new_u = np.zeros((int(ny), int(nx)))
    new_w = np.zeros((int(ny), int(nx)))
    
    if flow_regime == 'laminar':
        u_water, w_water = da.laminar_flowfield(xScal, uScal, numScal, l32)
    elif flow_regime == 'turbulent':
        u_water, w_water = da.turbulent_flowfield(xScal, uScal, numScal, new_u, new_w, l32)
    
    # In[6]:
    
    # definitions and parameters
    
    diam = grain_diam_max * np.logspace((np.log10(grain_diam_min/grain_diam_max)), 0, n)
    All_Initial_Conditions = np.zeros(shape = (n, numPrtkl, 5))
    All_Impacts = np.zeros(shape = (n, 500000, 10))
    All_Deposits = np.zeros(shape = (n, 500000, 5))
    All_Distances = np.zeros(shape = (n, numPrtkl, 2))
    
    # loop over diameter array to run the saltation function for each grainsize
    i = 0
    for D in diam:
        if D < 0.0063:
            grain = 'silt'
        elif D >= 0.0063 and D < 0.2:
            grain = 'sand'
        elif D >= 0.2 and D < 6.4:
            grain = 'pebbles'
        elif D >= 6.4 and D < 25.6:
            grain = 'cobbles'
        elif D >= 25.6:
            grain = 'boulders'
    
        rho_quartz = 2.65  # g*cm^-3
        rho_ls = 2.3
        cb = 0.01    #bedload sediment concentration
        
        cH = np.max(z0)   # crest height
    
        # In[10]:
        
        impact_data, loc_data, init_con, distance_traveled= da.sediment_saltation(x0, z0, w_water, u_water, u_w0, D, dx, theta2, mu_water, cH, l32, numPrtkl, max_time, abrasion_start_location)
        
        All_Initial_Conditions[i, :, :] = init_con
        All_Impacts[i, :len(impact_data), :] = impact_data
        All_Distances[i, :len(distance_traveled), :] = distance_traveled
        
        print('diam = ' + str(diam[i]) + ' cm')
        i += 1
        
        if n <= 30:
            fig, axs = spl.trajectory_figures(l32, numScal, D, grain, x0, z0, loc_data)
            plt.show()
    
    # ####save all data
    import datetime
    now = datetime.datetime.now()
    time_stamp = now.strftime('%Y-%m-%d')
    np.save(join(outfolder,'Impacts-'+str(l32)+flow_regime+time_stamp), All_Impacts)
    np.save(join(outfolder,'InitialConditions-'+str(l32)+flow_regime+time_stamp), All_Initial_Conditions)
    np.save(join(outfolder,'TravelDistances-'+str(l32)+flow_regime+time_stamp), All_Distances)
    
    print('data saved, scallop length = ', l32)
    
    # # ####plot results; all plotting schemes available in scallopplotlib.py
    # pars, stdevs, res, fig, axs = spl.average_velocities_plot_fit_to_Dietrich(rho_quartz, rho_water, diam, l32, All_Impacts, numPrtkl)
    # plt.show()
    
    # # fig, axs = spl.seperate_impact_locations_plot(EnergyAtImpact, diam, x0, z0, XAtImpact, ZAtImpact, uScal, l32, numScal)
    # # plt.show()
    
    # # fig, axs = spl.abrasion_and_dissolution_plot_2(x0, diam, NormErosionAvg, l32)
    # # plt.show()
    
    # fig, axs = spl.number_of_impacts_at_loc_plot(diam, x0, z0, l32, All_Impacts, All_Initial_Conditions, numScal)
    # plt.show()
    
    # fig, axs = spl.travel_distance(All_Distances, diam, l32)
    # plt.show()