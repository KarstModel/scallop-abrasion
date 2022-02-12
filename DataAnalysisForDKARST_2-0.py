# -*- coding: utf-8 -*-
"""
Version 2-0, 12 February 2022
Rachel Bosch

Created on Thurs March 18 21:36:05 2021

@author: rachel bosch

This code replaces "DataAnalysisScratchPad" to reflect major changes in data storage and retrieval.
"""
import numpy as np
from matplotlib import pyplot as plt
import darthabrader as da

Initial_Conditions1 = np.load('outputs\InitialConditions-1turbulent2022-01-14.npy')
Initial_Conditions2 = np.load('outputs\InitialConditions-2.5turbulent2022-01-15.npy')
Initial_Conditions5 = np.load('outputs\InitialConditions-5turbulent2022-01-15.npy')
Initial_Conditions10 = np.load('outputs\InitialConditions-10turbulent2022-01-15.npy')
# =============================================================================
# Form of Initial Conditions array:
#   initial conditions saved for each particle in simulation      
#       shape = (n, numPrtkl, 5) 
#           n = number of grain sizes in diameter array
#           numPrtkl = number of particles simulated for each grain size
#           0 = x, 1 = z, 2 = u, 3 = w, D = particle diameter 
# =============================================================================
    
Impact_Data1 = np.load('outputs\Impacts-1turbulent2022-01-14.npy')
Impact_Data2 = np.load('outputs\Impacts-2.5turbulent2022-01-15.npy')
Impact_Data5 = np.load('outputs\Impacts-5turbulent2022-01-15.npy')
Impact_Data10 = np.load('outputs\Impacts-10turbulent2022-01-15.npy')
# =============================================================================
# Form of Impact Data array:
#   data collected every time a particle impacts the bedrock surface
#       shape = (n, 100000, 9)
#           n = number of grain sizes in diameter array
#           100000 pre-allocated to be greater than number of time-steps
#           0 = time, 1 = x, 2 = z, 3 = u, 4 = w, 5 = D, 6 = |Vel|, 7 = KE, 8 = particle ID,
#               links to numPartkl in Initial Conditions array, 9 = cumulative erosion
# =============================================================================

Deposition_Data1 = np.load('outputs\TravelDistances-1turbulent2022-01-14.npy')
Deposition_Data2 = np.load('outputs\TravelDistances-2.5turbulent2022-01-15.npy')
Deposition_Data5 = np.load('outputs\TravelDistances-5turbulent2022-01-15.npy')
Deposition_Data10 = np.load('outputs\TravelDistances-10turbulent2022-01-15.npy')
# =============================================================================
# Form of Deposit Data array:
#   data collected every time a particle impacts the bedrock surface
#   0 = cumulative distance traveled, 1 = maximum bounce height
# =============================================================================

scallop_lengths = [1, 2.5, 5, 10]
number_of_scallops= [400/scallop_lengths[0], 400/scallop_lengths[1], 400/scallop_lengths[2], 400/scallop_lengths[3]]
Impact_Data = [Impact_Data1, Impact_Data2, Impact_Data5, Impact_Data10]


Initial_Conditions = [Initial_Conditions1, Initial_Conditions2, Initial_Conditions5, Initial_Conditions10]
Deposition_Data = [Deposition_Data1, Deposition_Data2, Deposition_Data5, Deposition_Data10]
x_stretch = [8000, 8000, 1600, 400]
all_grains = np.zeros(shape = (len(scallop_lengths), len(Impact_Data1)))
all_impact_numbers = np.zeros(shape = (len(scallop_lengths), len(Impact_Data1)))                      
all_avg_energies = np.zeros(shape = (len(scallop_lengths), len(Impact_Data1)))                      
all_avg_distances = np.zeros(shape = (len(scallop_lengths), len(Deposition_Data1)))

Wentworth = ['medium silt', 'coarse silt', 'very fine sand', 'fine sand', 'medium sand', 'coarse sand', 'very coarse sand', 'granules', 'pebbles', 'cobbles']
Ww_max_sizes = [0.031, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 64, 256]

for i in range(len(scallop_lengths)):
    ###pull the turbulent flowfield data
    l32 = scallop_lengths[i]
    dx0 = 0.05/l32
    numScal = int(number_of_scallops[i])
    xScal = np.arange(0, numScal + dx0, dx0)
    uScal = np.arange(0, 1 + dx0, dx0)
    x0, z0 = da.scallop_array(xScal, uScal, numScal, l32)
    z0 = z0 - np.min(z0)
    
    #rebuild the diameter array
    n = np.shape(Impact_Data[i])[0]
    rho_water = 1
    Re = 23300     #Reynold's number from scallop formation experiments (Blumberg and Curl, 1974)
    mu_water = 0.01307  # g*cm^-1*s^-1  #because we are in cgs, value of kinematic viscosity of water = dynamic
    u_w0 = (Re * mu_water) / (l32 * rho_water)   # cm/s, assume constant downstream, x-directed velocity equal to average velocity of water as in Curl (1974)
    grain_diam_max = (2**(np.log2(0.00055249*u_w0**2)+3))/10 
    grain_diam_min = 0.0177
    if grain_diam_max > 10:
        grain_diam_max = 10
    diam = grain_diam_max * np.logspace((np.log10(grain_diam_min/grain_diam_max)), 0, int(n))
    
    #break diameter array into bins by Wentworth classification
    m_silt_all = np.where(diam <= Ww_max_sizes[0], diam, 0)
    m_silt = np.delete(m_silt_all, np.where(m_silt_all==0))
    c_silt_all = np.where(diam > Ww_max_sizes[0], diam, 0)
    c_silt_all = np.where(c_silt_all <= Ww_max_sizes[1], c_silt_all, 0)
    c_silt = np.delete(c_silt_all, np.where(c_silt_all==0))
    vf_sand_all = np.where(diam > Ww_max_sizes[1], diam, 0)
    vf_sand_all = np.where(vf_sand_all <= Ww_max_sizes[2], vf_sand_all, 0)
    vf_sand = np.delete(vf_sand_all, np.where(vf_sand_all==0))
    f_sand_all = np.where(diam > Ww_max_sizes[2], diam, 0)
    f_sand_all = np.where(f_sand_all <= Ww_max_sizes[3], f_sand_all, 0)
    f_sand = np.delete(f_sand_all, np.where(f_sand_all==0))
    m_sand_all = np.where(diam > Ww_max_sizes[3], diam, 0)
    m_sand_all = np.where(m_sand_all <= Ww_max_sizes[4], m_sand_all, 0)
    m_sand = np.delete(m_sand_all, np.where(m_sand_all==0))
    c_sand_all = np.where(diam > Ww_max_sizes[4], diam, 0)
    c_sand_all = np.where(c_sand_all <= Ww_max_sizes[5], c_sand_all, 0)
    c_sand = np.delete(c_sand_all, np.where(c_sand_all==0))
    vc_sand_all = np.where(diam > Ww_max_sizes[5], diam, 0)
    vc_sand_all = np.where(vc_sand_all <= Ww_max_sizes[6], vc_sand_all, 0)
    vc_sand = np.delete(vc_sand_all, np.where(vc_sand_all==0))
    granules_all = np.where(diam > Ww_max_sizes[6], diam, 0)
    granules_all = np.where(granules_all <= Ww_max_sizes[7], granules_all, 0)
    granules = np.delete(granules_all, np.where(granules_all==0))
    pebbles_all = np.where(diam > Ww_max_sizes[7], diam, 0)
    pebbles_all = np.where(pebbles_all <= Ww_max_sizes[8], pebbles_all, 0)
    pebbles = np.delete(pebbles_all, np.where(pebbles_all==0))
    cobbles_all = np.where(diam > Ww_max_sizes[8], diam, 0)
    cobbles = np.delete(cobbles_all, np.where(cobbles_all==0))
    
    grain_size_bins_all = [m_silt_all, c_silt_all, vf_sand_all, f_sand_all, m_sand_all, c_sand_all, vc_sand_all, granules_all, pebbles_all, cobbles_all]
    grain_size_bins = [m_silt, c_silt, vf_sand, f_sand, m_sand, c_sand, vc_sand, granules, pebbles, cobbles]

    

######## set up for number_of_impacts_plot(diameter_array, NumberOfImpactsByGS, scallop_length, x_array):
    
    for k in range(len(grain_size_bins_all)):
        ## impacts at locations plot
        fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
        axs.set_xlim(0, l32*20)
        number_of_impacts_at_x = np.zeros_like(x0)
        number_of_new_impacts_at_x = np.zeros_like(x0)
        
       
        
        for j in range(len(grain_size_bins_all[k])):
            # new_impact_indices = np.where([Impact_Data[i][j, :, 7] != 0])[1]
            
            new_impact_indices = (np.round((Impact_Data[i][j, np.where([Impact_Data[i][j, :, 7] != 0])[1], 1])/0.05, 0)).astype(int)
            number_of_new_impacts_at_x[new_impact_indices] = 1
            number_of_impacts_at_x += number_of_new_impacts_at_x
            
            
        plt.fill_between(x0, z0/4, 0, alpha = 1, color = 'grey')
        plt.scatter(x0, number_of_impacts_at_x)
        fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                            wspace=0.4, hspace=0.1)
        plt.title('Number of particle impacts at each location of '+Wentworth[k]+' on '+str(l32)+' cm Scallops')
        axs.set_xlabel('x (cm)')
        axs.set_ylabel('number of impacts')
        plt.show()
