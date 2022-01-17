# -*- coding: utf-8 -*-
"""
Created on Thurs March 18 21:36:05 2021

@author: rachel bosch

This code replaces "DataAnalysisScratchPad" to reflect major changes in data storage and retrieval.
"""
import numpy as np
from matplotlib import pyplot as plt
import darthabrader as da
from scipy.optimize import curve_fit
from matplotlib import colors
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes 
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

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

######## set up for number_of_impacts_plot(diameter_array, NumberOfImpactsByGS, scallop_length, x_array):
    number_of_impacts=np.zeros_like(diam)
    normalized_energies=np.zeros_like(diam)
    averaged_travel_dist=np.zeros_like(diam)
    for j in range(len(diam)):
        GS = Impact_Data[i][j, :, 5][Impact_Data[i][j, :, 7] != 0]
        number_of_impacts[j] = len(GS)/(numScal*l32)
        KE = Impact_Data[i][j, :, 7][Impact_Data[i][j, :, 7]!=0]
        S = Deposition_Data[i][j, :, 0][Deposition_Data[i][j, :, 0]!=0]
        if np.any(KE):
            normalized_energies[j] = np.average(KE)
        else:
            normalized_energies[j] = 0
        if np.any(S):
            averaged_travel_dist[j] = np.average(S)
        else:
            averaged_travel_dist[j] = 0
        
    all_impact_numbers[i, :] = number_of_impacts
    all_grains[i, :] = diam
    all_avg_energies[i, :] = normalized_energies
    all_avg_distances[i, :] = averaged_travel_dist
    
    ## impacts at locations plot
    fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
    GetMaxEnergies = Impact_Data[i][:, :, 7][Impact_Data[i][:, :, 7] != 0]
    ColorScheme = np.log10(GetMaxEnergies)  ## define color scheme to be consistent for every plot
    ColorNumbers = ColorScheme[np.logical_not(np.isnan(ColorScheme))] 
    ColorMax = np.ceil(np.max(ColorNumbers))
    my_colors = cm.get_cmap('YlGn', 256)
    # axs.set_xlim(0, int(number_of_scallops[i]*scallop_lengths[i]))
    axs.set_xlim(0, 50)
    axs.set_ylim(0, l32)
    for j in range(len(diam)):
        GS = Impact_Data[i][j, :, 5][Impact_Data[i][j, :, 7] != 0]
        initial_z_idxs = np.array(Impact_Data[i][j, :, 8][Impact_Data[i][j, :, 7] != 0], dtype = int)
        impact_x = Impact_Data[i][j, :, 1][Impact_Data[i][j, :, 7] != 0]
        findColors = (np.log10(Impact_Data[i][j, :, 7][Impact_Data[i][j, :, 7] != 0]))/ColorMax
        axs.scatter(impact_x, Initial_Conditions[i][j, initial_z_idxs, 1] , c = my_colors(findColors), s = 50 * GS)
    plt.fill_between(x0, z0/4, 0, alpha = 1, color = 'grey')
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                        wspace=0.4, hspace=0.1)
    axs.axvspan(0, 50, facecolor='mediumblue', zorder = 0)
    plt.title('Particle impacts at each location by fall height on '+str(l32)+' cm Scallops')
    cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
    norm = colors.Normalize(vmin = 0, vmax = ColorMax)
    plt.colorbar(cm.ScalarMappable(norm = norm, cmap='YlGn'), cax = cb_ax)
    cb_ax.set_ylabel('log10 of Kinetic energy of impact (ergs)')
    axs.set_xlabel('x (cm)')
    axs.set_ylabel('fall height (cm)')
    plt.show()

    fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
    # GetMaxEnergies = Impact_Data[i][:, :, 7][Impact_Data[i][:, :, 7] != 0][Impact_Data[i][:, :, 1]>100]
    GetMaxEnergies = Impact_Data[i][:, :, 7][Impact_Data[i][:, :, 7] != 0]
    ColorScheme = np.log10(GetMaxEnergies)  ## define color scheme to be consistent for every plot
    ColorNumbers = ColorScheme[np.logical_not(np.isnan(ColorScheme))] 
    ColorMax = np.ceil(np.max(ColorNumbers))
    my_colors = cm.get_cmap('cividis', 256)
    axs.set_xlim(0, l32)
    #axs.set_aspect('equal')
    for j in range(len(diam)):
        GS = Impact_Data[i][j, :, 5][Impact_Data[i][j, :, 7] != 0]
        initial_z_idxs = np.array(Impact_Data[i][j, :, 8][Impact_Data[i][j, :, 7] != 0], dtype = int)
        impact_x = Impact_Data[i][j, :, 1][Impact_Data[i][j, :, 7] != 0]
        scallop_phase = impact_x % l32
        findColors = (np.log10(Impact_Data[i][j, :, 7][Impact_Data[i][j, :, 7] != 0]))/ColorMax
        axs.scatter(scallop_phase, GS*10 , c = my_colors(findColors))
    # plt.fill_between(x0, z0 * 50/l32**2, 0, alpha = 1, color = 'grey')
    #plt.contourf(new_X, new_Z, w_water, alpha = 1, vmin = -20, vmax = 20, cmap = 'seismic', zorder = 0)
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                        wspace=0.4, hspace=0.1)
    plt.title('Particle impacts by grain size on '+str(l32)+' cm scallops', fontsize = 18)
    cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
    #cb2_ax = fig.add_axes([0.93, 0.1, 0.02, 0.8])
    norm = colors.Normalize(vmin = 0, vmax = ColorMax)
    #norm2 = colors.Normalize(vmin = -20, vmax = 20)
    plt.colorbar(cm.ScalarMappable(norm = norm, cmap='cividis'), cax = cb_ax)
    #plt.colorbar(cm.ScalarMappable(norm = norm2, cmap='seismic'), cax = cb2_ax)
    cb_ax.set_ylabel('log10 of Kinetic energy of impact (ergs)', fontsize = 16)
    #cb2_ax.set_ylabel('vertical component of water velocity (cm/s)')
    axs.set_xlabel('x (cm)', fontsize = 16)
    axs.set_ylabel('grain size (mm)', fontsize = 16)
    
    plt.show()

#################comparing dissolution and abrasion
cb_max = -3
cb_tiny = -6
cb =  7 * np.logspace(cb_tiny, cb_max, 4)

for h in range(len(cb)):
    fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
    erosion_rates = np.zeros(shape = (len(scallop_lengths), len(Impact_Data1)))

    for i in range(len(scallop_lengths)):
        for j in range(len(diam)):
            if np.any(Deposition_Data[i][j, :, 1]):
                cb_sim = np.shape(Initial_Conditions1)[1]*np.pi*(all_grains[i,j])**2/(4*np.average(Deposition_Data[i][j, :, 1])*5000)   
                total_elapsed_time = np.max(Impact_Data[i][j, :, 0])            
                Abrasion_Rate = (Impact_Data[i][j, :, 9][Impact_Data[i][j, :, 6] < 0])/(total_elapsed_time)
                if np.any(Abrasion_Rate):
                    erosion_rates[i, j] = cb[h]*np.sum(Abrasion_Rate)/cb_sim
            else:
                erosion_rates[i,j]=0
        axs.scatter(all_grains[i, :]*10, (erosion_rates[i, :]), label = '(abrasion * scallop length) on '+str(scallop_lengths[i])+' cm scallops')

    axs.set_xlim(0.1, 110)
    diss_min1 = (5*1.735*10**-8)  #minimum dissolution rate (mm/yr) (Grm et al., 2017) scaled for 1 cm scallops
    diss_max1 = (5*4*10**-8)  #maximum dissolution rate (mm/yr) (Hammer et al., 2011)  scaled for 1 cm scallops
    diss_min2 = (2*1.735*10**-8)  #minimum dissolution rate (mm/yr) (Grm et al., 2017) scaled for 2.5 cm scallops
    diss_max2 = (2*4*10**-8)  #maximum dissolution rate (mm/yr) (Hammer et al., 2011)  scaled for 2.5 cm scallops
    diss_min5 = (1.735*10**-8)  #minimum dissolution rate (mm/yr) (Grm et al., 2017)
    diss_max5 = (4*10**-8)  #maximum dissolution rate (mm/yr) (Hammer et al., 2011)
    diss_min10 = (0.5*1.735*10**-8)  #minimum dissolution rate (mm/yr) (Grm et al., 2017) scaled for 10 cm scallops
    diss_max10 = (0.5*4*10**-8)  #maximum dissolution rate (mm/yr) (Hammer et al., 2011)  scaled for 10 cm scallops
    
    diss_min = [diss_min1, diss_min2, diss_min5, diss_min10]
    diss_max = [diss_max1, diss_max2, diss_max5, diss_max10]


    x = np.linspace(0.1, 110)
    plt.fill_between(x, diss_min1, diss_max1, alpha = 0.4, color = 'dodgerblue', label = 'dissolutional range over 1 cm scallops')
    plt.fill_between(x, diss_min2, diss_max2, alpha = 0.4, color = 'peru', label = 'dissolutional range over 2.5 cm scallops')
    plt.fill_between(x, diss_min5, diss_max5, alpha = 0.4, color = 'cyan', label = 'dissolutional range over 5 cm scallops')
    plt.fill_between(x, diss_min10, diss_max10, alpha = 0.4, color = 'gray', label = 'dissolutional range over 10 cm scallops')

    # axs.set_ylim(0, 2 * diss_max1)
    plt.semilogx()
    #plt.legend(loc = 'upper left')
    axs.set_title('Sediment Concentration =' +str(round(cb[h], 6)), fontsize = 18)
    axs.set_xlabel('particle grainsize (mm)', fontsize = 16)
    axs.set_ylabel('erosion rate (cm/s)', fontsize = 16)
    axs.grid(True, which = 'both', axis = 'both')
    
    
    plt.show()


#################comparing dissolution and abrasion, bust out into four plots
cb_max = 0.015
cb_tiny = 7 * 10**-6
cb = cb_max * np.logspace((np.log10(cb_tiny/cb_max)), 0, 51)

abrasion_rates = np.zeros(shape = (len(scallop_lengths), len(Impact_Data1), len(cb)))
erosion_difference = np.zeros(shape = (len(scallop_lengths), len(Impact_Data1), len(cb)))

percent_change_max = [174800, 130700, 278900, 174800]
hiding_size = [0, 0.05, 0.08, 0.1]
x_min = [0.177, 0.4, 0.7, 0.9]


for i in range(len(scallop_lengths)):
    fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
    mean_dissolution = 0.5 * (diss_min[i] + diss_max[i])
    ColorMax = np.ceil(100)
    my_colors = cm.get_cmap('RdYlBu_r', 256)
    
    for j in range(len(all_grains[i, :])):
        for k in range(len(cb)):
            if np.any(Deposition_Data[i][j, :, 1]):
                cb_sim = np.shape(Initial_Conditions1)[1]*np.pi*(all_grains[i,j])**2/(4*np.average(Deposition_Data[i][j, :, 1])*5000)   
                total_elapsed_time = np.max(Impact_Data[i][j, :, 0])            
                Abrasion_Rate = (Impact_Data[i][j, :, 9][Impact_Data[i][j, :, 6] < 0])/(total_elapsed_time)
                if np.any(Abrasion_Rate):
                    abrasion_rates[i, j, k] = cb[k]*np.sum(Abrasion_Rate)/cb_sim
            else:
                abrasion_rates[i,j, k]=0
            erosion_difference[i,j,k] = ((abrasion_rates[i, j, k] - mean_dissolution)/mean_dissolution)*100

            findColors = (erosion_difference[i,j,k])/ColorMax
            
            if all_grains[i, j] < hiding_size[i]:
                c = 'gainsboro'
            # elif erosion_difference[i,j,k] < -39:
            #     c = 'aqua'
            # elif erosion_difference[i,j,k] > 39:
            #     c = 'darkgoldenrod'
            elif erosion_difference[i,j,k] > -39 and erosion_difference[i,j,k] < 39:
                c = 'yellow'
            else:
                c = my_colors(findColors)
            axs.scatter((all_grains[i, j]*10), cb[k], color = c, marker = 's', s = 100) 

    plt.semilogx()
    plt.semilogy()
    #axs.set_xlim(x_min[i], (np.max(all_grains[i,:]))*10)
    axs.set_xlim(x_min[0], 100)
    axs.set_title('Relative erosional processes over ' +str(scallop_lengths[i])+ '-cm scallops', fontsize = 18)
    axs.set_xlabel('particle grainsize (mm)', fontsize = 16)
    axs.set_ylabel('sediment concentration', fontsize = 16)
    cb_ax = fig.add_axes([0.93, 0.1, 0.02, 0.8])
    norm = colors.Normalize(vmin = -ColorMax, vmax = ColorMax)
    plt.colorbar(cm.ScalarMappable(norm = norm, cmap='RdYlBu_r'), cax = cb_ax)
    cb_ax.set_ylabel('percent change abrasion to dissolution (cm/s)', fontsize = 16)
    
    plt.show()

#################comparing dissolution and abrasion, bust out into four plots, grain sizes as ratios
cb_max = 0.015
cb_tiny = 7 * 10**-6
cb = cb_max * np.logspace((np.log10(cb_tiny/cb_max)), 0, 51)

abrasion_rates = np.zeros(shape = (len(scallop_lengths), len(Impact_Data1), len(cb)))
erosion_difference = np.zeros(shape = (len(scallop_lengths), len(Impact_Data1), len(cb)))

percent_change_max = [174800, 130700, 278900, 174800]
hiding_size = [0, 0.05, 0.08, 0.1]
x_min = [0.177, 0.4, 0.7, 0.9]



fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
mean_dissolution = 0.5 * (diss_min[i] + diss_max[i])
ColorMax = np.ceil(100)
my_colors = cm.get_cmap('RdYlBu_r', 256)
for i in range(len(scallop_lengths)):    
    for j in range(len(all_grains[i, :])):
        for k in range(len(cb)):
            if np.any(Deposition_Data[i][j, :, 1]):
                cb_sim = np.shape(Initial_Conditions1)[1]*np.pi*(all_grains[i,j])**2/(4*np.average(Deposition_Data[i][j, :, 1])*5000)   
                total_elapsed_time = np.max(Impact_Data[i][j, :, 0])            
                Abrasion_Rate = (Impact_Data[i][j, :, 9][Impact_Data[i][j, :, 6] < 0])/(total_elapsed_time)
                if np.any(Abrasion_Rate):
                    abrasion_rates[i, j, k] = cb[k]*np.sum(Abrasion_Rate)/cb_sim
            else:
                abrasion_rates[i,j, k]=0
            erosion_difference[i,j,k] = ((abrasion_rates[i, j, k] - mean_dissolution)/mean_dissolution)*100

            findColors = (erosion_difference[i,j,k])/ColorMax
            
            # if all_grains[i, j] < hiding_size[i]:
            #     c = 'gainsboro'
            # elif erosion_difference[i,j,k] < -39:
            #     c = 'aqua'
            # elif erosion_difference[i,j,k] > 39:
            #     c = 'darkgoldenrod'
            if erosion_difference[i,j,k] > -39 and erosion_difference[i,j,k] < 39 and i == 0:
                c = 'tab:red'
                a = 1
            elif erosion_difference[i,j,k] > -39 and erosion_difference[i,j,k] < 39 and i == 1:
                c = 'tab:blue'
                a = 1
            elif erosion_difference[i,j,k] > -39 and erosion_difference[i,j,k] < 39 and i == 2:
                c = 'tab:green'
                a = 1
            elif erosion_difference[i,j,k] > -39 and erosion_difference[i,j,k] < 39 and i == 3:
                c = 'tab:purple'
                a = 1
            else:
                c = '1'
                a = 0
            # else:
            #     c = my_colors(findColors)
            axs.scatter((all_grains[i, j]/scallop_lengths[i]), cb[k], color = c, marker = 's', s = 100, alpha = a) 

plt.semilogx()
plt.semilogy()
#axs.set_xlim(x_min[i], (np.max(all_grains[i,:]))*10)
#axs.set_xlim(x_min[0], 100)
axs.set_title('Relative erosional processes over scallops', fontsize = 20)
axs.set_xlabel('particle grainsize : scallop length', fontsize = 18)
axs.set_ylabel('sediment concentration', fontsize = 18)
cb_ax = fig.add_axes([0.93, 0.1, 0.02, 0.8])
norm = colors.Normalize(vmin = -ColorMax, vmax = ColorMax)
plt.colorbar(cm.ScalarMappable(norm = norm, cmap='RdYlBu_r'), cax = cb_ax)
cb_ax.set_ylabel('percent change abrasion to dissolution (cm/s)', fontsize = 16)

plt.show()