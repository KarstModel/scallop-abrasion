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
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes 
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib import colors
from matplotlib import cm
from matplotlib.colors import ListedColormap

Initial_Conditions1 = np.load('outputs\InitialConditions-1turbulent2021-03-20.csv.npy')
Initial_Conditions2 = np.load('outputs\InitialConditions-2.5turbulent2021-03-20.csv.npy')
Initial_Conditions5 = np.load('outputs\InitialConditions-5turbulent2021-03-19.csv.npy')
Initial_Conditions10 = np.load('outputs\InitialConditions-10turbulent2021-03-20.csv.npy')
# =============================================================================
# Form of Initial Conditions array:
#   initial conditions saved for each particle in simulation      
#       shape = (n, numPrtkl, 5) 
#           n = number of grain sizes in diameter array
#           numPrtkl = number of particles simulated for each grain size
#           0 = x, 1 = z, 2 = u, 3 = w, D = particle diameter 
# =============================================================================
    
Impact_Data1 = np.load('outputs\Impacts-1turbulent2021-03-20.csv.npy')
Impact_Data2 = np.load('outputs\Impacts-2.5turbulent2021-03-20.csv.npy')
Impact_Data5 = np.load('outputs\Impacts-5turbulent2021-03-19.csv.npy')
Impact_Data10 = np.load('outputs\Impacts-10turbulent2021-03-20.csv.npy')
# =============================================================================
# Form of Impact Data array:
#   data collected every time a particle impacts the bedrock surface
#       shape = (n, 100000, 9)
#           n = number of grain sizes in diameter array
#           100000 pre-allocated to be greater than number of time-steps
#           0 = time, 1 = x, 2 = z, 3 = u, 4 = w, 5 = D, 6 = |Vel|, 7 = KE, 8 = particle ID,
#               links to numPartkl in Initial Conditions array
# =============================================================================

B = 8.82*10**-12  # s**2·cm**-2,  abrasion coefficient (Bosch and Ward, 2021)

scallop_lengths = [1, 2.5, 5, 10]
subscripts = [1, 2, 5, 10]
number_of_scallops = [80, 80, 40, 20]
Impact_Data = [Impact_Data1, Impact_Data2, Impact_Data5, Impact_Data10]
Initial_Conditions = [Initial_Conditions1, Initial_Conditions2, Initial_Conditions5, Initial_Conditions10]
x_stretch = [8000, 8000, 1600, 400]



for i in range(len(scallop_lengths)):
    
    ###pull the turbulent flowfield data
    l32 = scallop_lengths[i]
    dx0 = 0.05/l32
    numScal = number_of_scallops[i]
    xScal = np.arange(0, numScal + dx0, dx0)
    uScal = np.arange(0, 1 + dx0, dx0)
    x0, z0 = da.scallop_array(xScal, uScal, numScal, l32)
    z0 = z0 - np.min(z0)
    nx = l32*20 + 1
    ny = l32*20 + 1
    nnx = l32*20*numScal + 1
    new_x = np.linspace(0, l32/numScal, int(nx))
    new_new_x = np.linspace(0, l32/numScal, int(nnx))
    new_z = np.linspace(0, l32/numScal, int(ny))
    new_u = np.zeros((int(ny), int(nx)))
    new_w = np.zeros((int(ny), int(nx)))
    new_X, new_Z = np.meshgrid(new_new_x, new_z)
    new_X = new_X*x_stretch[i]
    new_Z = new_Z*40/l32
    u_water, w_water = da.turbulent_flowfield(xScal, uScal, numScal, new_u, new_w, l32)
    
    #rebuild the diameter array
    n = np.shape(Impact_Data[i])[0]
    grain_diam_max = 0.5 * l32 
    grain_diam_min = 0.05
    diam = grain_diam_max * np.logspace((np.log10(grain_diam_min/grain_diam_max)), 0, int(n))
    
    ## impacts at locations plot
    fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
    GetMaxEnergies = Impact_Data[i][:, :, 7][Impact_Data[i][:, :, 7] != 0]
    ColorScheme = np.log10(GetMaxEnergies)  ## define color scheme to be consistent for every plot
    ColorNumbers = ColorScheme[np.logical_not(np.isnan(ColorScheme))] 
    ColorMax = np.ceil(np.max(ColorNumbers))
    my_colors = cm.get_cmap('YlGn_r', 256)
    # axs.set_xlim(0, int(number_of_scallops[i]*scallop_lengths[i]))
    axs.set_xlim(0, 50)
    for j in range(len(diam)):
        GS = Impact_Data[i][j, :, 5][Impact_Data[i][j, :, 7] != 0]
        initial_z_idxs = np.array(Impact_Data[i][j, :, 8][Impact_Data[i][j, :, 7] != 0], dtype = int)
        impact_x = Impact_Data[i][j, :, 1][Impact_Data[i][j, :, 7] != 0]
        findColors = (np.log10(Impact_Data[i][j, :, 7][Impact_Data[i][j, :, 7] != 0]))/ColorMax
        axs.scatter(impact_x, Initial_Conditions[i][j, initial_z_idxs, 1] , c = my_colors(findColors), s = 50 * GS)
    plt.fill_between(x0, z0/4, 0, alpha = 1, color = 'grey')
    plt.contourf(new_X, new_Z, w_water, alpha = 0.5, vmin = -20, vmax = 20, cmap='seismic', zorder = 0)
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                        wspace=0.4, hspace=0.1)
    plt.title('Particle impacts at each location by fall height on '+str(l32)+' cm Scallops')
    cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
    cb2_ax = fig.add_axes([0.93, 0.1, 0.02, 0.8])
    norm = colors.Normalize(vmin = 0, vmax = ColorMax)
    norm2 = colors.Normalize(vmin = -20, vmax = 20)
    plt.colorbar(cm.ScalarMappable(norm = norm, cmap='YlGn_r'), cax = cb_ax)
    plt.colorbar(cm.ScalarMappable(norm = norm2, cmap='seismic'), cax = cb2_ax)
    cb_ax.set_ylabel('log10 of Kinetic energy of impact (ergs)')
    cb2_ax.set_ylabel('water velcoity (cm/s)')
    axs.set_xlabel('x (cm)')
    axs.set_ylabel('fall height (cm)')
    plt.show()

    #####abrasion v dissolution 
    # abrasion_and_dissolution_plot_2(x_array, diam, NormErosionAvg, scallop_length):
    cb_max = 0.02
    cb_tiny = 4 * 10**-5
    cb_old = 0.01
    cb = np.linspace(cb_tiny, cb_max, 5)
    
    fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
    for j in range(len(diam)):
        for k in range(len(cb)):
            GS = Impact_Data[i][j, :, 5][Impact_Data[i][j, :, 6] < 0]
            Abrasion_Rate = -B * cb[k]/cb_old * (Impact_Data[i][j, :, 6][Impact_Data[i][j, :, 6] < 0])**3 *315576000 #  mm * s / (cm * yr)
            Normalized_Abrasion_Rate = Abrasion_Rate/((Abrasion_Rate > 0).sum())
            axs.scatter((GS*10), (Normalized_Abrasion_Rate), label = 'bedload concentration = '+str(round(cb[k], 5)))
    axs.set_xlim(0.9, grain_diam_max*10)
    diss_min = 5.66* 5 / l32   #minimum dissolution rate (mm/yr) (Grm et al., 2017)
    diss_max = 12.175* 5 / l32  #maximum dissolution rate (mm/yr) (Hammer et al., 2011)
    x = np.linspace(0.9, grain_diam_max*10)
    plt.fill_between(x, diss_min, diss_max, alpha = 0.4, color = 'gray', label = 'dissolutional range')
    plt.semilogx()
    #plt.legend(loc = 'upper left')
    axs.set_xlim(.9,30)
    axs.set_title('Abrasion Rate Normalized by Number of Impacts on '+str(l32)+' cm Scallops')
    axs.set_xlabel('particle grainsize (mm)')
    axs.set_ylabel('abrasional erosion rate (mm/yr)')
    axs.grid(True, which = 'both', axis = 'both')
    plt.show()
    
    # ####fitting velocity data to Dietrich settling curve--not owrking quite right
    # rho_sediment = 2.65
    # rho_fluid = 1
        
    # fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
    # g = 981 # cm*s^-2
    # nu = 0.01307  # g*cm^-1*s^-1
    
    # for j in range(len(diam)):
    #     not_nan_idx = np.where(~np.isnan(Impact_Data[i][j, :, 6]))
    #     diameter_array = Impact_Data[i][j, :, 5][not_nan_idx]
    #     VelocityAvg=Impact_Data[i][j, :, 6][not_nan_idx]
    #     D_star = ((rho_sediment-rho_fluid)*g*(diameter_array)**3)/(rho_fluid*nu)
    #     W_star = (rho_fluid*VelocityAvg**3)/((rho_sediment-rho_fluid)*g*nu)
    #     W_star_Dietrich = (1.71 * 10**-4 * D_star**2)
    #     axs.scatter(diameter_array*10, W_star, label = 'simulated impact velocity')
    #     axs.plot(diameter_array*10, W_star_Dietrich, c = 'g', label = 'settling velocity (Dietrich, 1982)')
        
    #     def settling_velocity(D_star, r, s):
    #         return r * D_star**s
        
    #     pars, cov = curve_fit(f=settling_velocity, xdata=D_star, ydata=W_star, p0=[1.71 * 10**-4, 2], bounds=(-np.inf, np.inf))
    #     # Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
    #     stdevs = np.sqrt(np.diag(cov))
    #     # Calculate the residuals
    #     res = W_star - settling_velocity(D_star, *pars)
    #     axs.plot(diameter_array*10, settling_velocity(D_star, *pars), linestyle='--', linewidth=2, color='black', label = 'fitted to Equation (13), with r= '+str(round(pars[0], 2))+' and s= '+str(round(pars[1], 2)))
    
    # plt.legend()
    # plt.semilogy()
    # axs.set_xlabel('diameter (mm)')
    # axs.set_ylabel('dimensionless settling velocity, Dstar') 
    # axs.set_title('Particle velocities over '+str(l32)+' cm scallops, fit to Settling Velocity of Natural Particles (Dietrich, 1982)')
    
    # plt.show()
    
        ## impacts by scallop phase plot, scallop crest == 0, 2*pi
        ### try linearly 
    fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
    GetMaxEnergies = Impact_Data[i][:, :, 7][Impact_Data[i][:, :, 7] != 0]
    ColorScheme = np.log10(GetMaxEnergies)  ## define color scheme to be consistent for every plot
    ColorNumbers = ColorScheme[np.logical_not(np.isnan(ColorScheme))] 
    ColorMax = np.ceil(np.max(ColorNumbers))
    my_colors = cm.get_cmap('YlGn_r', 256)
    axs.set_xlim(0, l32)
    for j in range(len(diam)):
        GS = Impact_Data[i][j, :, 5][Impact_Data[i][j, :, 7] != 0]
        initial_z_idxs = np.array(Impact_Data[i][j, :, 8][Impact_Data[i][j, :, 7] != 0], dtype = int)
        impact_x = Impact_Data[i][j, :, 1][Impact_Data[i][j, :, 7] != 0]
        scallop_phase = impact_x % l32
        findColors = (np.log10(Impact_Data[i][j, :, 7][Impact_Data[i][j, :, 7] != 0]))/ColorMax
        axs.scatter(scallop_phase, Initial_Conditions[i][j, initial_z_idxs, 1] , c = my_colors(findColors), s = 50 * GS)
    plt.fill_between(x0, z0, 0, alpha = 1, color = 'grey')
    plt.contourf(new_X, new_Z, w_water, alpha = 1, vmin = -20, vmax = 20, cmap = 'seismic', zorder = 0)
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                        wspace=0.4, hspace=0.1)
    plt.title('Particle impacts at each location by fall height on '+str(l32)+' cm Scallops')
    cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
    cb2_ax = fig.add_axes([0.93, 0.1, 0.02, 0.8])
    norm = colors.Normalize(vmin = 0, vmax = ColorMax)
    norm2 = colors.Normalize(vmin = -20, vmax = 20)
    plt.colorbar(cm.ScalarMappable(norm = norm, cmap='YlGn_r'), cax = cb_ax)
    plt.colorbar(cm.ScalarMappable(norm = norm2, cmap='seismic'), cax = cb2_ax)
    cb_ax.set_ylabel('log10 of Kinetic energy of impact (ergs)')
    cb2_ax.set_ylabel('vertical component of water velocity (cm/s)')
    axs.set_xlabel('x (cm)')
    axs.set_ylabel('fall height (cm)')
    plt.show()

          ## impacts by scallop phase plot, scallop crest == 0, 2*pi
        ### try radially (distance from center proportional to impact energy?)
    fig = plt.figure()
    axs = fig.add_subplot(111, projection = 'polar')
    labels = 'crest', 'lee', 'trough', 'stoss'
    theta = [0.126, 1.257, np.pi, 5.969]
    GetMaxEnergies = Impact_Data[i][:, :, 7][Impact_Data[i][:, :, 7] != 0]
    ColorScheme = np.log10(GetMaxEnergies)  ## define color scheme to be consistent for every plot
    ColorNumbers = ColorScheme[np.logical_not(np.isnan(ColorScheme))] 
    ColorMax = np.ceil(np.max(ColorNumbers))
    my_colors = cm.get_cmap('Wistia', 256)
    for j in range(len(diam)):
        GS = Impact_Data[i][j, :, 5][Impact_Data[i][j, :, 7] != 0]
        initial_z_idxs = np.array(Impact_Data[i][j, :, 8][Impact_Data[i][j, :, 7] != 0], dtype = int)
        fall_heights = Initial_Conditions[i][j, initial_z_idxs, 1]
        impact_x = Impact_Data[i][j, :, 1][Impact_Data[i][j, :, 7] != 0]
        scallop_phase = 2*np.pi*(impact_x % l32)/l32 + np.pi/2
        findColors = (np.log10(Impact_Data[i][j, :, 7][Impact_Data[i][j, :, 7] != 0]))/ColorMax
        axs.scatter(scallop_phase, fall_heights, c = my_colors(findColors), s = 50 * GS, zorder = 2)
        # if  np.max(initial_z_idxs) > max_fh:
        #     max_fh = np.max(initial_z_idxs)
    axs.bar(theta, l32, width = theta, alpha = 0.5)
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                        wspace=0.4, hspace=0.1)
    plt.title('Particle impacts at each location on '+str(l32)+' cm Scallops')
    plt.xticks([])
    cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
    norm = colors.Normalize(vmin = 0, vmax = ColorMax)
    plt.colorbar(cm.ScalarMappable(norm = norm, cmap='Wistia'), cax = cb_ax)
    cb_ax.set_ylabel('log10 of Kinetic energy of impact (ergs)')
    axs.set_ylabel('fall height (cm)')
    plt.show()   
    
