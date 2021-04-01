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

Initial_Conditions1 = np.load('outputs\InitialConditions-1turbulent2021-03-24.npy')
Initial_Conditions2 = np.load('outputs\InitialConditions-2.5turbulent2021-03-23.npy')
Initial_Conditions5 = np.load('outputs\InitialConditions-5turbulent2021-03-23.npy')
Initial_Conditions10 = np.load('outputs\InitialConditions-10turbulent2021-03-24.npy')
# =============================================================================
# Form of Initial Conditions array:
#   initial conditions saved for each particle in simulation      
#       shape = (n, numPrtkl, 5) 
#           n = number of grain sizes in diameter array
#           numPrtkl = number of particles simulated for each grain size
#           0 = x, 1 = z, 2 = u, 3 = w, D = particle diameter 
# =============================================================================
    
Impact_Data1 = np.load('outputs\Impacts-1turbulent2021-03-24.npy')
Impact_Data2 = np.load('outputs\Impacts-2.5turbulent2021-03-23.npy')
Impact_Data5 = np.load('outputs\Impacts-5turbulent2021-03-23.npy')
Impact_Data10 = np.load('outputs\Impacts-10turbulent2021-03-24.npy')
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
number_of_scallops=np.zeros_like(scallop_lengths)
number_of_scallops[0] = np.rint(450*1.5**(-scallop_lengths[0]))
number_of_scallops[1] = np.rint(450*1.5**(-scallop_lengths[1]))
number_of_scallops[2] = 40
number_of_scallops[3] = np.rint(450*1.5**(-scallop_lengths[3]))
Impact_Data = [Impact_Data1, Impact_Data2, Impact_Data5, Impact_Data10]
Initial_Conditions = [Initial_Conditions1, Initial_Conditions2, Initial_Conditions5, Initial_Conditions10]
x_stretch = [8000, 8000, 1600, 400]
all_grains = np.zeros(shape = (len(scallop_lengths), len(Impact_Data1)))
all_impact_numbers = np.zeros(shape = (len(scallop_lengths), len(Impact_Data1)))                      
all_avg_energies = np.zeros(shape = (len(scallop_lengths), len(Impact_Data1)))                      


for i in range(len(scallop_lengths)):
    
    ###pull the turbulent flowfield data
    l32 = scallop_lengths[i]
    dx0 = 0.05/l32
    numScal = int(number_of_scallops[i])
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
    rho_water = 1
    Re = 23300     #Reynold's number from scallop formation experiments (Blumberg and Curl, 1974)
    mu_water = 0.01307  # g*cm^-1*s^-1  #because we are in cgs, value of kinematic viscosity of water = dynamic
    u_w0 = (Re * mu_water) / (l32 * rho_water)   # cm/s, assume constant downstream, x-directed velocity equal to average velocity of water as in Curl (1974)
    grain_diam_max = (2**-(-np.log2(5.525*(u_w0/100)**2)-3))/10 
    grain_diam_min = 0.05
    diam = grain_diam_max * np.logspace((np.log10(grain_diam_min/grain_diam_max)), 0, int(n))
    
    # ## impacts at locations plot
    # fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
    # GetMaxEnergies = Impact_Data[i][:, :, 7][Impact_Data[i][:, :, 7] != 0]
    # ColorScheme = np.log10(GetMaxEnergies)  ## define color scheme to be consistent for every plot
    # ColorNumbers = ColorScheme[np.logical_not(np.isnan(ColorScheme))] 
    # ColorMax = np.ceil(np.max(ColorNumbers))
    # my_colors = cm.get_cmap('YlGn', 256)
    # # axs.set_xlim(0, int(number_of_scallops[i]*scallop_lengths[i]))
    # axs.set_xlim(0, 50)
    # axs.set_ylim(0, l32)
    # for j in range(len(diam)):
    #     GS = Impact_Data[i][j, :, 5][Impact_Data[i][j, :, 7] != 0]
    #     initial_z_idxs = np.array(Impact_Data[i][j, :, 8][Impact_Data[i][j, :, 7] != 0], dtype = int)
    #     impact_x = Impact_Data[i][j, :, 1][Impact_Data[i][j, :, 7] != 0]
    #     findColors = (np.log10(Impact_Data[i][j, :, 7][Impact_Data[i][j, :, 7] != 0]))/ColorMax
    #     axs.scatter(impact_x, Initial_Conditions[i][j, initial_z_idxs, 1] , c = my_colors(findColors), s = 50 * GS)
    # plt.fill_between(x0, z0/4, 0, alpha = 1, color = 'grey')
    # fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
    #                     wspace=0.4, hspace=0.1)
    # axs.axvspan(0, 50, facecolor='mediumblue', zorder = 0)
    # plt.title('Particle impacts at each location by fall height on '+str(l32)+' cm Scallops')
    # cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
    # norm = colors.Normalize(vmin = 0, vmax = ColorMax)
    # plt.colorbar(cm.ScalarMappable(norm = norm, cmap='YlGn'), cax = cb_ax)
    # cb_ax.set_ylabel('log10 of Kinetic energy of impact (ergs)')
    # axs.set_xlabel('x (cm)')
    # axs.set_ylabel('fall height (cm)')
    # plt.show()

# ######## set up for number_of_impacts_plot(diameter_array, NumberOfImpactsByGS, scallop_length, x_array):
#     number_of_impacts=np.zeros_like(diam)
#     normalized_energies=np.zeros_like(diam)
#     for j in range(len(diam)):
#         GS = Impact_Data[i][j, :, 5][Impact_Data[i][j, :, 7] != 0]
#         number_of_impacts[j] = len(GS)/(numScal*l32)
#         KE = Impact_Data[i][j, :, 7][Impact_Data[i][j, :, 7]!=0]
#         if np.any(KE):
#             normalized_energies[j] = np.average(KE)
#         else:
#             normalized_energies[j] = 0
#     all_impact_numbers[i, :] = number_of_impacts
#     all_grains[i, :] = diam
#     all_avg_energies[i, :] = normalized_energies
    
    # #####abrasion v dissolution 
    # # abrasion_and_dissolution_plot_2(x_array, diam, NormErosionAvg, scallop_length):
    # cb_max = 0.02
    # cb_tiny = 4 * 10**-5
    # cb_old = 0.01
    # cb = np.linspace(cb_tiny, cb_max, 5)
    
    # fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
    # for j in range(len(diam)):
    #     for k in range(len(cb)):
    #         GS = Impact_Data[i][j, :, 5][Impact_Data[i][j, :, 6] < 0]
    #         Abrasion_Rate = -B * cb[k]/cb_old * (Impact_Data[i][j, :, 6][Impact_Data[i][j, :, 6] < 0])**3 *315576000 #  mm * s / (cm * yr)
    #         Normalized_Abrasion_Rate = Abrasion_Rate/((Abrasion_Rate > 0).sum())
    #         axs.scatter((GS*10), (Normalized_Abrasion_Rate))#, label = 'bedload concentration = '+str(round(cb[k], 5)))
    # axs.set_xlim(0.1, grain_diam_max*10)
    # diss_min = 5.66* 5 / l32   #minimum dissolution rate (mm/yr) (Grm et al., 2017)
    # diss_max = 12.175* 5 / l32  #maximum dissolution rate (mm/yr) (Hammer et al., 2011)
    # x = np.linspace(0.1, grain_diam_max*10)
    # plt.fill_between(x, diss_min, diss_max, alpha = 0.4, color = 'gray', label = 'dissolutional range')
    # plt.semilogx()
    # plt.legend(loc = 'upper left')
    # axs.set_title('Abrasion Rate Normalized by Number of Impacts on '+str(l32)+' cm Scallops')
    # axs.set_xlabel('particle grainsize (mm)')
    # axs.set_ylabel('abrasional erosion rate (mm/yr)')
    # axs.grid(True, which = 'both', axis = 'both')
    
    # #select the x-range for the zoomed region
    # x1 = 0.1
    # x2 = np.min([l32*5, grain_diam_max*10])
    
    # # select y-range for zoomed region
    # y1 = 0
    # y2 = diss_max
    
    # # Make the zoom-in plot:
    # axins = zoomed_inset_axes(axs, 3.5, bbox_to_anchor=(0,0), loc = 'upper left')
    # for j in range(len(diam)):
    #     for k in range(len(cb)):
    #         GS = Impact_Data[i][j, :, 5][Impact_Data[i][j, :, 6] < 0]
    #         Abrasion_Rate = -B * cb[k]/cb_old * (Impact_Data[i][j, :, 6][Impact_Data[i][j, :, 6] < 0])**3 *315576000 #  mm * s / (cm * yr)
    #         Normalized_Abrasion_Rate = Abrasion_Rate/((Abrasion_Rate > 0).sum())
    #         axins.scatter((GS*10), (Normalized_Abrasion_Rate), s= 50)#, label = 'bedload concentration = '+str(round(cb[k], 5)))
    # axins.fill_between(x, diss_min, diss_max, alpha = 0.6, color = 'r', label = 'dissolutional range')
    # axins.set_xlim(x1, x2)
    # axins.set_ylim(y1, y2)
    # axins.grid(True, which = 'both', axis = 'both')
    # plt.xticks(visible=True)
    # plt.yticks(visible=True)
    # axins.legend(loc = 'upper center')
    # mark_inset(axs, axins, loc1=2, loc2=1, fc="none", ec="0.5")
    # plt.show()
    
    # ####fitting velocity data to Dietrich settling curve--not working quite right, old version
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

    ####fitting velocity data to Dietrich settling curve--not working quite right, reworking 2021-03-28
    rho_sediment = 2.65
    rho_fluid = 1
        
    fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
    g = 981 # cm*s^-2
    nu = 0.01307  # g*cm^-1*s^-1
    
    D_star = np.zeros_like(diam)
    W_star = np.zeros_like(diam)
    
    for j in range(len(diam)):
        not_nan_idx = np.where(~np.isnan(Impact_Data[i][j, :, 6]))
        diameter_array = np.average(Impact_Data[i][j, :, 5][not_nan_idx])
        VelocityAvg = np.average(Impact_Data[i][j, :, 6][not_nan_idx])
        D_star[j] = ((rho_sediment-rho_fluid)*g*(diameter_array)**3)/(rho_fluid*nu)
        W_star[j] = -(rho_fluid*VelocityAvg**3)/((rho_sediment-rho_fluid)*g*nu)
    W_star_Dietrich = (1.71 * 10**-4 * D_star**2)
    axs.scatter(D_star, W_star, c = 'b', label = 'simulated impact velocity')
#        axs.scatter(diameter_array*10, VelocityAvg)#, label = 'simulated impact velocity')

    axs.scatter(D_star, W_star_Dietrich, c = 'r', label = 'settling velocity (Dietrich, 1982)')
        
    def settling_velocity(D_star, r, s):
        return r * D_star**s
    
    pars, cov = curve_fit(f=settling_velocity, xdata=D_star, ydata=W_star, p0=[1.71 * 10**-4, 2], bounds=(-np.inf, np.inf))
    # Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
    stdevs = np.sqrt(np.diag(cov))
    # Calculate the residuals
    res = W_star - settling_velocity(D_star, *pars)
    axs.plot(D_star, settling_velocity(D_star, *pars), linestyle='--', linewidth=2, color='black', label = 'fit, r= '+str(round(pars[0], 10))+' and s= '+str(round(pars[1], 2)))
        
    plt.legend()
    plt.semilogy()
    #plt.semilogx()
    axs.set_xlabel('dimensionless grain size, D*')
    axs.set_ylabel('dimensionless settling velocity, W*') 
    axs.set_title('Particle velocities over '+str(l32)+' cm scallops, fit to Settling Velocity of Natural Particles (Dietrich, 1982)')
    
    plt.show()


    
    #     # impacts by scallop phase plot, scallop crest == 0, 2*pi
    #     ## try linearly 
    # fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
    # GetMaxEnergies = Impact_Data[i][:, :, 7][Impact_Data[i][:, :, 7] != 0]
    # ColorScheme = np.log10(GetMaxEnergies)  ## define color scheme to be consistent for every plot
    # ColorNumbers = ColorScheme[np.logical_not(np.isnan(ColorScheme))] 
    # ColorMax = np.ceil(np.max(ColorNumbers))
    # my_colors = cm.get_cmap('winter_r', 256)
    # axs.set_xlim(0, l32)
    # #axs.set_aspect('equal')
    # for j in range(len(diam)):
    #     GS = Impact_Data[i][j, :, 5][Impact_Data[i][j, :, 7] != 0]
    #     initial_z_idxs = np.array(Impact_Data[i][j, :, 8][Impact_Data[i][j, :, 7] != 0], dtype = int)
    #     impact_x = Impact_Data[i][j, :, 1][Impact_Data[i][j, :, 7] != 0]
    #     scallop_phase = impact_x % l32
    #     findColors = (np.log10(Impact_Data[i][j, :, 7][Impact_Data[i][j, :, 7] != 0]))/ColorMax
    #     axs.scatter(scallop_phase, GS*10 , c = my_colors(findColors))
    # plt.fill_between(x0, z0, 0, alpha = 1, color = 'grey')
    # #plt.contourf(new_X, new_Z, w_water, alpha = 1, vmin = -20, vmax = 20, cmap = 'seismic', zorder = 0)
    # fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
    #                     wspace=0.4, hspace=0.1)
    # plt.title('Particle impacts by grain size on '+str(l32)+' cm scallops')
    # cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
    # #cb2_ax = fig.add_axes([0.93, 0.1, 0.02, 0.8])
    # norm = colors.Normalize(vmin = 0, vmax = ColorMax)
    # #norm2 = colors.Normalize(vmin = -20, vmax = 20)
    # plt.colorbar(cm.ScalarMappable(norm = norm, cmap='winter_r'), cax = cb_ax)
    # #plt.colorbar(cm.ScalarMappable(norm = norm2, cmap='seismic'), cax = cb2_ax)
    # cb_ax.set_ylabel('log10 of Kinetic energy of impact (ergs)')
    # #cb2_ax.set_ylabel('vertical component of water velocity (cm/s)')
    # axs.set_xlabel('x (cm)')
    # axs.set_ylabel('grain size (mm)')
    
    # if l32 == 1 or l32 == 2.5:
    #         #select the x-range for the zoomed region
    #     x1 = 0
    #     x2 = l32
        
    #     # select y-range for zoomed region
    #     y1 = 0
    #     y2 = l32*5
        
    #     # Make the zoom-in plot:
    #     axins = zoomed_inset_axes(axs, 2, bbox_to_anchor=(0,0), loc = 'upper left')
    #     GetMaxEnergies = Impact_Data[i][:, :, 7][Impact_Data[i][:, :, 7] != 0]
    #     ColorScheme = np.log10(GetMaxEnergies)  ## define color scheme to be consistent for every plot
    #     ColorNumbers = ColorScheme[np.logical_not(np.isnan(ColorScheme))] 
    #     ColorMax = np.ceil(np.max(ColorNumbers))
    #     my_colors = cm.get_cmap('winter_r', 256)

    #     for j in range(len(diam)):
    #         GS = Impact_Data[i][j, :, 5][Impact_Data[i][j, :, 7] != 0]
    #         initial_z_idxs = np.array(Impact_Data[i][j, :, 8][Impact_Data[i][j, :, 7] != 0], dtype = int)
    #         impact_x = Impact_Data[i][j, :, 1][Impact_Data[i][j, :, 7] != 0]
    #         scallop_phase = impact_x % l32
    #         findColors = (np.log10(Impact_Data[i][j, :, 7][Impact_Data[i][j, :, 7] != 0]))/ColorMax
    #         axins.scatter(scallop_phase, GS*10 , c = my_colors(findColors))
    #     axins.set_xlim(x1, x2)
    #     axins.set_ylim(y1, y2)
    #     axins.set_title('impacts by grains with D < 0.5*scallop length')
    #     axins.grid(True, which = 'both', axis = 'both')
    #     plt.xticks(visible=True)
    #     plt.yticks(visible=True)
    #     mark_inset(axs, axins, loc1=2, loc2=1, fc="none", ec="0.5")
    # plt.show()


#           ## impacts by scallop phase plot, scallop crest == 0, 2*pi
#         ### try radially (distance from center proportional to impact energy?)
#     fig = plt.figure()
#     axs = fig.add_subplot(111, projection = 'polar')
#     labels = 'crest', 'lee', 'trough', 'stoss'
#     theta = [0.47*np.pi, 0.795*np.pi, 1.3*np.pi, 1.975*np.pi]
#     width = [0.14*np.pi, 0.51*np.pi, 0.5*np.pi, 0.85*np.pi]
#     c = 'lightsteelblue', 'slateblue', 'darkslateblue', 'steelblue'
#     GetMaxEnergies = Impact_Data[i][:, :, 7][Impact_Data[i][:, :, 7] != 0]
#     ColorScheme = np.log10(GetMaxEnergies)  ## define color scheme to be consistent for every plot
#     ColorNumbers = ColorScheme[np.logical_not(np.isnan(ColorScheme))] 
#     ColorMax = np.ceil(np.max(ColorNumbers))
#     my_colors = cm.get_cmap('YlGn', 256)
#     for j in range(len(diam)):
#         GS = Impact_Data[i][j, :, 5][Impact_Data[i][j, :, 7] != 0]
#         initial_z_idxs = np.array(Impact_Data[i][j, :, 8][Impact_Data[i][j, :, 7] != 0], dtype = int)
#         fall_heights = Initial_Conditions[i][j, initial_z_idxs, 1]
#         impact_x = Impact_Data[i][j, :, 1][Impact_Data[i][j, :, 7] != 0]
#         scallop_phase = 2*np.pi*(impact_x % l32)/l32 + np.pi/2
#         findColors = (np.log10(Impact_Data[i][j, :, 7][Impact_Data[i][j, :, 7] != 0]))/ColorMax
#         axs.scatter(scallop_phase, GS*10, c = my_colors(findColors), zorder = 1)#, s = 50 * GS)
#     axs.bar(theta, grain_diam_max*10, width = width, alpha = 0.5, color = c, zorder = 0)
#     fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
#                         wspace=0.4, hspace=0.1)
#     plt.title('Particle impacts at each location on '+str(l32)+' cm Scallops')
#     plt.xticks([])
#     cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
#     norm = colors.Normalize(vmin = 0, vmax = ColorMax)
#     plt.colorbar(cm.ScalarMappable(norm = norm, cmap='YlGn'), cax = cb_ax)
#     cb_ax.set_ylabel('log10 of Kinetic energy of impact (ergs)')
#     axs.annotate(labels[0],
#                 xy=(theta[0], l32),  # theta, radius
#                 xytext=(0.425, 0.8),    # fraction, fraction
#                 textcoords='figure fraction',
#                 )
#     axs.annotate(labels[1],
#                 xy=(theta[1], l32),  # theta, radius
#                 xytext=(0.25, 0.7),    # fraction, fraction
#                 textcoords='figure fraction',
#                 )
#     axs.annotate(labels[2],
#                 xy=(theta[2], l32),  # theta, radius
#                 xytext=(0.25, 0.25),    # fraction, fraction
#                 textcoords='figure fraction',
#                 )
#     axs.annotate(labels[3],
#                 xy=(theta[3], l32),  # theta, radius
#                 xytext=(0.6, 0.4),    # fraction, fraction
#                 textcoords='figure fraction',
#                 )
#     axs.annotate('grain size (mm)',
#                 xy=(theta[3], l32),
#                 xytext=(0.65, 0.7),    # fraction, fraction
#                 textcoords='figure fraction',
# )
#     plt.show()   
    
#     ###reference scallop
# fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
# labels = 'crest', 'lee', 'trough', 'stoss'
# axs.set_xlim(0, 2*np.pi)
# axs.set_ylim(0, 1.5)
# axs.set_aspect('equal')
# plt.fill_between(x0*2*np.pi/l32, z0/1.25, 0, alpha = 1, color = 'grey')
# axs.axvspan(0, 0.04*np.pi, facecolor='lightsteelblue', alpha = 0.5, zorder = 0)
# axs.axvspan(0.04*np.pi, 0.55*np.pi, facecolor='slateblue', alpha = 0.5, zorder = 0)
# axs.axvspan(0.55*np.pi, 1.05*np.pi, facecolor='darkslateblue', alpha = 0.5, zorder = 0)
# axs.axvspan(1.05*np.pi, 1.9*np.pi, facecolor='steelblue', alpha = 0.5, zorder = 0)
# axs.axvspan(1.9*np.pi, 2*np.pi, facecolor='lightsteelblue', alpha = 0.5, zorder = 0)
# axs.annotate(labels[0],
#             xy=(0.02*np.pi, 1.5),  # theta, radius
#             xytext=(0.01, 0.8),    # fraction, fraction
#             textcoords='axes fraction',
#             )
# axs.annotate(labels[1],
#             xy=(0.08*np.pi, 1.5),  # theta, radius
#             xytext=(0.11, 0.8),    # fraction, fraction
#             textcoords='axes fraction',
#             )
# axs.annotate(labels[2],
#             xy=(0.6*np.pi, 1.5),  # theta, radius
#             xytext=(0.35, 0.8),    # fraction, fraction
#             textcoords='axes fraction',
#             )
# axs.annotate(labels[3],
#             xy=(1.45*np.pi, 1.5),  # theta, radius
#             xytext=(0.725, 0.8),    # fraction, fraction
#             textcoords='axes fraction',
#             )
# axs.annotate(labels[0],
#             xy=(1.45*np.pi, 1.5),  # theta, radius
#             xytext=(0.96, 0.8),    # fraction, fraction
#             textcoords='axes fraction',
#             )
# axs.set_xlabel('distance along scallop (radians)')
# axs.set_ylabel('fraction of height to scallop crest') 
# plt.show()

# #pretty scalloped profile only
# fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
# axs.set_xlim(0, 50)
# axs.set_ylim(-0.5, 2)
# axs.set_aspect('equal')
# plt.fill_between(x0, z0, 0, alpha = 1, color = 'grey')

# ######## number_of_impacts_plot(diameter_array, NumberOfImpactsByGS, scallop_length, x_array):

# fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
# for i in range(len(scallop_lengths)):
#     axs.scatter(all_avg_energies[i, :], all_impact_numbers[i, :], label = 'impacts on '+str(scallop_lengths[i])+' cm scallop')
# plt.title('Number of particle impacts on scallops per cm of streambed length')
# axs.set_xlabel('avg KE')
# axs.set_ylabel('number of impacts')
# plt.legend()
# plt.semilogx()
# axs.grid(True, which = 'both', axis = 'x')
# plt.show()
