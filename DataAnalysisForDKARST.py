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

Initial_Conditions1 = np.load('outputs\InitialConditions-1turbulent2021-05-28.npy')
Initial_Conditions2 = np.load('outputs\InitialConditions-2.5turbulent2021-05-28.npy')
Initial_Conditions5 = np.load('outputs\InitialConditions-5turbulent2021-05-28.npy')
Initial_Conditions10 = np.load('outputs\InitialConditions-10turbulent2021-05-28.npy')
# =============================================================================
# Form of Initial Conditions array:
#   initial conditions saved for each particle in simulation      
#       shape = (n, numPrtkl, 5) 
#           n = number of grain sizes in diameter array
#           numPrtkl = number of particles simulated for each grain size
#           0 = x, 1 = z, 2 = u, 3 = w, D = particle diameter 
# =============================================================================
    
Impact_Data1 = np.load('outputs\Impacts-1turbulent2021-05-28.npy')
Impact_Data2 = np.load('outputs\Impacts-2.5turbulent2021-05-28.npy')
Impact_Data5 = np.load('outputs\Impacts-5turbulent2021-05-28.npy')
Impact_Data10 = np.load('outputs\Impacts-10turbulent2021-05-28.npy')
# =============================================================================
# Form of Impact Data array:
#   data collected every time a particle impacts the bedrock surface
#       shape = (n, 100000, 9)
#           n = number of grain sizes in diameter array
#           100000 pre-allocated to be greater than number of time-steps
#           0 = time, 1 = x, 2 = z, 3 = u, 4 = w, 5 = D, 6 = |Vel|, 7 = KE, 8 = particle ID,
#               links to numPartkl in Initial Conditions array, 9 = cumulative erosion
# =============================================================================

Deposition_Data1 = np.load('outputs\TravelDistances-1turbulent2021-05-28.npy')
Deposition_Data2 = np.load('outputs\TravelDistances-2.5turbulent2021-05-28.npy')
Deposition_Data5 = np.load('outputs\TravelDistances-5turbulent2021-05-28.npy')
Deposition_Data10 = np.load('outputs\TravelDistances-10turbulent2021-05-28.npy')
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
    
#     ## impacts at locations plot
#     fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
#     GetMaxEnergies = Impact_Data[i][:, :, 7][Impact_Data[i][:, :, 7] != 0]
#     ColorScheme = np.log10(GetMaxEnergies)  ## define color scheme to be consistent for every plot
#     ColorNumbers = ColorScheme[np.logical_not(np.isnan(ColorScheme))] 
#     ColorMax = np.ceil(np.max(ColorNumbers))
#     my_colors = cm.get_cmap('YlGn', 256)
#     # axs.set_xlim(0, int(number_of_scallops[i]*scallop_lengths[i]))
#     axs.set_xlim(0, 50)
#     axs.set_ylim(0, l32)
#     for j in range(len(diam)):
#         GS = Impact_Data[i][j, :, 5][Impact_Data[i][j, :, 7] != 0]
#         initial_z_idxs = np.array(Impact_Data[i][j, :, 8][Impact_Data[i][j, :, 7] != 0], dtype = int)
#         impact_x = Impact_Data[i][j, :, 1][Impact_Data[i][j, :, 7] != 0]
#         findColors = (np.log10(Impact_Data[i][j, :, 7][Impact_Data[i][j, :, 7] != 0]))/ColorMax
#         axs.scatter(impact_x, Initial_Conditions[i][j, initial_z_idxs, 1] , c = my_colors(findColors), s = 50 * GS)
#     plt.fill_between(x0, z0/4, 0, alpha = 1, color = 'grey')
#     fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
#                         wspace=0.4, hspace=0.1)
#     axs.axvspan(0, 50, facecolor='mediumblue', zorder = 0)
#     plt.title('Particle impacts at each location by fall height on '+str(l32)+' cm Scallops')
#     cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
#     norm = colors.Normalize(vmin = 0, vmax = ColorMax)
#     plt.colorbar(cm.ScalarMappable(norm = norm, cmap='YlGn'), cax = cb_ax)
#     cb_ax.set_ylabel('log10 of Kinetic energy of impact (ergs)')
#     axs.set_xlabel('x (cm)')
#     axs.set_ylabel('fall height (cm)')
#     plt.show()

# #     ####fitting velocity data to Dietrich settling curve
# #     rho_sediment = 2.65
# #     rho_fluid = 1
        
# #     fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
# #     g = 981 # cm*s^-2
# #     nu = 0.01307  # g*cm^-1*s^-1
    
# #     D_star = np.zeros_like(diam)
# #     W_star = np.zeros_like(diam)
    
# #     for j in range(len(diam)):
# #         not_nan_idx = np.where(~np.isnan(Impact_Data[i][j, :, 6]))
# #         diameter_array = np.average(Impact_Data[i][j, :, 5][not_nan_idx])
# #         VelocityAvg = np.average(Impact_Data[i][j, :, 6][not_nan_idx])
# #         D_star[j] = ((rho_sediment-rho_fluid)*g*(diameter_array)**3)/(rho_fluid*nu)
# #         W_star[j] = -(rho_fluid*VelocityAvg**3)/((rho_sediment-rho_fluid)*g*nu)
# #     W_star_Dietrich = (1.71 * 10**-4 * D_star**2)
# #     axs.scatter(D_star, W_star, c = 'b', label = 'simulated impact velocity')
# # #        axs.scatter(diameter_array*10, VelocityAvg)#, label = 'simulated impact velocity')

# #     axs.scatter(D_star, W_star_Dietrich, c = 'r', label = 'settling velocity (Dietrich, 1982)')
        
# #     def settling_velocity(D_star, r, s):
# #         return r * D_star**s
    
# #     pars, cov = curve_fit(f=settling_velocity, xdata=D_star, ydata=W_star, p0=[1.71 * 10**-4, 2], bounds=(-np.inf, np.inf))
# #     # Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
# #     stdevs = np.sqrt(np.diag(cov))
# #     # Calculate the residuals
# #     res = W_star - settling_velocity(D_star, *pars)
# #     axs.plot(D_star, settling_velocity(D_star, *pars), linestyle='--', linewidth=2, color='black', label = 'fit, r= '+str(round(pars[0], 10))+' and s= '+str(round(pars[1], 2)))
        
# #     plt.legend()
# #     plt.semilogy()
# #     #plt.semilogx()
# #     axs.set_xlabel('dimensionless grain size, D*')
# #     axs.set_ylabel('dimensionless settling velocity, W*') 
# #     axs.set_title('Particle velocities over '+str(l32)+' cm scallops, fit to Settling Velocity of Natural Particles (Dietrich, 1982)')
    
# #     plt.show()


    
        # impacts by scallop phase plot, scallop crest == 0, 2*pi
        ## try linearly 
    fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
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
    plt.fill_between(x0, z0 * 50/l32**2, 0, alpha = 1, color = 'grey')
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
    
#     # if l32 == 1 or l32 == 2.5:
#     #         #select the x-range for the zoomed region
#     #     x1 = 0
#     #     x2 = l32
        
#     #     # select y-range for zoomed region
#     #     y1 = 0
#     #     y2 = l32*5
        
#     #     # Make the zoom-in plot:
#     #     axins = zoomed_inset_axes(axs, 2, bbox_to_anchor=(0,0), loc = 'upper left')
#     #     GetMaxEnergies = Impact_Data[i][:, :, 7][Impact_Data[i][:, :, 7] != 0]
#     #     ColorScheme = np.log10(GetMaxEnergies)  ## define color scheme to be consistent for every plot
#     #     ColorNumbers = ColorScheme[np.logical_not(np.isnan(ColorScheme))] 
#     #     ColorMax = np.ceil(np.max(ColorNumbers))
#     #     my_colors = cm.get_cmap('winter_r', 256)

#     #     for j in range(len(diam)):
#     #         GS = Impact_Data[i][j, :, 5][Impact_Data[i][j, :, 7] != 0]
#     #         initial_z_idxs = np.array(Impact_Data[i][j, :, 8][Impact_Data[i][j, :, 7] != 0], dtype = int)
#     #         impact_x = Impact_Data[i][j, :, 1][Impact_Data[i][j, :, 7] != 0]
#     #         scallop_phase = impact_x % l32
#     #         findColors = (np.log10(Impact_Data[i][j, :, 7][Impact_Data[i][j, :, 7] != 0]))/ColorMax
#     #         axins.scatter(scallop_phase, GS*10 , c = my_colors(findColors))
#     #     axins.set_xlim(x1, x2)
#     #     axins.set_ylim(y1, y2)
#     #     axins.set_title('impacts by grains with D < 0.5*scallop length')
#     #     axins.grid(True, which = 'both', axis = 'both')
#     #     plt.xticks(visible=True)
#     #     plt.yticks(visible=True)
#     #     mark_inset(axs, axins, loc1=2, loc2=1, fc="none", ec="0.5")
    plt.show()

#     #     # deposition by scallop phase plot, scallop crest == 0, 2*pi
#     #     ## try linearly 
#     # fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
#     # axs.set_xlim(0, l32)
#     # #axs.set_aspect('equal')
#     # for j in range(len(diam)):
#     #     GS_dep = Deposition_Data[i][j, :, 3][Deposition_Data[i][j, :, 3] > 0]
#     #     deposit_x = Deposition_Data[i][j, :, 1][Deposition_Data[i][j, :, 3] > 0]
#     #     scallop_phase = deposit_x % l32
#     #     axs.scatter(scallop_phase, GS_dep*10)
#     # plt.fill_between(x0, z0, 0, alpha = 1, color = 'grey')
#     # #plt.contourf(new_X, new_Z, w_water, alpha = 1, vmin = -20, vmax = 20, cmap = 'seismic', zorder = 0)
#     # fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
#     #                     wspace=0.4, hspace=0.1)
#     # plt.title('Particle deposits by grain size on '+str(l32)+' cm scallops')
#     # cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
#     # axs.set_xlabel('x (cm)')
#     # axs.set_ylabel('grain size (mm)')
#     # plt.show()


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

# #def travel_distance(All_Distances, diameter_array, scallop_length):
# fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
# for i in range(len(scallop_lengths)):
    
#     axs.scatter(all_grains[i, :]*10, all_avg_distances[i, :], label = str(scallop_lengths[i])+' cm scallop')
# plt.semilogy()
# plt.semilogx()
# plt.legend()
# axs.set_ylabel('average travel distance (cm)') 
# axs.set_title('Average distance traveled by grain size over scallops')
# axs.set_xlabel('grain size (mm)')
# plt.show()

# fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
# for i in range(len(scallop_lengths)):
#     axs.scatter(all_grains[i, :]*10, all_impact_numbers[i, :], label = 'impacts on '+str(scallop_lengths[i])+' cm scallop')
# plt.title('Number of particle impacts on scallops per cm of streambed length')
# axs.set_xlabel('grain size (mm)')
# axs.set_ylabel('number of impacts')
# plt.legend()
# plt.semilogx()
# axs.grid(True, which = 'both', axis = 'x')
# plt.show()

#################comparing dissolution and abrasion
cb_max = 0.02
cb_tiny = 4 * 10**-5
cb = np.linspace(cb_tiny, cb_max, 5)

for h in range(len(cb)):
    fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
    erosion_rates = np.zeros(shape = (len(scallop_lengths), len(Impact_Data1)))

    for i in range(len(scallop_lengths)):
        for j in range(len(diam)):
            if np.any(Deposition_Data[i][j, :, 1]):
                cb_sim = np.shape(Initial_Conditions1)[1]*np.pi*(all_grains[i,j])**2/(4*np.average(Deposition_Data[i][j, :, 1])*5000)   #####approximating bounce-height as 10/l32. update when the value is stored in simulation
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
    plt.legend(loc = 'upper left')
    axs.set_title('Abrasion and Dissolution Rates Over Scallops with Sediment Concentration =' +str(round(cb[h], 5)))
    axs.set_xlabel('particle grainsize (mm)')
    axs.set_ylabel('erosion rate (cm/s)')
    axs.grid(True, which = 'both', axis = 'both')
    
    
    plt.show()

# #################comparing dissolution and abrasion, 3d scatter
# cb_max = 0.02
# cb_tiny = 4 * 10**-5
# cb = np.linspace(cb_tiny, cb_max, 21)


# fig = plt.figure(figsize = (11,8.5))
# ax = fig.add_subplot(projection='3d')

# abrasion_rates = np.zeros(shape = (len(scallop_lengths), len(Impact_Data1), len(cb)))


# for i in range(len(scallop_lengths)):
    
#     for j in range(len(all_grains[i, :])):
#         for k in range(len(cb)):
#             if np.any(Deposition_Data[i][j, :, 1]):
#                 cb_sim = np.shape(Initial_Conditions1)[1]*np.pi*(all_grains[i,j])**2/(4*np.average(Deposition_Data[i][j, :, 1])*5000)   
#                 total_elapsed_time = np.max(Impact_Data[i][j, :, 0])            
#                 Abrasion_Rate = (Impact_Data[i][j, :, 9][Impact_Data[i][j, :, 6] < 0])/(total_elapsed_time)
#                 if np.any(Abrasion_Rate):
#                     abrasion_rates[i, j, k] = cb[k]*np.sum(Abrasion_Rate)/cb_sim
#             else:
#                 abrasion_rates[i,j, k]=0
#             if abrasion_rates [i, j, k] < diss_min [i]:
#                 m = 'o'
#                 c = 'b'
#             if abrasion_rates [i, j, k] >= diss_min [i] and abrasion_rates [i, j, k] <= diss_max[i]:
#                 m = '^'
#                 c = 'orange'
#             if abrasion_rates [i, j, k] > diss_max[i]:
#                 m = '*'
#                 c= 'brown'
                
            
#             ax.scatter(np.log10(all_grains[i, j]*10), scallop_lengths[i], cb[k], marker = m, color = c, alpha = 0.5) 


# ax.invert_xaxis()
# ax.set_title('dissolution = o, both = ^, abrasion = *')
# ax.set_xlabel('log10[particle grainsize (mm)]')
# ax.set_zlabel('sediment concentration')
# ax.set_ylabel('scallop length (cm)')


# plt.show()

#################comparing dissolution and abrasion, bust out into four plots
cb_max = 0.02
cb_tiny = 4 * 10**-5
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
            # elif erosion_difference[i,j,k] > -39 and erosion_difference[i,j,k] < 39:
            #     c = 'magenta'
            else:
                c = my_colors(findColors)
            axs.scatter((all_grains[i, j]*10), cb[k], color = c, marker = 's') 

    plt.semilogx()
    plt.semilogy()
    axs.set_xlim(x_min[i], (np.max(all_grains[i,:]))*10)
    axs.set_title('Relative erosional processes over ' +str(scallop_lengths[i])+ '-cm scallops')
    axs.set_xlabel('particle grainsize (mm)')
    axs.set_ylabel('sediment concentration')
    cb_ax = fig.add_axes([0.93, 0.1, 0.02, 0.8])
    norm = colors.Normalize(vmin = -ColorMax, vmax = ColorMax)
    plt.colorbar(cm.ScalarMappable(norm = norm, cmap='RdYlBu_r'), cax = cb_ax)
    cb_ax.set_ylabel('percent change abrasion to dissolution (cm/s)')
    
    plt.show()

#################comparing dissolution and abrasion, collapse all to 2D
cb_max = 0.02
cb_tiny = 4 * 10**-5
cb = cb_max * np.logspace((np.log10(cb_tiny/cb_max)), 0, 51)

abrasion_rates = np.zeros(shape = (len(scallop_lengths), len(Impact_Data1), len(cb)))
erosion_difference = np.zeros(shape = (len(scallop_lengths), len(Impact_Data1), len(cb)))

percent_change_max = [174800, 130700, 278900, 174800]
hiding_size = [0, 0.05, 0.08, 0.1]
x_min = [0.177, 0.4, 0.7, 0.9]

fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
for i in range(len(scallop_lengths)):

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
            elif erosion_difference[i,j,k] < -39:
                c = 'cyan'
            elif erosion_difference[i,j,k] > 39:
                c = 'magenta'
            elif erosion_difference[i,j,k] > -39 and erosion_difference[i,j,k] < 39:
                c = 'yellow'
            else:
                c = my_colors(findColors)
            axs.scatter((all_grains[i, j]/scallop_lengths[i]), cb[k], color = c, marker = 's', alpha = 0.5) 

plt.semilogx()
plt.semilogy()
#axs.set_xlim(x_min[i], (np.max(all_grains[i,:]))*10)
axs.set_title('Relative erosional processes over scallops')
axs.set_xlabel('particle grainsize/scallop length')
axs.set_ylabel('sediment concentration')
cb_ax = fig.add_axes([0.93, 0.1, 0.02, 0.8])
norm = colors.Normalize(vmin = -ColorMax, vmax = ColorMax)
plt.colorbar(cm.ScalarMappable(norm = norm, cmap='RdYlBu_r'), cax = cb_ax)
cb_ax.set_ylabel('percent change abrasion to dissolution (cm/s)')

plt.show()
