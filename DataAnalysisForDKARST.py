# -*- coding: utf-8 -*-
"""
Created on Thurs March 18 21:36:05 2021

@author: rachel bosch

This code replaces "DataAnalysisScratchPad" to reflect major changes in data storage and retrieval.
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

Initial_Conditions = np.load('outputs\InitialConditions-2.5turbulent2021-03-18.csv.npy')
# =============================================================================
# Form of Initial Conditions array:
#   initial conditions saved for each particle in simulation      
#       shape = (n, numPrtkl, 5) 
#           n = number of grain sizes in diameter array
#           numPrtkl = number of particles simulated for each grain size
#           0 = x, 1 = z, 2 = u, 3 = w, D = particle diameter 
# =============================================================================
    
Impact_Data = np.load('outputs\Impacts-2.5turbulent2021-03-18.csv.npy')
# =============================================================================
# Form of Impact Data array:
#   data collected every time a particle impacts the bedrock surface
#       shape = (n, 100000, 8)
#           n = number of grain sizes in diameter array
#           100000 pre-allocated to be greater than number of time-steps
#           0 = time, 1 = x, 2 = z, 3 = u, 4 = w, 5 = D, 6 = |Vel|, 7 = KE
# =============================================================================


# l32 = 1
# dx0 = 0.05/l32
# numScal = 60
# xScal1 = np.arange(0, numScal+dx0,dx0)  #x-array for scallop field
# uScal1 = np.arange(0,1+dx0,dx0)  #x-array for a single scallop
# x01, z01 = da.scallop_array(xScal1, uScal1, numScal, l32)   #initial scallop profile, dimensions in centimeters
# z01 = z01 - np.min(z01)

# l32 = 2.5
# dx0 = 0.05/l32
# numScal = 48
# xScal2 = np.arange(0, numScal+dx0,dx0)  #x-array for scallop field
# uScal2 = np.arange(0,1+dx0,dx0)  #x-array for a single scallop
# x02_5, z02_5 = da.scallop_array(xScal2, uScal2, numScal, l32)   #initial scallop profile, dimensions in centimeters
# z02_5 = z02_5 - np.min(z02_5)

# l32 = 5
# dx0 = 0.05/l32
# numScal = 36
# xScal5 = np.arange(0, numScal+dx0,dx0)  #x-array for scallop field
# uScal5 = np.arange(0,1+dx0,dx0)  #x-array for a single scallop
# x05, z05 = da.scallop_array(xScal5, uScal5, numScal, l32)   #initial scallop profile, dimensions in centimeters
# z05 = z05 - np.min(z05)

# l32 = 10
# dx0 = 0.05/l32
# numScal = 24
# xScal10 = np.arange(0, numScal+dx0,dx0)  #x-array for scallop field
# uScal10 = np.arange(0,1+dx0,dx0)  #x-array for a single scallop
# x010, z010 = da.scallop_array(xScal10, uScal10, numScal, l32)   #initial scallop profile, dimensions in centimeters
# z010 = z010 - np.min(z010)



# #####number_of_impacts_at_loc_plot(diameter_array, XAtImpact, scallop_x, scallop_z, scallop_length):
   
# fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
# GetMaxEnergies = KEAI1[-1, :][KEAI1[-1, :] != 0]
# ColorScheme = np.log10(GetMaxEnergies)  ## define color scheme to be consistent for every plot
# ColorNumbers = ColorScheme[np.logical_not(np.isnan(ColorScheme))] 
# ColorMax = np.ceil(np.max(ColorNumbers))
# my_colors = cm.get_cmap('gist_rainbow_r', 256)
# axs.set_xlim(0, 60)
# for i in range(len(Diam1)):
#     GS = np.ones_like(XAtImpact1)*Diam1[i]*10
#     KEAI1[i, :][KEAI1[i, :]==0] = np.nan
#     findColors = (np.log10(KEAI1[i, :]))/ColorMax
#     axs.scatter(XAtImpact1[i, :], GS[i, :], c = my_colors(findColors))
# plt.fill_between(x01, z01, 0, alpha = 1, color = 'grey')
# fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
#                     wspace=0.4, hspace=0.1)
# plt.title('Particle impacts at each location by grainsize on 1 cm Scallops')
# cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
# norm = colors.Normalize(vmin = 0, vmax = ColorMax)
# plt.colorbar(cm.ScalarMappable(norm = norm, cmap='gist_rainbow_r'), cax = cb_ax)
# cb_ax.set_ylabel('log10 of Kinetic energy of impact (ergs)')
# axs.set_xlabel('x (cm)')
# axs.set_ylabel('particle grainsize (mm)')
# plt.show()
   
# fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
# GetMaxEnergies = KEAI2_5[-1, :][KEAI2_5[-1, :] != 0]
# ColorScheme = np.log10(GetMaxEnergies)  ## define color scheme to be consistent for every plot
# ColorNumbers = ColorScheme[np.logical_not(np.isnan(ColorScheme))] 
# ColorMax = np.ceil(np.max(ColorNumbers))
# my_colors = cm.get_cmap('gist_rainbow_r', 256)
# axs.set_xlim(0, 48*2.5)
# for i in range(len(Diam2_5)):
#     GS = np.ones_like(XAtImpact2_5)*Diam2_5[i]*10
#     KEAI2_5[i, :][KEAI2_5[i, :]==0] = np.nan
#     findColors = (np.log10(KEAI2_5[i, :]))/ColorMax
#     axs.scatter(XAtImpact2_5[i, :], GS[i, :], c = my_colors(findColors))
# plt.fill_between(x02_5, z02_5, 0, alpha = 1, color = 'grey')
# fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
#                     wspace=0.4, hspace=0.1)
# plt.title('Particle impacts at each location by grainsize on 2.5 cm Scallops')
# cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
# norm = colors.Normalize(vmin = 0, vmax = ColorMax)
# plt.colorbar(cm.ScalarMappable(norm = norm, cmap='gist_rainbow_r'), cax = cb_ax)
# cb_ax.set_ylabel('log10 of Kinetic energy of impact (ergs)')
# axs.set_xlabel('x (cm)')
# axs.set_ylabel('particle grainsize (mm)')
# plt.show()
   
# fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
# GetMaxEnergies = KEAI5[-1, :][KEAI5[-1, :] != 0]
# ColorScheme = np.log10(GetMaxEnergies)  ## define color scheme to be consistent for every plot
# ColorNumbers = ColorScheme[np.logical_not(np.isnan(ColorScheme))] 
# ColorMax = np.ceil(np.max(ColorNumbers))
# my_colors = cm.get_cmap('gist_rainbow_r', 256)
# axs.set_xlim(0, 36*5)
# for i in range(len(Diam5)):
#     GS = np.ones_like(XAtImpact5)*Diam5[i]*10
#     KEAI5[i, :][KEAI5[i, :]==0] = np.nan
#     findColors = (np.log10(KEAI5[i, :]))/ColorMax
#     axs.scatter(XAtImpact5[i, :], GS[i, :], c = my_colors(findColors))
# plt.fill_between(x05, z05, 0, alpha = 1, color = 'grey')
# fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
#                     wspace=0.4, hspace=0.1)
# plt.title('Particle impacts at each location by grainsize on 5 cm Scallops')
# cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
# norm = colors.Normalize(vmin = 0, vmax = ColorMax)
# plt.colorbar(cm.ScalarMappable(norm = norm, cmap='gist_rainbow_r'), cax = cb_ax)
# cb_ax.set_ylabel('log10 of Kinetic energy of impact (ergs)')
# axs.set_xlabel('x (cm)')
# axs.set_ylabel('particle grainsize (mm)')
# plt.show()

# fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
# GetMaxEnergies = KEAI10[-1, :][KEAI10[-1, :] != 0]
# ColorScheme = np.log10(GetMaxEnergies)  ## define color scheme to be consistent for every plot
# ColorNumbers = ColorScheme[np.logical_not(np.isnan(ColorScheme))] 
# ColorMax = np.ceil(np.max(ColorNumbers))
# my_colors = cm.get_cmap('gist_rainbow_r', 256)
# axs.set_xlim(0, 24*10)
# for i in range(len(Diam10)):
#     GS = np.ones_like(XAtImpact10)*Diam10[i]*10
#     KEAI10[i, :][KEAI10[i, :]==0] = np.nan
#     findColors = (np.log10(KEAI10[i, :]))/ColorMax
#     axs.scatter(XAtImpact10[i, :], GS[i, :], c = my_colors(findColors))
# plt.fill_between(x010, z010, 0, alpha = 1, color = 'grey')
# fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
#                     wspace=0.4, hspace=0.1)
# plt.title('Particle impacts at each location by grainsize on 10 cm Scallops')
# cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
# norm = colors.Normalize(vmin = 0, vmax = ColorMax)
# plt.colorbar(cm.ScalarMappable(norm = norm, cmap='gist_rainbow_r'), cax = cb_ax)
# cb_ax.set_ylabel('log10 of Kinetic energy of impact (ergs)')
# axs.set_xlabel('x (cm)')
# axs.set_ylabel('particle grainsize (mm)')
# plt.show()

   


# #####abrasion v dissolution 
# # abrasion_and_dissolution_plot_2(x_array, diam, NormErosionAvg, scallop_length):
# cb_max = 0.02
# cb_tiny = 4 * 10**-5
# cb_old = 0.01
# cb = np.linspace(cb_tiny, cb_max, 5)

# fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
# l32 = 1
# for i in range(len(cb)):
#     NEA1_new = NEA1_old * cb[i]/cb_old
#     axs.scatter((Diam1*10), (NEA1_new), label = 'bedload concentration = '+str(round(cb[i], 5)))
# diss_min = 5.66* 5 / l32   #minimum dissolution rate (mm/yr) (Grm et al., 2017)
# diss_max = 12.175* 5 / l32  #maximum dissolution rate (mm/yr) (Hammer et al., 2011)
# x = np.linspace(0.9, 50)
# plt.fill_between(x, diss_min, diss_max, alpha = 0.4, color = 'gray', label = 'dissolutional range')
# plt.semilogx()
# plt.legend(loc = 'upper left')
# axs.set_xlim(.9,30)
# axs.set_title('Abrasion Rate Normalized by Number of Impacts on '+str(l32)+' cm Scallops')
# axs.set_xlabel('particle grainsize (mm)')
# axs.set_ylabel('abrasional erosion rate (mm/yr)')
# axs.grid(True, which = 'both', axis = 'both')
# plt.show()

# fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
# l32 = 2.5
# for i in range(len(cb)):
#     NEA2_5_new = NEA2_5_old * cb[i]/cb_old
#     axs.scatter((Diam2_5*10), (NEA2_5_new), label = 'bedload concentration = '+str(round(cb[i], 5)))
# diss_min = 5.66* 5 / l32   #minimum dissolution rate (mm/yr) (Grm et al., 2017)
# diss_max = 12.175* 5 / l32  #maximum dissolution rate (mm/yr) (Hammer et al., 2011)
# x = np.linspace(0.9, 50)
# plt.fill_between(x, diss_min, diss_max, alpha = 0.4, color = 'gray', label = 'dissolutional range')
# plt.semilogx()
# plt.legend(loc = 'upper left')
# axs.set_xlim(.9,30)
# axs.set_title('Abrasion Rate Normalized by Number of Impacts on '+str(l32)+' cm Scallops')
# axs.set_xlabel('particle grainsize (mm)')
# axs.set_ylabel('abrasional erosion rate (mm/yr)')
# axs.grid(True, which = 'both', axis = 'both')
# plt.show()

# fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
# l32 = 5
# for i in range(len(cb)):
#     NEA5_new = NEA5_old * cb[i]/cb_old
#     axs.scatter((Diam5*10), (NEA5_new), label = 'bedload concentration = '+str(round(cb[i], 5)))
# diss_min = 5.66* 5 / l32   #minimum dissolution rate (mm/yr) (Grm et al., 2017)
# diss_max = 12.175* 5 / l32  #maximum dissolution rate (mm/yr) (Hammer et al., 2011)
# x = np.linspace(0.9, 50)
# plt.fill_between(x, diss_min, diss_max, alpha = 0.4, color = 'gray', label = 'dissolutional range')
# plt.semilogx()
# plt.legend(loc = 'upper left')
# axs.set_xlim(.9,30)
# axs.set_title('Abrasion Rate Normalized by Number of Impacts on '+str(l32)+' cm Scallops')
# axs.set_xlabel('particle grainsize (mm)')
# axs.set_ylabel('abrasional erosion rate (mm/yr)')
# axs.grid(True, which = 'both', axis = 'both')
# plt.show()

# fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
# l32 = 10
# for i in range(len(cb)):
#     NEA10_new = NEA10_old * cb[i]/cb_old
#     axs.scatter((Diam10*10), (NEA10_new), label = 'bedload concentration = '+str(round(cb[i], 5)))
# diss_min = 5.66* 5 / l32   #minimum dissolution rate (mm/yr) (Grm et al., 2017)
# diss_max = 12.175* 5 / l32  #maximum dissolution rate (mm/yr) (Hammer et al., 2011)
# x = np.linspace(0.9, 50)
# plt.fill_between(x, diss_min, diss_max, alpha = 0.4, color = 'gray', label = 'dissolutional range')
# plt.semilogx()
# plt.legend(loc = 'upper left')
# axs.set_xlim(.9,30)
# axs.set_title('Abrasion Rate Normalized by Number of Impacts on '+str(l32)+' cm Scallops')
# axs.set_xlabel('particle grainsize (mm)')
# axs.set_ylabel('abrasional erosion rate (mm/yr)')
# axs.grid(True, which = 'both', axis = 'both')
# plt.show()


# # ####fitting velocity data to Dietrich settling curve
# # rho_sediment = 2.65
# # rho_fluid = 1
# # diameter_array1 = genfromtxt('./outputs/diam5turbulent2021-02-11.csv', delimiter=',')
# # diameter_array2 = genfromtxt('./outputs2/diam5turbulent2021-02-13.csv', delimiter=',')
# # diameter_array3 = genfromtxt('./outputs2/diam5turbulent2021-02-14.csv', delimiter=',')
# # scallop_length = 5
# # VelocityAvg1 = genfromtxt('./outputs/VelocityAvg5turbulent2021-02-11.csv', delimiter=',')
# # VelocityAvg2 = genfromtxt('./outputs2/VelocityAvg5turbulent2021-02-13.csv', delimiter=',')
# # VelocityAvg3 = genfromtxt('./outputs2/VelocityAvg5turbulent2021-02-14.csv', delimiter=',')

# # diameter_array = np.append(diameter_array2, diameter_array3, axis=0)
# # VelocityAvg = np.append(VelocityAvg2, VelocityAvg3, axis = 0)
    
# # fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
# # g = 981 # cm*s^-2
# # nu = 0.01307  # g*cm^-1*s^-1
# # not_nan_idx = np.where(~np.isnan(VelocityAvg1))
# # diameter_array=diameter_array1[not_nan_idx]
# # VelocityAvg=VelocityAvg1[not_nan_idx]
# # D_star = ((rho_sediment-rho_fluid)*g*(diameter_array)**3)/(rho_fluid*nu)
# # W_star = (rho_fluid*VelocityAvg**3)/((rho_sediment-rho_fluid)*g*nu)
# # W_star_Dietrich = (1.71 * 10**-4 * D_star**2)
# # axs.scatter(diameter_array*10, W_star, label = 'simulated impact velocity')
# # axs.plot(diameter_array*10, W_star_Dietrich, c = 'g', label = 'settling velocity (Dietrich, 1982)')

# # def settling_velocity(D_star, r, s):
# #     return r * D_star**s

# # pars, cov = curve_fit(f=settling_velocity, xdata=D_star, ydata=W_star, p0=[1.71 * 10**-4, 2], bounds=(-np.inf, np.inf))
# # # Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
# # stdevs = np.sqrt(np.diag(cov))
# # # Calculate the residuals
# # res = W_star - settling_velocity(D_star, *pars)
# # axs.plot(diameter_array*10, settling_velocity(D_star, *pars), linestyle='--', linewidth=2, color='black', label = 'fitted to Equation (13), with r= '+str(round(pars[0], 2))+' and s= '+str(round(pars[1], 2)))

# # plt.legend()
# # plt.semilogy()
# # axs.set_xlabel('diameter (mm)')
# # axs.set_ylabel('dimensionless settling velocity, Dstar') 
# # axs.set_title('Particle velocities over '+str(scallop_length)+' cm scallops, fit to Settling Velocity of Natural Particles (Dietrich, 1982)')

# # plt.show()


#####number_of_impacts_at_loc_plot(diameter_array, XAtImpact, scallop_x, scallop_z, scallop_length):
    #### now we will limit to plotting the first four scallops and add the flow field 
l32 = 1
dx0 = 0.05/l32
numScal = 6
xScal1 = np.arange(0, numScal+dx0,dx0)  #x-array for scallop field
uScal1 = np.arange(0,1+dx0,dx0)  #x-array for a single scallop
x01, z01 = da.scallop_array(xScal1, uScal1, numScal, l32)   #initial scallop profile, dimensions in centimeters
z01 = z01 - np.min(z01)
nx = l32*20 + 1
ny = l32*20 + 1
nnx = l32*20*numScal + 1
new_x = np.linspace(0, l32/numScal, int(nx))
new_new_x = np.linspace(0, l32/numScal, int(nnx))
new_z = np.linspace(0, l32/numScal, int(ny))
new_u = np.zeros((int(ny), int(nx)))
new_w = np.zeros((int(ny), int(nx)))
new_X, new_Z = np.meshgrid(new_new_x, new_z)
new_X = new_X*36
new_Z = new_Z*60
u_water, w_water = da.turbulent_flowfield(xScal1, uScal1, numScal, new_u, new_w, l32)
fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
GetMaxEnergies = KEAI1[-1, :][KEAI1[-1, :] != 0]
ColorScheme = np.log10(GetMaxEnergies)  ## define color scheme to be consistent for every plot
ColorNumbers = ColorScheme[np.logical_not(np.isnan(ColorScheme))] 
ColorMax = np.ceil(np.max(ColorNumbers))
my_colors = cm.get_cmap('YlGn_r', 256)
axs.set_xlim(0, 6)
for i in range(len(Diam1)):
    GS = np.ones_like(XAtImpact1)*Diam1[i]*10
    KEAI1[i, :][KEAI1[i, :]==0] = np.nan
    findColors = (np.log10(KEAI1[i, :]))/ColorMax
    axs.scatter(XAtImpact1[i, :], GS[i, :], c = my_colors(findColors), zorder = 0)
plt.fill_between(x01, z01*8, 0, alpha = 1, color = 'grey')
#plt.quiver(new_X[::20, ::20], new_Z[::20, ::20], u_water[::20, ::20], w_water[::20, ::20])
plt.contourf(new_X, new_Z, w_water, alpha = 0.5, vmin = -25, vmax = 25, cmap='seismic')
fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                    wspace=0.4, hspace=0.1)
plt.title('Particle impacts at each location by grainsize on 1 cm Scallops')
cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
cb2_ax = fig.add_axes([0.93, 0.1, 0.02, 0.8])
norm = colors.Normalize(vmin = 0, vmax = ColorMax)
norm2 = colors.Normalize(vmin = -25, vmax = 25)
plt.colorbar(cm.ScalarMappable(norm = norm, cmap='YlGn_r'), cax = cb_ax)
plt.colorbar(cm.ScalarMappable(norm = norm2, cmap='seismic'), cax = cb2_ax)
cb_ax.set_ylabel('log10 of Kinetic energy of impact (ergs)')
cb2_ax.set_ylabel('vertical water velocity (cm/s)')
axs.set_xlabel('x (cm)')
axs.set_ylabel('particle grainsize (mm)')
plt.show()
  
l32 = 2.5
dx0 = 0.05/l32
numScal = 6
xScal2 = np.arange(0, numScal+dx0,dx0)  #x-array for scallop field
uScal2 = np.arange(0,1+dx0,dx0)  #x-array for a single scallop
x02_5, z02_5 = da.scallop_array(xScal2, uScal2, numScal, l32)   #initial scallop profile, dimensions in centimeters
z02_5 = z02_5 - np.min(z02_5)
nx = l32*20 + 1
ny = l32*20 + 1
nnx = l32*20*numScal + 1
new_x = np.linspace(0, l32/numScal, int(nx))
new_new_x = np.linspace(0, l32/numScal, int(nnx))
new_z = np.linspace(0, l32/numScal, int(ny))
new_u = np.zeros((int(ny), int(nx)))
new_w = np.zeros((int(ny), int(nx)))
new_X, new_Z = np.meshgrid(new_new_x, new_z)
new_X = new_X*36
new_Z = new_Z*20
u_water, w_water = da.turbulent_flowfield(xScal2, uScal2, numScal, new_u, new_w, l32)
fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
GetMaxEnergies = KEAI2_5[-1, :][KEAI2_5[-1, :] != 0]
ColorScheme = np.log10(GetMaxEnergies)  ## define color scheme to be consistent for every plot
ColorNumbers = ColorScheme[np.logical_not(np.isnan(ColorScheme))] 
ColorMax = np.ceil(np.max(ColorNumbers))
my_colors = cm.get_cmap('YlGn_r', 256)
axs.set_xlim(0, 6*2.5)
for i in range(len(Diam2_5)):
    GS = np.ones_like(XAtImpact2_5)*Diam2_5[i]*10
    KEAI2_5[i, :][KEAI2_5[i, :]==0] = np.nan
    findColors = (np.log10(KEAI2_5[i, :]))/ColorMax
    axs.scatter(XAtImpact2_5[i, :], GS[i, :], c = my_colors(findColors), zorder = 0)
plt.fill_between(x02_5, z02_5*4, 0, alpha = 1, color = 'grey')
#plt.quiver(new_X[::20, ::20], new_Z[::20, ::20], u_water[::20, ::20], w_water[::20, ::20])
plt.contourf(new_X, new_Z, w_water, alpha = 0.5, vmin = -15, vmax = 15, cmap='seismic')
fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                    wspace=0.4, hspace=0.1)
plt.title('Particle impacts at each location by grainsize on 2.5 cm Scallops')
cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
cb2_ax = fig.add_axes([0.93, 0.1, 0.02, 0.8])
norm = colors.Normalize(vmin = 0, vmax = ColorMax)
norm2 = colors.Normalize(vmin = -15, vmax = 15)
plt.colorbar(cm.ScalarMappable(norm = norm, cmap='YlGn_r'), cax = cb_ax)
plt.colorbar(cm.ScalarMappable(norm = norm2, cmap='seismic'), cax = cb2_ax)
cb_ax.set_ylabel('log10 of Kinetic energy of impact (ergs)')
cb2_ax.set_ylabel('vertical water velocity (cm/s)')
axs.set_xlabel('x (cm)')
axs.set_ylabel('particle grainsize (mm)')
plt.show()
 
  
l32 = 5
dx0 = 0.05/l32
numScal = 4
xScal5 = np.arange(0, numScal+dx0,dx0)  #x-array for scallop field
uScal5 = np.arange(0,1+dx0,dx0)  #x-array for a single scallop
x05, z05 = da.scallop_array(xScal5, uScal5, numScal, l32)   #initial scallop profile, dimensions in centimeters
z05 = z05 - np.min(z05)
nx = l32*20 + 1
ny = l32*20 + 1
nnx = l32*20*numScal + 1
new_x = np.linspace(0, l32/numScal, int(nx))
new_new_x = np.linspace(0, l32/numScal, int(nnx))
new_z = np.linspace(0, l32/numScal, int(ny))
new_u = np.zeros((int(ny), int(nx)))
new_w = np.zeros((int(ny), int(nx)))
new_X, new_Z = np.meshgrid(new_new_x, new_z)
new_X = new_X*16
new_Z = new_Z*10
u_water, w_water = da.turbulent_flowfield(xScal5, uScal5, numScal, new_u, new_w, l32)
fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
GetMaxEnergies = KEAI5[-1, :][KEAI5[-1, :] != 0]
ColorScheme = np.log10(GetMaxEnergies)  ## define color scheme to be consistent for every plot
ColorNumbers = ColorScheme[np.logical_not(np.isnan(ColorScheme))] 
ColorMax = np.ceil(np.max(ColorNumbers))
my_colors = cm.get_cmap('YlGn_r', 256)
axs.set_xlim(0, 4*5)
for i in range(len(Diam5)):
    GS = np.ones_like(XAtImpact5)*Diam5[i]*10
    KEAI5[i, :][KEAI5[i, :]==0] = np.nan
    findColors = (np.log10(KEAI5[i, :]))/ColorMax
    axs.scatter(XAtImpact5[i, :], GS[i, :], c = my_colors(findColors), zorder = 0)
plt.fill_between(x05, z05*2, 0, alpha = 1, color = 'grey')
#plt.quiver(new_X[::20, ::20], new_Z[::20, ::20], u_water[::20, ::20], w_water[::20, ::20])
plt.contourf(new_X, new_Z, w_water, alpha = 0.5, vmin = -15, vmax = 15, cmap='seismic')
fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                    wspace=0.4, hspace=0.1)
plt.title('Particle impacts at each location by grainsize on 5 cm Scallops')
cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
cb2_ax = fig.add_axes([0.93, 0.1, 0.02, 0.8])
norm = colors.Normalize(vmin = 0, vmax = ColorMax)
norm2 = colors.Normalize(vmin = -15, vmax = 15)
plt.colorbar(cm.ScalarMappable(norm = norm, cmap='YlGn_r'), cax = cb_ax)
plt.colorbar(cm.ScalarMappable(norm = norm2, cmap='seismic'), cax = cb2_ax)
cb_ax.set_ylabel('log10 of Kinetic energy of impact (ergs)')
cb2_ax.set_ylabel('vertical water velocity (cm/s)')
axs.set_xlabel('x (cm)')
axs.set_ylabel('particle grainsize (mm)')
plt.show()


l32 = 10
dx0 = 0.05/l32
numScal = 4
xScal10 = np.arange(0, numScal+dx0,dx0)  #x-array for scallop field
uScal10 = np.arange(0,1+dx0,dx0)  #x-array for a single scallop
x010, z010 = da.scallop_array(xScal10, uScal10, numScal, l32)   #initial scallop profile, dimensions in centimeters
z010 = z010 - np.min(z010)
nx = l32*20 + 1
ny = l32*20 + 1
nnx = l32*20*numScal + 1
new_x = np.linspace(0, l32/numScal, int(nx))
new_new_x = np.linspace(0, l32/numScal, int(nnx))
new_z = np.linspace(0, l32/numScal, int(ny))
new_u = np.zeros((int(ny), int(nx)))
new_w = np.zeros((int(ny), int(nx)))
new_X, new_Z = np.meshgrid(new_new_x, new_z)
new_X = new_X*16
new_Z = new_Z*5
u_water, w_water = da.turbulent_flowfield(xScal10, uScal10, numScal, new_u, new_w, l32)
fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
GetMaxEnergies = KEAI10[-1, :][KEAI10[-1, :] != 0]
ColorScheme = np.log10(GetMaxEnergies)  ## define color scheme to be consistent for every plot
ColorNumbers = ColorScheme[np.logical_not(np.isnan(ColorScheme))] 
ColorMax = np.ceil(np.max(ColorNumbers))
my_colors = cm.get_cmap('YlGn_r', 256)
axs.set_xlim(0, 4*10)
for i in range(len(Diam10)):
    GS = np.ones_like(XAtImpact10)*Diam10[i]*10
    KEAI10[i, :][KEAI10[i, :]==0] = np.nan
    findColors = (np.log10(KEAI10[i, :]))/ColorMax
    axs.scatter(XAtImpact10[i, :], GS[i, :], c = my_colors(findColors), zorder = 0)
plt.fill_between(x010, z010, 0, alpha = 1, color = 'grey')
#plt.quiver(new_X[::20, ::20], new_Z[::20, ::20], u_water[::20, ::20], w_water[::20, ::20])
plt.contourf(new_X, new_Z, w_water, alpha = 0.5, vmin = -10, vmax = 10, cmap='seismic')
fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                    wspace=0.4, hspace=0.1)
plt.title('Particle impacts at each location by grainsize on 10 cm Scallops')
cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
cb2_ax = fig.add_axes([0.93, 0.1, 0.02, 0.8])
norm = colors.Normalize(vmin = 0, vmax = ColorMax)
norm2 = colors.Normalize(vmin = -10, vmax = 10)
plt.colorbar(cm.ScalarMappable(norm = norm, cmap='YlGn_r'), cax = cb_ax)
plt.colorbar(cm.ScalarMappable(norm = norm2, cmap='seismic'), cax = cb2_ax)
cb_ax.set_ylabel('log10 of Kinetic energy of impact (ergs)')
cb2_ax.set_ylabel('vertical water velocity (cm/s)')
axs.set_xlabel('x (cm)')
axs.set_ylabel('particle grainsize (mm)')
plt.show()



