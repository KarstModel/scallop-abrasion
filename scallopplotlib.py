# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 06:20:38 2021

@author: rachelbosch

This is a post-processing library to accompany DKARST.py

Data referenced in these functions:
# =============================================================================
# Initial Conditions array:
#   initial conditions saved for each particle in simulation      
#       shape = (n, numPrtkl, 5) 
#           n = number of grain sizes in diameter array
#           numPrtkl = number of particles simulated for each grain size
#           0 = x, 1 = z, 2 = u, 3 = w, D = particle diameter 
# =============================================================================
    
# =============================================================================
# Impact Data array:
#   data collected every time a particle impacts the bedrock surface
#       shape = (n, 100000, 9)
#           n = number of grain sizes in diameter array
#           100000 pre-allocated to be greater than number of time-steps
#           0 = time, 1 = x, 2 = z, 3 = u, 4 = w, 5 = D, 6 = |Vel|, 7 = KE, 8 = particle ID,
#               links to numPartkl in Initial Conditions array
# =============================================================================

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

def __init__(self):
    pass

def trajectory_figures(scallop_length, number_of_scallops, diameter, grain_type, scallop_x, scallop_z, loc_data):
    fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,4))    
    #axs.set_aspect('equal')
    #axs.set_xlim(0, 40)
    axs.set_ylim(0,2)
    axs.plot (scallop_x, scallop_z, 'grey')
    ld = np.array(loc_data, dtype = object)
    
    for p in ld[(np.random.randint(len(ld)-1,size=50)).astype(int)]:
        axs.plot(p[p[:,1]>0, 1], p[p[:,1]>0,2], 2)
        
        
    plt.fill_between(scallop_x, scallop_z, 0, alpha = 1, color = 'grey', zorder=101)
    axs.set_ylabel('z (cm)')
    axs.set_xlabel('x (cm)')
    axs.set_title('Trajectories of randomly selected ' + str(round(diameter*10, 3)) + ' mm '+ grain_type +' on ' +str(scallop_length)+ ' cm floor scallops.')
    return fig, axs


def average_velocities_plot_fit_to_Dietrich(rho_sediment, rho_fluid, diameter_array, scallop_length, VelocityAtImpact, numPrtkl):
    fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
    g = 981 # cm*s^-2
    nu = 0.01307  # g*cm^-1*s^-1
    AvgVelocity = np.zeros_like(diameter_array)
    for j in range(len(AvgVelocity)):
        AvgVelocity[j] = np.average(VelocityAtImpact[j,(np.logical_not(np.isnan(VelocityAtImpact[j, :])))][VelocityAtImpact[j,(np.logical_not(np.isnan(VelocityAtImpact[j, :])))]<0])
    D_star = ((rho_sediment-rho_fluid)*g*(diameter_array)**3)/(rho_fluid*nu)
    W_star = -(rho_fluid*AvgVelocity**3)/((rho_sediment-rho_fluid)*g*nu)
    W_star_Dietrich = 1.71 * 10**-4 * D_star**2
    axs.scatter(diameter_array, W_star, label = 'simulated impact velocity')
    axs.plot(diameter_array, W_star_Dietrich, c = 'g', label = 'settling velocity (Dietrich, 1982)')

    def settling_velocity(D_star, r, s):
        return r * D_star**s
    
    pars, cov = curve_fit(f=settling_velocity, xdata=D_star, ydata=W_star, p0=[1.71 * 10**-4, 2], bounds=(-np.inf, np.inf))
    # Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
    stdevs = np.sqrt(np.diag(cov))
    # Calculate the residuals
    res = W_star - settling_velocity(D_star, *pars)
    axs.plot(diameter_array, settling_velocity(D_star, *pars), linestyle='--', linewidth=2, color='black', label = 'fitted to Equation (13), with r= '+str(round(pars[0], 2))+' and s= '+str(round(pars[1], 2)))
    
    plt.legend()
    plt.semilogy()
    axs.set_xlabel('dimensionless diameter')
    axs.set_ylabel('dimensionless settling velocity') 
    axs.set_title('Particle velocities over '+str(scallop_length)+' cm scallops, with ' +str(numPrtkl)+ ' particles')
    
    return pars, stdevs, res, fig, axs

def average_velocities_mult_scallop_lengths(rho_sediment, rho_fluid):
    fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
    Diam1 = genfromtxt('./outputs/diam1turbulent.csv', delimiter=',')
    Vels1 = genfromtxt('./outputs/VelocityAvg1turbulent.csv', delimiter=',')
    Diam2_5 = genfromtxt('./outputs/diam2.5turbulent.csv', delimiter=',')
    Vels2_5 = genfromtxt('./outputs/VelocityAvg2.5turbulent.csv', delimiter=',')
    Diam5 = genfromtxt('./outputs/diam5turbulent.csv', delimiter=',')
    Vels5 = genfromtxt('./outputs/VelocityAvg5turbulent.csv', delimiter=',')
    Diam10 = genfromtxt('./outputs/diam10turbulent.csv', delimiter=',')
    Vels10 = genfromtxt('./outputs/VelocityAvg10turbulent.csv', delimiter=',')
    
    not_nan_idx_1 = np.where(~np.isnan(Vels1))
    diam1 = Diam1[not_nan_idx_1]
    vels1 =Vels1[not_nan_idx_1]
    not_nan_idx_2_5 = np.where(~np.isnan(Vels2_5))
    diam2_5 = Diam2_5[not_nan_idx_2_5]
    vels2_5 = Vels2_5[not_nan_idx_2_5]
    not_nan_idx_5 = np.where(~np.isnan(Vels5))
    diam5 = Diam5[not_nan_idx_5]
    vels5 =Vels5[not_nan_idx_5]
    not_nan_idx_10 = np.where(~np.isnan(Vels10))
    diam10 = Diam10[not_nan_idx_10]
    vels10 =Vels10[not_nan_idx_10]
    
    w_s = da.settling_velocity(rho_sediment, rho_fluid, diam10)
    axs.plot(diam10*10, -w_s, c = 'g', label = 'settling velocity (Ferguson and Church, 2004)')
    axs.scatter((diam1*10), vels1, label = 'simulated impact velocity on 1.0 cm scallops')
    axs.scatter((diam2_5*10), vels2_5, label = 'simulated impact velocity on 2.5 cm scallops')
    axs.scatter((diam5*10), vels5, label = 'simulated impact velocity on 5.0 cm scallops')
    axs.scatter((diam10*10), vels10, label = 'simulated impact velocity on 5.0 cm scallops')
        
    plt.legend()
    plt.semilogx()
    axs.set_xlabel('particle grainsize (mm)')
    axs.set_ylabel('velocity (cm/s)') 
    axs.set_title('Particle velocities at impact upon scalloped floors')
    
    return fig, axs

##combine velocity data sets and fit a curve of form Dietrich (1982)
def velocity_curve_fitting(rho_sediment, rho_fluid):
    fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
    Diam1 = genfromtxt('./outputs/diam1turbulent.csv', delimiter=',')
    Vels1 = genfromtxt('./outputs/VelocityAvg1turbulent.csv', delimiter=',')
    Diam2_5 = genfromtxt('./outputs/diam2.5turbulent.csv', delimiter=',')
    Vels2_5 = genfromtxt('./outputs/VelocityAvg2.5turbulent.csv', delimiter=',')
    Diam5 = genfromtxt('./outputs/diam5turbulent.csv', delimiter=',')
    Vels5 = genfromtxt('./outputs/VelocityAvg5turbulent.csv', delimiter=',')
    Diam10 = genfromtxt('./outputs/diam10turbulent.csv', delimiter=',')
    Vels10 = genfromtxt('./outputs/VelocityAvg10turbulent.csv', delimiter=',')
    
    not_nan_idx_1 = np.where(~np.isnan(Vels1))
    diam1 = Diam1[not_nan_idx_1]
    vels1 =Vels1[not_nan_idx_1]
    not_nan_idx_2_5 = np.where(~np.isnan(Vels2_5))
    diam2_5 = Diam2_5[not_nan_idx_2_5]
    vels2_5 = Vels2_5[not_nan_idx_2_5]
    not_nan_idx_5 = np.where(~np.isnan(Vels5))
    diam5 = Diam5[not_nan_idx_5]
    vels5 =Vels5[not_nan_idx_5]
    not_nan_idx_10 = np.where(~np.isnan(Vels10))
    diam10 = Diam10[not_nan_idx_10]
    vels10 =Vels10[not_nan_idx_10]
    
    diameters = diam1
    diameters.append(diam2_5)
    diameters.append(diam5)
    diameters.append(diam10)
    velocities = vels1
    velocities.append(vels2_5)
    velocities.append(vels5)
    velocities.append(vels10)
    
    g = -981 # cm*s^-2
    nu = 0.01307  # g*cm^-1*s^-1
    D_star_con = ((rho_sediment-rho_fluid)*g*(diameters)**3)/(rho_fluid*nu)
    W_star_con = (rho_fluid*velocities**3)/((rho_sediment-rho_fluid)*g*nu)
    W_star_Dietrich_con = 1.71 * 10**-4 * D_star_con**2
    axs.scatter(D_star_con, W_star_con, label = 'simulated impact velocity')
    axs.plot(D_star_con, W_star_Dietrich_con, c = 'g', label = 'settling velocity (Dietrich, 1982)')

    
    def settling_velocity(D_star, r, s):
        return r * D_star**s
    
    pars, cov = curve_fit(f=settling_velocity, xdata=D_star_con, ydata=W_star_con, p0=[1.71 * 10**-4, 2], bounds=(-np.inf, np.inf))
    # Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
    stdevs = np.sqrt(np.diag(cov))
    # Calculate the residuals
    res = W_star_con - settling_velocity(D_star_con, *pars)
    axs.plot(D_star_con, settling_velocity(D_star_con, *pars), linestyle='--', linewidth=2, color='black', label = 'fitted to Equation (13), with r= '+str(round(pars[0], 2))+' and s= '+str(round(pars[1], 2)))

    pars, cov = curve_fit(f=settling_velocity, xdata=diameters, ydata=velocities, p0=[24, 1.2], bounds=(-np.inf, np.inf))
    # Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
    stdevs = np.sqrt(np.diag(cov))
    # Calculate the residuals
    res = velocities - settling_velocity(D_star_con, *pars)
    axs.plot(D_star_con, settling_velocity(D_star_con, *pars), linestyle='--', linewidth=2, color='black', label = 'fitted to Equation (13), with C_1= '+str(round(pars[0], 2))+' and C_2= '+str(round(pars[1], 2)))
    
    w_s = da.settling_velocity(rho_sediment, rho_fluid, diameters)
    axs.plot(diameters*10, -w_s, c = 'g', label = 'settling velocity (Ferguson and Church, 2004)')
    axs.scatter((diameters*10), velocities, label = 'simulated impact velocity on all scallop lengths')
        
    plt.legend()
    plt.semilogx()
    axs.set_xlabel('particle grainsize (mm)')
    axs.set_ylabel('velocity (cm/s)') 
    axs.set_title('Particle velocities at impact upon scalloped floors')
    
    return pars, stdevs, res, fig, axs





# # =============================================================================
# # Two different ways to get erosion rate due to abrasion:
#     # 1. Work-energy theorem: work done on limestone = change in kinetic energy of sandstone grain
#     # 2. Erosion rate due to abrasion expression from Lamb et al. (2008). 
# # =============================================================================
    ### 1. WORK-ENERGY THEOREM (these results are very much the wrong order of magnitude)
def work_energy_theorem_plot(EnergyAtImpact, rho_bedrock, diameter_array, x_array):
    CoR = 0.4    #coefficient of restitution for sandstone impinging on limestone
    WorkDoneAtImpact = EnergyAtImpact * (1 - CoR)   #work done on limestone, work-energy theorem
    
    for j in range(len(diameter_array)):
        fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
        
        mlr = (np.pi * rho_bedrock * diameter_array[j]**3)/6    # mass limestone removed
        dv_dt = WorkDoneAtImpact/(mlr * diameter_array[j])
        E_we = dv_dt * rho_bedrock * 24 * 3600  # convert cm*s**-1 to g*cm**-2*day*-1
        
        if diameter_array[j] < 0.0063:
            grain = 'silt'
        elif diameter_array[j] >= 0.0063 and diameter_array[j] < 0.2:
            grain = 'sand'
        elif diameter_array[j] >= 0.2:
            grain = 'gravel'
        axs.set_xlim(15, 25)
    
        #axs.set_aspect('equal')
        axs.plot(x_array, (E_we[j]))
       
        
        axs.set_xlim(15, 25)
        axs.set_ylabel('Erosion rate (g/(cm^2*day))')
        axs.set_xlabel('x (cm)')
        
        axs.set_title('Erosion rate by work-energy theorem, ' + str(round(diameter_array[j]*10, 3)) + ' mm '+ grain +' on floor scallops')
    
    plt.show()
    return fig, axs

# ### 2. BUFFALO WILD WINGS AND WECK
def lamb_erosion_rates_plot(diameter_array, ErosionAtImpact, x_array, scallop_length):
    fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
    for j in range(len(diameter_array)):
        E_bw3 = -(ErosionAtImpact[j, :]*10*3600*24*365.25)  
        for k in range(len(E_bw3)):                                  
            if E_bw3[k] < 0:
                E_bw3[k] = 0
        axs.scatter(x_array, E_bw3, label=str(round(diameter_array[j]*10, 1)) + ' mm')
    axs.semilogy()       
    axs.set_xlim(15, 25)
    axs.set_ylabel('Erosion rate (mm/yr)')
    axs.set_xlabel('x (cm)')   
    axs.set_title('Erosion rates on '+str(scallop_length)+' cm floor scallops')
    plt.legend()
    
    return fig, axs

####Total abrasion Over One Scallop
def abrasion_one_scallop_plot(diameter_array, NormErosionAvg, NumberOfImpactsByGS, scallop_length):
    fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
    
    axs.scatter((diameter_array*10), (NormErosionAvg), label = 'simulated abrasional erosion on '+str(scallop_length)+' cm scallops')
    plt.semilogx()
    plt.legend(loc = 'center left')
    axs.set_title('Abrasion Rate Normalized by Number of Impacts (>=5)')
    axs.set_xlabel('particle grainsize (mm)')
    axs.set_ylabel('abrasional erosion rate (mm/yr)')
    axs.grid(True, which = 'both', axis = 'both')
    
    return fig, axs

def abrasion_one_scallop_plot_mult_scallop_lengths():
    fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
    Diam1 = genfromtxt('./outputs/diam1turbulent.csv', delimiter=',')
    NEA1 = genfromtxt('./outputs/NormErosionAvg1turbulent.csv', delimiter=',')
    Diam2_5 = genfromtxt('./outputs/diam2.5turbulent.csv', delimiter=',')
    NEA2_5 = genfromtxt('./outputs/NormErosionAv2.51turbulent.csv', delimiter=',')
    Diam5 = genfromtxt('./outputs/diam5turbulent.csv', delimiter=',')
    NEA5 = genfromtxt('./outputs/NormErosionAvg5turbulent.csv', delimiter=',')
    Diam10 = genfromtxt('./outputs/diam10turbulent.csv', delimiter=',')
    NEA10 = genfromtxt('./outputs/NormErosionAvg10turbulent.csv', delimiter=',')
    
    axs.scatter((Diam1*10), (NEA1), label = 'simulated abrasional erosion on 1.0 cm scallops')
    axs.scatter((Diam2_5*10), (NEA2_5), label = 'simulated abrasional erosion on 2.5 cm scallops')
    axs.scatter((Diam5*10), (NEA5), label = 'simulated abrasional erosion on 5.0 cm scallops')
    axs.scatter((Diam10*10), (NEA10), label = 'simulated abrasional erosion on 10 cm scallops')
    plt.semilogx()
    plt.legend(loc = 'center left')
    axs.set_title('Abrasion Rate Normalized by Number of Impacts (>=5)')
    axs.set_xlabel('particle grainsize (mm)')
    axs.set_ylabel('abrasional erosion rate (mm/yr)')
    axs.grid(True, which = 'both', axis = 'both')
    
    return fig, axs
    
####Total abrasion Over 5 cm Scallop with dissolution comparison
def abrasion_and_dissolution_plot(x_array, diam, ErosionAtImpact): 
    fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
    # Diam5 = genfromtxt('./outputs2/diam5turbulent2021-02-08.csv', delimiter=',')
    # EAI5 = genfromtxt('./outputs2/ErosionAtImpact5turbulent2021-02-08.csv', delimiter=',')
    Diam5 = diam
    EAI5 = ErosionAtImpact
    ES5 = np.zeros_like(Diam5)
    NEA5 = np.zeros_like(Diam5)
    NOI5 = np.zeros_like(Diam5)
    for s in range(len(Diam5)):
        ES5[s] = -np.sum(EAI5[s, int(len(x_array)/12):int(len(x_array))][EAI5[s, int(len(x_array)/12):int(len(x_array))]<0]*1000*36*24*365.25)
        NP5 = len(EAI5[s, int(len(x_array)/12):int(len(x_array))][EAI5[s, int(len(x_array)/12):int(len(x_array))]<0])
        NOI5[s] = NP5
        if NP5 > 0:
            NEA5[s] = ES5[s]/NP5
        else:
            NEA5[s] = 0
    axs.scatter((Diam5*10), (NEA5), label = 'simulated abrasional erosion')
    diss_min = 5.66    #minimum dissolution rate (mm/yr) (Grm et al., 2017)
    diss_max = 12.175  #maximum dissolution rate (mm/yr) (Hammer et al., 2011)
    x = np.linspace(0.9, 5)
    plt.fill_between(x, diss_min, diss_max, alpha = 0.6, color = 'r')
    
    plt.semilogx()
    plt.legend(loc = 'center left')
    #plt.semilogy()
    # axs.set_ylim(30,200)
    axs.set_xlim(.9,30)
    axs.set_title('Abrasion Rate Normalized by Number of Impacts on 5 cm Scallops (>=5)')
    axs.set_xlabel('particle grainsize (mm)')
    axs.set_ylabel('abrasional erosion rate (mm/yr)')
    axs.grid(True, which = 'both', axis = 'both')
    
    #select the x-range for the zoomed region
    x1 = 1.5
    x2 = 4.5
    
    # select y-range for zoomed region
    y1 = 0
    y2 = 50
    
    # Make the zoom-in plot:
    axins = zoomed_inset_axes(axs, 4, bbox_to_anchor=(0,0), loc = 'upper left')
    axins.scatter((Diam5*10), (NEA5))
    axins.fill_between(x, diss_min, diss_max, alpha = 0.6, color = 'r', label = 'dissolutional range')
    #axins.plot((Diam5*10), 10**y, 'r', color = 'g')
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    plt.xticks(visible=True)
    plt.yticks(visible=True)
    axins.legend(loc = 'upper center')
    mark_inset(axs, axins, loc1=2, loc2=1, fc="none", ec="0.5")
    return fig, axs, axins

def abrasion_and_dissolution_plot_2(x_array, diam, NormErosionAvg, scallop_length):
    cb_max = 0.02
    cb_tiny = 4 * 10**-5
    cb_old = 0.01
    cb = np.linspace(cb_tiny, cb_max, 5)
    
    fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
    #Diam5 = genfromtxt('./outputs2/diam5turbulent2021-02-09.csv', delimiter=',')
    Diam5 = diam
    #NEA5_old = genfromtxt('./outputs2/NormErosionAvg5turbulent2021-02-09.csv', delimiter=',')
    NEA5_old = NormErosionAvg
    for i in range(len(cb)):
        NEA5_new = NEA5_old * cb[i]/cb_old
        axs.scatter((Diam5*10), (NEA5_new), label = 'bedload concentration = '+str(round(cb[i], 5)))
    
    diss_min = 5.66 * 5 / scallop_length    #minimum dissolution rate (mm/yr) (Grm et al., 2017) over 5 cm scallops, scaling from Curl (1966)
    diss_max = 13.472 * 5 / scallop_length #maximum dissolution rate (mm/yr) (Hammer et al., 2011) over 5 cm scallops, scaling from Curl (1966)
    x = np.linspace(0.9, 25)
    plt.fill_between(x, diss_min, diss_max, alpha = 0.4, color = 'gray', label = 'dissolutional range')
    
    plt.semilogx()
    plt.legend(loc = 'upper left')
    axs.set_xlim(.9,30)
    axs.set_title('Abrasion Rate Normalized by Number of Impacts on '+str(scallop_length)+' cm Scallops')
    axs.set_xlabel('particle grainsize (mm)')
    axs.set_ylabel('abrasional erosion rate (mm/yr)')
    axs.grid(True, which = 'both', axis = 'both')
    
    
    return fig, axs

def abrasion_by_slope(scallop_local_slope, ErosionAtImpact, diameter_array, scallop_length):
    fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
    cb = 0.01
    for i in range(len(ErosionAtImpact)):
        axs.scatter(scallop_local_slope, -ErosionAtImpact[i, :]*1000*36*24*365.25*cb/0.01, label = 'particle diameter = '+str(round(diameter_array[i], 3)))
    axs.set_title('Local slope of scallop profile versus surface abrasion on '+str(scallop_length)+' cm scallops, cb = '+str(cb))
    plt.legend(loc = 'upper right')
    axs.set_xlabel('scallop profile local slope (dz/dx)')
    axs.set_ylabel('abrasional erosion rate (mm/yr)')    
    return fig, axs
    
    

    
def number_of_impacts_plot(diameter_array, NumberOfImpactsByGS, scallop_length, x_array):
    fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
    Diam5 = genfromtxt('diam5.csv', delimiter=',')
    EAI5 = genfromtxt('ErosionAtImpact5.csv', delimiter=',')
    ES5 = np.zeros_like(Diam5)
    NEA5 = np.zeros_like(Diam5)
    NOI5 = np.zeros_like(Diam5)
    for s in range(len(Diam5)):
        ES5[s] = -np.sum(EAI5[s, int(len(x_array)/12):int(len(x_array))][EAI5[s, int(len(x_array)/12):int(len(x_array))]<0]*1000*36*24*365.25)
        NP5 = len(EAI5[s, int(len(x_array)/12):int(len(x_array))][EAI5[s, int(len(x_array)/12):int(len(x_array))]<0])
        NOI5[s] = NP5
        if NP5 > 0:
            NEA5[s] = ES5[s]/NP5
        else:
            NEA5[s] = 0
    axs.scatter(diameter_array*10, NumberOfImpactsByGS, label = 'impacts on '+str(scallop_length)+' cm scallop')
    axs.scatter(Diam5*10, NOI5, label = 'impacts on 5 cm scallop', zorder = 0)
    plt.title('Number of particle impacts on one scallop')
    axs.set_xlabel('particle grainsize (mm)')
    axs.set_ylabel('number of impacts')
    plt.semilogx()
    plt.legend(loc='upper left')
    axs.grid(True, which = 'both', axis = 'x')
    
    return fig, axs
  
def number_of_impacts_at_loc_plot(diameter_array, scallop_x, scallop_z, scallop_length, All_Impacts, initial_conditions, numScal):
    fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
    GetMaxEnergies = All_Impacts[:, :, 7][All_Impacts[:, :, 7] != 0]
    ColorScheme = np.log10(GetMaxEnergies)  ## define color scheme to be consistent for every plot
    ColorNumbers = ColorScheme[np.logical_not(np.isnan(ColorScheme))] 
    ColorMax = np.ceil(np.max(ColorNumbers))
    my_colors = cm.get_cmap('gist_rainbow_r', 256)
    axs.set_xlim(0, numScal*scallop_length)
    for i in range(len(diameter_array)):
        GS = All_Impacts[i, :, 5][All_Impacts[i, :, 7] != 0]
        initial_z_idxs = np.array(All_Impacts[i, :, 8][All_Impacts[i, :, 7] != 0], dtype = int)
        impact_x = All_Impacts[i, :, 1][All_Impacts[i, :, 7] != 0]
        #All_Impacts[i, :, 7][All_Impacts[i, :, 7] == 0] = np.nan
        findColors = (np.log10(All_Impacts[i, :, 7][All_Impacts[i, :, 7] != 0]))/ColorMax
        #axs.scatter(XAtImpact[i, :], GS[i, :], c = my_colors(findColors))
        axs.scatter(impact_x, initial_conditions[i, initial_z_idxs, 1] , c = my_colors(findColors), s = 50 * GS)
    plt.fill_between(scallop_x, scallop_z/20, 0, alpha = 1, color = 'grey')
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                        wspace=0.4, hspace=0.1)
    #plt.title('Particle impacts at each location by grainsize on '+str(scallop_length)+' cm Scallops')
    plt.title('Particle impacts at each location by fall height on '+str(scallop_length)+' cm Scallops')
    cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
    norm = colors.Normalize(vmin = 0, vmax = ColorMax)
    plt.colorbar(cm.ScalarMappable(norm = norm, cmap='gist_rainbow_r'), cax = cb_ax)
    cb_ax.set_ylabel('log10 of Kinetic energy of impact (ergs)')
    axs.set_xlabel('x (cm)')
    #axs.set_ylabel('particle grainsize (mm)')
    axs.set_ylabel('fall height (cm)')

    
    return fig, axs
  
def Gearys_test(NormErosionAvg):   
    TSS = 0 #total sum of squares
    sum_abs = 0
    for s in range(len(NormErosionAvg[NormErosionAvg > 0])):
        square = (np.log10(NormErosionAvg[NormErosionAvg > 0][s]) - np.average(np.log10(NormErosionAvg[NormErosionAvg > 0])))**2
        diff_abs = np.abs(np.log10(NormErosionAvg[NormErosionAvg > 0][s]) - np.average(np.log10(NormErosionAvg[NormErosionAvg > 0])))
        TSS = TSS + square
        sum_abs = sum_abs +diff_abs
    sigma=np.sqrt(TSS/len(NormErosionAvg[NormErosionAvg > 0]))
    numerator= (np.sqrt(np.pi/2))*sum_abs/(len(NormErosionAvg[NormErosionAvg > 0]))
    Gearys_test = numerator/sigma          #### confirm that data is log-normally distributed
    
    return Gearys_test

def impact_locations_plot(EnergyAtImpact, diameter_array, x_array, scallop_profile, XAtImpact, ZAtImpact, uScal, scallop_length, number_of_scallops):   
    GetMaxEnergies = EnergyAtImpact[-1, :][EnergyAtImpact[-1, :] != 0]
    ColorScheme = np.log10(GetMaxEnergies)  ## define color scheme to be consistent for every plot
    ColorNumbers = ColorScheme[np.logical_not(np.isnan(ColorScheme))] 
    ColorMax = np.ceil(np.max(ColorNumbers))
    KEAvg = np.zeros_like(diameter_array)
    my_colors = cm.get_cmap('Paired', 256)
    fig, axs = plt.subplots(nrows = len(diameter_array), ncols = 1, sharex=True, figsize = (11, 17))
    for j in range(len(diameter_array)):
        #axs[j].set_xlim(scallop_length*number_of_scallops/2, (scallop_length*number_of_scallops))
        axs[j].set_ylim(-0.5, 1.5)
        axs[j].set_aspect('equal')
        axs[j].plot(x_array, scallop_profile, 'grey')
        EnergyAtImpact[j, :][EnergyAtImpact[j, :]==0] = np.nan
        findColors = (np.log10(EnergyAtImpact[j, :]))/ColorMax 
        axs[j].scatter(XAtImpact[j, :], ZAtImpact[j, :], c = my_colors(findColors) )
        KEAvg = np.average(np.logical_not(np.isnan(EnergyAtImpact[j, :])))
        axs[j].set_ylabel('z (cm)') 
        axs[j].set_title('D = ' +str(round(diameter_array[j]*10, 2)) + ' mm                     avg.KE = ' + str(round(KEAvg, 2)) + ' ergs')
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                        wspace=0.4, hspace=0.1)
    cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
    norm = colors.Normalize(vmin = 0, vmax = ColorMax)
    plt.colorbar(cm.ScalarMappable(norm = norm, cmap='Paired'), cax = cb_ax)
    cb_ax.set_ylabel('log10 of Kinetic energy of impact (ergs)')
    axs[-1].set_xlabel('x (cm)')
    return fig, axs

def seperate_impact_locations_plot(EnergyAtImpact, diameter_array, x_array, scallop_profile, XAtImpact, ZAtImpact, uScal, scallop_length, number_of_scallops):   
    
    GetMaxEnergies = EnergyAtImpact[-1, :][EnergyAtImpact[-1, :] != 0]
    ColorScheme = np.log10(GetMaxEnergies)  ## define color scheme to be consistent for every plot
    ColorNumbers = ColorScheme[np.logical_not(np.isnan(ColorScheme))] 
    ColorMax = np.ceil(np.max(ColorNumbers))
    
    my_colors = cm.get_cmap('prism', 256)
    
    for j in range(len(diameter_array)):
        fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
        axs.set_xlim(scallop_length*2, scallop_length*4)
        axs.set_ylim(-1, 2)
        axs.set_aspect('equal')
        axs.plot(x_array, scallop_profile, 'grey')
        EnergyAtImpact[j, :][EnergyAtImpact[j, :]==0] = np.nan
        findColors = (np.log10(EnergyAtImpact[j, :]))/ColorMax 
        axs.scatter(XAtImpact[j, :], ZAtImpact[j, :], c = my_colors(findColors) )
        KEAvg = np.average(EnergyAtImpact[j,(np.logical_not(np.isnan(EnergyAtImpact[j, :])))][EnergyAtImpact[j,(np.logical_not(np.isnan(EnergyAtImpact[j, :])))]>0])
        KEMax = np.max(EnergyAtImpact[j,(np.logical_not(np.isnan(EnergyAtImpact[j, :])))])
        Max_x = XAtImpact[j, np.where(EnergyAtImpact[j, :]==np.max(EnergyAtImpact[j,(np.logical_not(np.isnan(EnergyAtImpact[j, :])))]))[0][0]]
    
        # Max_x[j] = KEXAtImpact[j, np.where(EnergyAtImpact[j, int(len(x_array)/2):int(len(x_array)/2+len(uScal))][EnergyAtImpact[j, int(len(x_array)/2):int(len(x_array)/2+len(uScal))]>0]==np.max(EnergyAtImpact[j, int(len(x_array)/2):int(len(x_array)/2+len(uScal))][EnergyAtImpact[j, int(len(x_array)/2):int(len(x_array)/2+len(uScal))]>0][np.where(~np.isnan(EnergyAtImpact[j, int(len(x_array)/2):int(len(x_array)/2+len(uScal))][EnergyAtImpact[j, int(len(x_array)/2):int(len(x_array)/2+len(uScal))]>0]))]))[0][0]]
        
        axs.set_ylabel('z (cm)') 
        axs.set_title('D= ' +str(round(diameter_array[j]*10, 2)) + ' mm, avg.KE= ' + str(round(KEAvg,2)) + ' ergs, MaxKE= ' +str(round(KEMax,2))+' ergs at x= ' +str(round(Max_x,2)))
        axs.set_xlabel('x (cm)')
    
    return fig, axs

    

if __name__ == "__main__":
    print('tests not implemented')