# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 14:10:15 2020

@author: rachelbosch
"""


# In[1]:


import numpy as np
from matplotlib import pyplot as plt
import darthabrader as da
from numpy import genfromtxt

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
l32 = 5 # sauter-mean scallop length in cm
n = 6 # number of grainsizes to simulate in diameter array
numScal = 24  #number of scallops



# building the initial scallop array

dx0 = 0.05/l32
xScal = np.arange(0, numScal+dx0,dx0)  #x-array for scallop field
uScal = np.arange(0,1+dx0,dx0)  #x-array for a single scallop


x0, z0 = da.scallop_array(xScal, uScal, numScal, l32)   #initial scallop profile, dimensions in centimeters
z0 = z0 - np.min(z0)
cH = np.max(z0)   # crest height
dzdx = np.gradient(z0, x0)
theta2 = np.arctan(dzdx)  #slope angle at each point along scalloped profile


# import and process turbulent flow data set
TurbVel = genfromtxt('TurbulentFlowfield'+str(l32)+'.csv', delimiter=',')

# variable declarations
nx = l32*20 + 1
ny = l32*20 + 1
new_x = np.linspace(0, l32/numScal, int(nx))
new_z = np.linspace(0, l32/numScal, int(ny))
new_X, new_Z = np.meshgrid(new_x, new_z)
new_u = np.zeros((int(ny), int(nx)))
new_w = np.zeros((int(ny), int(nx)))

u_water, w_water = da.turbulent_flowfield(TurbVel, xScal, uScal, numScal, new_u, new_w)


fig = plt.figure(figsize=(11, 7), dpi=100)
plt.contourf(new_X, new_Z, np.sqrt(new_u**2 + new_w[:len(new_u)]**2), alpha = 0.5)
plt.colorbar()
plt.title('Velocity magnitude, turbulent flow')
plt.xlabel('X')
plt.ylabel('Z');




# In[6]:


# definitions and parameters

grain_diam_max = 0.5 * l32 
grain_diam_min = 0.035 * l32
diam = grain_diam_max * np.logspace((np.log10(grain_diam_min/grain_diam_max)), 0, n)
EnergyAtImpact = np.empty(shape = (len(diam), len(x0)))
XAtImpact = np.empty(shape = (len(diam), len(x0)))
ZAtImpact = np.empty(shape = (len(diam), len(x0)))
ErosionAtImpact = np.empty(shape = (len(diam), len(x0)))
VelocityAtImpact = np.empty(shape = (len(diam), len(x0)))
ParticleDrag = np.empty_like(diam)
ParticleReynolds = np.empty_like(diam)

i = 0
for D in diam:
    xi = np.linspace(0, 1, 5)
    delta = cH + (0.5 + 3.5 * xi)*D
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
    
    
    
    
    # In[9]:
    
    
    u_w0 = (Re * mu_water) / (l32 * rho_water)   # cm/s, assume constant downstream, x-directed velocity equal to average velocity of water as in Curl (1974)
    
    w_s = da.settling_velocity(rho_quartz, rho_water, D) 
    
    # In[10]:
    
    impact_data, loc_data= da.sediment_saltation(x0, z0, w_water, u_water, u_w0, w_s, D, 0.05, theta2, mu_water, cH, l32)
    
    ImpactEnergyAvg = np.empty_like(diam)
    TotalImpactEnergy = np.empty_like(diam)
    ImpactEnergyTotalAvg = np.average(impact_data[:, 6])
    NumberImpacts = np.count_nonzero(impact_data[:, 6])
    if (NumberImpacts == 0):
        ImpactEnergyAvg[i] = ImpactEnergyTotalAvg/NumberImpacts 
    TotalImpactEnergy[i] = np.sum(impact_data[300:401, 6])
    AverageVelocities = np.empty_like(diam)
    MaxVelocities = np.empty_like(diam)
    
    ParticleDrag[i] = np.average(impact_data[:, 8])
    
    ParticleReynolds[i] = np.average(impact_data[:, 7])
      
    EnergyAtImpact[i, :] = impact_data[:, 6]
    XAtImpact[i, :] = impact_data[:, 1]
    ZAtImpact[i, :] = impact_data[:, 2]
    VelocityAtImpact[i, :] = impact_data[:, 5]
    
    B = 9.4075*10**-12  # s**2·cm**-2
    ErosionAtImpact[i, :] = B * (impact_data[:, 5])**3    ##Lamb et al., 2008
    
    #of the grains, that have recorded impact, those with negative impact velocities are directed into the scalloped surface
    AverageVelocities[i]= np.average(impact_data[:,5][impact_data[:,5]<0])
    MaxVelocities[i] = -np.min(impact_data[:,5])
    
    print('diam = ' + str(diam[i]) + ' cm')
    i += 1
    
    # # trajectory figure
    # fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))    
    # axs.set_xlim(l32*numScal/2, (l32*numScal/2 + l32*4))
    # axs.set_ylim(0, l32*2)
    # axs.set_aspect('equal')
    # axs.plot (x0, z0, 'grey')
    # ld = np.array(loc_data, dtype=object)
    # for p in ld[(np.random.randint(len(loc_data),size=1000)).astype(int)]:
    #     axs.plot(p[:,1], p[:,2], 2, 'blue')
    # plt.fill_between(x0, z0, 0, alpha = 1, color = 'grey', zorder=101)
    # axs.set_ylabel('z (cm)')
    # axs.set_xlabel('x (cm)')
    # axs.set_title('Trajectories of randomly selected ' + str(round(D*10, 3)) + ' mm '+ grain +' on ' +str(l32)+ ' cm floor scallops, fall height = ' + str(round(Hf, 3)) + ' cm.')
    
#     # velocity exploration
#     ###histogram of last recorded velocities of all particles
#     fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))    
#     axs.hist(impact_data[:, 5], 20)
#     axs.set_xlabel('w_i (cm/s)')
#     axs.set_title('Histogram of impact velocities of ' + str(round(D*10, 3)) + ' mm ' + grain + ' on 5 cm floor scallops. Fall height = ' + str(round(Hf, 3)) + ' cm.')
    
#       ###velocities with time
#     fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))    
#     for p in ld[(np.random.randint(len(loc_data),size=100)).astype(int)]:
#           axs.plot(p[:,0], np.sqrt(p[:,4]**2 + p[:,3]**2), 2, 'blue')
#     axs.set_ylabel('v_s (cm)')
#     axs.set_xlabel('t (sec)')
#     axs.set_title('Velocity magnitudes of randomly selected ' + str(round(D*10, 3)) + ' mm '+ grain +' in flow over 5 cm scallops')
         
#     ### vertical velocities with time
#     fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))    
#     for p in ld[(np.random.randint(len(loc_data),size=100)).astype(int)]:
#         axs.plot(p[:,0], p[:,4], 2, 'blue')
#     axs.set_ylabel('v_s (cm)')
#     axs.set_xlabel('t (sec)')
#     axs.set_title('Vertical velocities of randomly selected ' + str(round(D*10, 3)) + ' mm '+ grain +' in flow over 5 cm scallops')



### average velocities plot 
fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
from scipy.optimize import curve_fit
Diam5 = genfromtxt('diam5.csv', delimiter=',')
Vels5 = genfromtxt('VelocityAvg5.csv', delimiter=',')
not_nan_idx = np.where(~np.isnan(Vels5))
Diam5=Diam5[not_nan_idx]
Vels5=Vels5[not_nan_idx]
# #Stokes = (1.65 * 981/(18*0.01307))*diam**2 
# w_s = da.settling_velocity(rho_quartz, rho_water, diam)
# w_s_2 = da.settling_velocity(rho_quartz, rho_water, Diam5)
# VelocityAvg = np.zeros_like(diam)
# for r in range(len(diam)):
#     VelocityAvg[r] = -np.average(VelocityAtImpact[r, int(len(x0)/2):int(len(x0)/2+len(uScal))][VelocityAtImpact[r, int(len(x0)/2):int(len(x0)/2+len(uScal))]<0])
# axs.scatter((diam), VelocityAvg, label = 'simulated impact velocity on '+str(l32)+' cm scallops')
# axs.plot(Diam5*10, -w_s_2, c = 'g', label = 'settling velocity (Ferguson and Church, 2004)')
#axs.plot(diam*10, Stokes, c = 'y', label = 'settling velocity (Stokes)')
axs.scatter((Diam5*10), Vels5, label = 'simulated impact velocity on 5.0 cm scallops', zorder = 0)

def settling_velocity(D, C_1, C_2):
    R = 1.65
    g = 981 # cm*s^-2
    nu = 0.01307  # g*cm^-1*s^-1
    return (R*g*D**2)/((C_1*nu)+np.sqrt(0.75*C_2*R*g*D**3))

pars, cov = curve_fit(f=settling_velocity, xdata=Diam5, ydata=Vels5, p0=[24, 1.2], bounds=(-np.inf, np.inf))
# Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
stdevs = np.sqrt(np.diag(cov))
# Calculate the residuals
res = Vels5 - settling_velocity(Diam5, *pars)
axs.plot(Diam5*10, settling_velocity(Diam5, *pars), linestyle='--', linewidth=2, color='black', label = 'fitted to Equation (13), with C_1= '+str(round(pars[0], 2))+' and C_2= '+str(round(pars[1], 2)))

plt.legend()
plt.semilogx()
#axs.set_xlim(0.01,0.4)
axs.set_xlabel('particle grainsize (mm)')
axs.set_ylabel('velocity (cm/s)') 
axs.set_title('Particle velocities over 5 cm scallops')
plt.show()


# # =============================================================================
# # Two different ways to get erosion rate due to abrasion:
#     # 1. Work-energy theorem: work done on limestone = change in kinetic energy of sandstone grain
#     # 2. Erosion rate due to abrasion expression from Lamb et al. (2008). 
# # =============================================================================
# # ### 1. WORK-ENERGY THEOREM (these results are very much the wrong order of magnitude)
# # CoR = 0.4    #coefficient of restitution for sandstone impinging on limestone
# # WorkDoneAtImpact = EnergyAtImpact * (1 - CoR)   #work done on limestone, work-energy theorem

# # for j in range(len(diam)):
# #     fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
    
# #     mlr = (np.pi * rho_ls * diam[j]**3)/6    # mass limestone removed
# #     dv_dt = WorkDoneAtImpact/(mlr * diam[j])
# #     E_we = dv_dt * rho_ls * 24 * 3600  # convert cm*s**-1 to g*cm**-2*day*-1
    
# #     if diam[j] < 0.0063:
# #         grain = 'silt'
# #     elif diam[j] >= 0.0063 and D < 0.2:
# #         grain = 'sand'
# #     elif diam[j] >= 0.2:
# #         grain = 'gravel'
# #     axs.set_xlim(15, 25)

# #     #axs.set_aspect('equal')
# #     axs.plot(x0, (E_we[j]))
   
    
# #     axs.set_xlim(15, 25)
# #     axs.set_ylabel('Erosion rate (g/(cm^2*day))')
# #     axs.set_xlabel('x (cm)')
    
# #     axs.set_title('Erosion rate by work-energy theorem, ' + str(round(diam[j]*10, 3)) + ' mm '+ grain +' on floor scallops')

# # plt.show()

# # ### 2. BUFFALO WILD WINGS AND WECK
# fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
# labels = str(np.around(diam*10, 3)) + ' mm '
# for j in range(len(diam)):
#     E_bw3 = -(ErosionAtImpact[j, :]*10*3600*24*365.25)  
#     for k in range(len(E_bw3)):                                  
#         if E_bw3[k] < 0:
#             E_bw3[k] = 0
#     axs.scatter(x0, E_bw3, label=str(round(diam[j]*10, 1)) + ' mm')
# axs.semilogy()       
# axs.set_xlim(15, 25)
# axs.set_ylabel('Erosion rate (mm/yr)')
# axs.set_xlabel('x (cm)')   
# axs.set_title('Erosion rates on '+str(l32)+' cm floor scallops')
# plt.legend()
# plt.show()



# ### Plot erosion rate (BW3 approach) v. scallop surface slope    


# for j in range(len(diam)):
#     fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
    
#     E_bw3 = -(ErosionAtImpact[j, :] * rho_ls * 24 * 3600 * 1000)  # convert Lamb et al. (2008) units to Hammer et al. (2011) units
#     for k in range(len(E_bw3)):                                  # cm*s**-1 to mg*cm**-2*day**-1
#         if E_bw3[k] < 0:
#             E_bw3[k] = 0
    
#     if diam[j] < 0.0063:
#         grain = 'silt'
#     elif diam[j] >= 0.0063 and D < 0.2:
#         grain = 'sand'
#     elif diam[j] >= 0.2:
#         grain = 'gravel'

#     axs.scatter(theta2, E_bw3)
   

#     axs.set_ylabel('Erosion rate (mg/(cm^2*day))')
#     axs.set_xlabel('dz/dx')
    
#     axs.set_title('Erosion rate ala BW3, D = ' + str(round(diam[j]*10, 3)) + ' mm, v. floor scallop local slope')

# plt.show()




# ## Plot abrasion and dissolution regimes

# E_bw3_array = np.zeros(shape=(len(diam), len(x0)))

# array_lengths = np.zeros_like(diam)

# fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))


# for j in range(len(diam)):
#     E_bw3_array[j, :] = -(ErosionAtImpact[j, :]*10*3600*24*365.25)  
#     PositiveErosion = E_bw3_array[j, :][E_bw3_array[j, :] > 0]                                  
#     test = len(PositiveErosion)
#     array_lengths[j] = test
# max_length = np.max(array_lengths)
# ErosionRanges = np.zeros(shape=(int(max_length), len(diam)))
# for k in range(len(diam)):
#     E_bw3_array[k, :] = -(ErosionAtImpact[k, :]*10*3600*24*365.25)  
#     PositiveErosion = E_bw3_array[k, :][E_bw3_array[k, :] > 0] 
#     column_length = len(PositiveErosion)
#     ErosionRanges[:column_length, k] = PositiveErosion

# # 'fill-between' for dissolutional domain
# diameters=np.zeros_like(ErosionRanges)
# for q in range(len(diam)):
#     diameters[:, q]=diam[q]
#     axs.scatter(diameters[:,q]*10, ErosionRanges[:, q])
# diss_min = 5.66    #minimum dissolution rate (mm/yr) (Grm et al., 2017)
# diss_max = 12.175  #maximum dissolution rate (mm/yr) (Hammer et al., 2011)
# x = diam[:]*10
# plt.fill_between(x, diss_min, diss_max, alpha = 0.8, color = 'grey')

# #fill-between for erosional regions
# diam_min = 0.1 #mm
# abrasion_begins = 1
# dissolution_ends = 2.5
# diam_max = 25
# axs.axvspan(diam_min, dissolution_ends, alpha=0.4, color='yellow')
# axs.axvspan(abrasion_begins, diam_max, alpha=0.4, color='cyan')
# plt.semilogy()
# plt.semilogx()
# axs.set_title('Erosional regimes over scallops')
# #axs.set_title('Simulated abrasion rates')
# axs.set_xlabel('Particle grain size (mm)')
# axs.set_ylabel('erosion rate (mm/yr)')
# axs.yaxis.grid(True)

# plt.show()


# ####Total abrasion Over One Scallop
# fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))

# ErosionSum = np.zeros_like(diam)
# NormErosionAvg = np.zeros_like(diam)
# NumberOfImpactsByGS = np.zeros_like(diam)
# for r in range(len(diam)):
#     ErosionSum[r] = -np.sum(ErosionAtImpact[r, int(len(x0)/2):int(len(x0)/2+len(uScal))][ErosionAtImpact[r, int(len(x0)/2):int(len(x0)/2+len(uScal))]<0]*1000*36*24*365.25)
#     NumberPositives = len(ErosionAtImpact[r, int(len(x0)/2):int(len(x0)/2+len(uScal))][ErosionAtImpact[r, int(len(x0)/2):int(len(x0)/2+len(uScal))]<0])
#     NumberOfImpactsByGS[r] = NumberPositives
#     if NumberPositives > 0:
#         NormErosionAvg[r] = ErosionSum[r]/NumberPositives
#     else:
#         NormErosionAvg[r] = 0


# NumerousImpacts = NumberOfImpactsByGS[NumberOfImpactsByGS>=5]
# impact_idx = np.where(NumberOfImpactsByGS>=5)
# NumImpDiam = diam[impact_idx]
# NormNumErosion = NormErosionAvg[impact_idx]
# axs.scatter((diam*10), (NormErosionAvg), label = 'simulated abrasional erosion on '+str(l32)+' cm scallops')
# plt.semilogx()
# plt.legend(loc = 'center left')
# axs.set_title('Abrasion Rate Normalized by Number of Impacts (>=5)')
# axs.set_xlabel('particle grainsize (mm)')
# axs.set_ylabel('abrasional erosion rate (mm/yr)')
# axs.grid(True, which = 'both', axis = 'both')

# plt.show()




# ####Total abrasion Over 5 cm Scallop with dissolution comparison
# from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes 
# from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
# Diam5 = genfromtxt('diam5.csv', delimiter=',')
# EAI5 = genfromtxt('ErosionAtImpact5.csv', delimiter=',')
# ES5 = np.zeros_like(Diam5)
# NEA5 = np.zeros_like(Diam5)
# NOI5 = np.zeros_like(Diam5)
# for s in range(len(Diam5)):
#     ES5[s] = -np.sum(EAI5[s, int(len(x0)/12):int(len(x0))][EAI5[s, int(len(x0)/12):int(len(x0))]<0]*1000*36*24*365.25)
#     NP5 = len(EAI5[s, int(len(x0)/12):int(len(x0))][EAI5[s, int(len(x0)/12):int(len(x0))]<0])
#     NOI5[s] = NP5
#     if NP5 > 0:
#         NEA5[s] = ES5[s]/NP5
#     else:
#         NEA5[s] = 0



# NumerousImpacts = NOI5[NOI5>=5]
# impact_idx = np.where(NOI5>=5)
# NumImpDiam = Diam5[impact_idx]
# NormNumErosion = NEA5[impact_idx]
# axs.scatter((Diam5*10), (NEA5), label = 'simulated abrasional erosion')
# diss_min = 5.66    #minimum dissolution rate (mm/yr) (Grm et al., 2017)
# diss_max = 12.175  #maximum dissolution rate (mm/yr) (Hammer et al., 2011)
# x = np.linspace(0.9, 5)
# plt.fill_between(x, diss_min, diss_max, alpha = 0.6, color = 'r')

# # first = len(Diam5)-len(NEA5[NEA5 > 0])
# # line_fit_2=np.polyfit(np.log10(Diam5[first:]*10), np.log10(NEA5[NEA5 > 0]), deg=1, full=True)
# # y = (line_fit_2[0][0])*(np.log10(Diam5*10)) + (line_fit_2[0][1])
# # axs.plot((Diam5*10), 10**y, 'r', label = 'fit curve, E_A = '+str(round((line_fit_2[0][0]), 3))+'*log(D) '+str(round((line_fit_2[0][1]),3))+'', color = 'g')




# plt.semilogx()
# plt.legend(loc = 'center left')
# #plt.semilogy()
# # axs.set_ylim(30,200)
# axs.set_xlim(.9,30)
# axs.set_title('Abrasion Rate Normalized by Number of Impacts on 5 cm Scallops (>=5)')
# axs.set_xlabel('particle grainsize (mm)')
# axs.set_ylabel('abrasional erosion rate (mm/yr)')
# axs.grid(True, which = 'both', axis = 'both')

# #select the x-range for the zoomed region
# x1 = 1.5
# x2 = 4.5

# # select y-range for zoomed region
# y1 = 0
# y2 = 50

# # Make the zoom-in plot:
# axins = zoomed_inset_axes(axs, 4, bbox_to_anchor=(0,0), loc = 'upper left')
# axins.scatter((Diam5*10), (NEA5))
# axins.fill_between(x, diss_min, diss_max, alpha = 0.6, color = 'r', label = 'dissolutional range')
# #axins.plot((Diam5*10), 10**y, 'r', color = 'g')
# axins.set_xlim(x1, x2)
# axins.set_ylim(y1, y2)
# plt.xticks(visible=True)
# plt.yticks(visible=True)
# axins.legend(loc = 'upper center')
# mark_inset(axs, axins, loc1=2, loc2=1, fc="none", ec="0.5")
# plt.draw()
# plt.show()





# ###Number of impacts plot
# fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))

# NI5 = NOI5[NOI5>=1]
# IG5 = Diam5[NOI5>=1]

# axs.scatter(diam*10, NumberOfImpactsByGS, label = 'impacts on 2.5 cm scallop')
# axs.scatter(Diam5*10, NOI5, label = 'impacts on 5 cm scallop', zorder = 0)
# plt.title('Number of particle impacts on one scallop')
# axs.set_xlabel('particle grainsize (mm)')
# axs.set_ylabel('number of impacts')
# plt.semilogx()
# plt.legend(loc='upper left')
# axs.grid(True, which = 'both', axis = 'x')
# plt.show()

# # =============================================================================
# # curve fitting and Geary's test for goodness of fit

first = len(Diam5)-len(NEA5[NEA5 > 0])
line_fit_2=np.polyfit(np.log10(Diam5[first:]*10), np.log10(NEA5[NEA5 > 0]), deg=1, full=True)
y = (line_fit_2[0][0])*(np.log10(Diam5*10)) + (line_fit_2[0][1])
#y = 3*(np.log10(diam*10)) -1.8
axs.plot((Diam5*10), 10**y, 'r', label = 'fit curve, log(E_A) = ')

TSS = 0 #total sum of squares
sum_abs = 0
for s in range(len(NormErosionAvg[NormErosionAvg > 0])):
    square = (np.log10(NormErosionAvg[NormErosionAvg > 0][s]) - np.average(np.log10(NormErosionAvg[NormErosionAvg > 0])))**2
    diff_abs = np.abs(np.log10(NormErosionAvg[NormErosionAvg > 0][s]) - np.average(np.log10(NormErosionAvg[NormErosionAvg > 0])))
    TSS = TSS + square
    sum_abs = sum_abs +diff_abs
sigma=np.sqrt(TSS/len(NormErosionAvg[NormErosionAvg > 0]))
numerator= (np.sqrt(np.pi/2))*sum_abs/(len(NormErosionAvg[NormErosionAvg > 0]))
Gearys_test_EA = numerator/sigma          #### confirm that data is log-normally distributed
# # =============================================================================

# # ### check for drag crisis
# # fig, ax1 = plt.subplots(nrows = 1, ncols = 1, figsize = (11,8.5))
# # plt.semilogx()
# # plt.semilogy()

# # color = 'tab:red'
# # ax1.set_xlabel('particle grainsize (mm)')
# # ax1.set_ylabel('drag coefficient at impact', color='b')
# # ax1.scatter(diam*10, ParticleDrag, color = 'b')
# # ax1.tick_params(axis='y', labelcolor='b')

# # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
# # plt.semilogy()

# # color = 'tab:blue'
# # ax2.set_ylabel('Particle Reynolds Number at impact', color='r')  # we already handled the x-label with ax1
# # ax2.scatter(diam*10, ParticleReynolds, color='r')
# # ax2.tick_params(axis='y', labelcolor='r')

# # fig.tight_layout()  # otherwise the right y-label is slightly clipped
# # plt.show()

# # impact location & energy-at-location plot

# from matplotlib import colors
# from matplotlib import cm

# GetMaxEnergies = EnergyAtImpact[-1, :][EnergyAtImpact[-1, :] != 0]
# ColorScheme = np.log10(GetMaxEnergies)  ## define color scheme to be consistent for every plot
# ColorNumbers = ColorScheme[np.logical_not(np.isnan(ColorScheme))] 
# ColorMax = np.ceil(np.max(ColorNumbers))
# KESum = np.zeros_like(diam)
# KEAvg = np.zeros_like(diam)
# NumKE = np.zeros_like(diam)

# my_colors = cm.get_cmap('cool', 256)
# fig, axs = plt.subplots(nrows = len(diam), ncols = 1, sharex=True, figsize = (11, 17))

# for j in range(len(diam)):
#     axs[j].set_xlim(55, 65)
#     axs[j].set_ylim(-0.5, 1.5)
#     axs[j].set_aspect('equal')
#     axs[j].plot(x0, z0, 'grey')
#     EnergyAtImpact[j, :][EnergyAtImpact[j, :]==0] = np.nan
#     findColors = (np.log10(EnergyAtImpact[j, :]))/ColorMax 
#     impact_dots = axs[j].scatter(XAtImpact[j, :], ZAtImpact[j, :], c = my_colors(findColors) )
    
#     KESum[j] = np.sum(EnergyAtImpact[j, int(len(x0)/2):int(len(x0)/2+len(uScal))][EnergyAtImpact[j, int(len(x0)/2):int(len(x0)/2+len(uScal))]>0])
#     NumPos = len(EnergyAtImpact[j, int(len(x0)/2):int(len(x0)/2+len(uScal))][EnergyAtImpact[j, int(len(x0)/2):int(len(x0)/2+len(uScal))]>0])
#     NumKE[j] = NumPos
#     if NumPos > 0:
#         KEAvg[j] = KESum[j]/NumPos
#     else:
#         KEAvg[j] = 0
    

#     axs[j].set_ylabel('z (cm)') 
#     axs[j].set_title('D = ' +str(round(diam[j]*10, 2)) + ' mm                     avg.KE = ' + str(round(KEAvg[j],2)) + ' ergs')
# #legend
# fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
#                     wspace=0.4, hspace=0.1)
# cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
# norm = colors.Normalize(vmin = 0, vmax = ColorMax)
# cbar = plt.colorbar(cm.ScalarMappable(norm = norm, cmap='cool'), cax = cb_ax)
# cb_ax.set_ylabel('log10 of Kinetic energy of impact (ergs)')
# axs[-1].set_xlabel('x (cm)')
# plt.show()

# # #####save all data
# # np.savetxt('VelocityAtImpact'+str(l32)+'.csv',VelocityAtImpact,delimiter=",")
# # np.savetxt('ImpactEnergyAvg'+str(l32)+'.csv',ImpactEnergyAvg,delimiter=",")
# # np.savetxt('VelocityAvg'+str(l32)+'.csv',VelocityAvg,delimiter=",")
# # np.savetxt('EnergyAtImpact'+str(l32)+'.csv',EnergyAtImpact,delimiter=",")
# # np.savetxt('XAtImpact'+str(l32)+'.csv',XAtImpact,delimiter=",")
# # np.savetxt('ZAtImpact'+str(l32)+'.csv',ZAtImpact,delimiter=",")
# # np.savetxt('ErosionAtImpact'+str(l32)+'.csv',ErosionAtImpact,delimiter=",")
# # np.savetxt('AverageVelocities'+str(l32)+'.csv',AverageVelocities,delimiter=",")
# # np.savetxt('MaxVelocities'+str(l32)+'.csv',MaxVelocities,delimiter=",")
# # np.savetxt('diam'+str(l32)+'.csv',diam,delimiter=",")
# # np.savetxt('TotalImpactEnergy'+str(l32)+'.csv',TotalImpactEnergy,delimiter=",")
# # np.savetxt('ParticleDrag'+str(l32)+'.csv',ParticleDrag,delimiter=",")
# # np.savetxt('ParticleReynolds'+str(l32)+'.csv',ParticleReynolds,delimiter=",")
# # np.savetxt('NormErosionAvg'+str(l32)+'.csv',NormErosionAvg,delimiter=",")


