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
l32 = 1 # choose 1, 2.5, 5, or 10, sauter-mean scallop length in cm
n = 1000  #number of grainsizes to simulate in diameter array
numScal = 24  #number of scallops
flow_regime = 'turbulent'    ### choose 'laminar' or 'turbulent'
if flow_regime == 'laminar':
    l32 = 5

# =============================================================================

#build the bedrock scallop array

dx0 = 0.05/l32
xScal = np.arange(0, numScal+dx0,dx0)  #x-array for scallop field
uScal = np.arange(0,1+dx0,dx0)  #x-array for a single scallop

x0, z0 = da.scallop_array(xScal, uScal, numScal, l32)   #initial scallop profile, dimensions in centimeters
z0 = z0 - np.min(z0)
cH = np.max(z0)   # crest height
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
grain_diam_max = 0.5 * l32 
grain_diam_min = 0.002 * l32

diam = grain_diam_max * np.logspace((np.log10(grain_diam_min/grain_diam_max)), 0, n)
EnergyAtImpact = np.empty(shape = (len(diam), len(x0)))
XAtImpact = np.empty(shape = (len(diam), len(x0)))
ZAtImpact = np.empty(shape = (len(diam), len(x0)))
ErosionAtImpact = np.empty(shape = (len(diam), len(x0)))
VelocityAtImpact = np.empty(shape = (len(diam), len(x0)))
ParticleDrag = np.empty_like(diam)
ParticleReynolds = np.empty_like(diam)
ImpactEnergyAvg = np.empty_like(diam)
TotalImpactEnergy = np.empty_like(diam)
AverageVelocities = np.empty_like(diam)
MaxVelocities = np.empty_like(diam)

# loop over diameter array to run the saltation function for each grainsize
i = 0
for D in diam:
    xi = np.linspace(0, 1, 5)
    delta = cH + (0.5 + 3.5 * xi)*D   # bedload thickness equation (Wilson, 1987)
    Hf = delta[1]    #beload thickness (cm)
    
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
    B = 9.4075*10**-12  # s**2·cm**-2,  abrasion coefficient (Bosch and Ward, 2021)
    
    u_w0 = (Re * mu_water) / (l32 * rho_water)   # cm/s, assume constant downstream, x-directed velocity equal to average velocity of water as in Curl (1974)
    w_s = da.settling_velocity(rho_quartz, rho_water, D) 
    
    # In[10]:
    
    impact_data, loc_data= da.sediment_saltation(x0, z0, w_water, u_water, u_w0, w_s, D, 0.05, theta2, mu_water, cH, l32)
    
    ###sort output data into arrays
    ImpactEnergyTotalAvg = np.average(impact_data[:, 6])
    NumberImpacts = np.count_nonzero(impact_data[:, 6])
    if (NumberImpacts == 0):
        ImpactEnergyAvg[i] = ImpactEnergyTotalAvg/NumberImpacts 
    TotalImpactEnergy[i] = np.sum(impact_data[300:401, 6])
    ParticleDrag[i] = np.average(impact_data[:, 8])
    ParticleReynolds[i] = np.average(impact_data[:, 7])
    EnergyAtImpact[i, :] = impact_data[:, 6]
    XAtImpact[i, :] = impact_data[:, 1]
    ZAtImpact[i, :] = impact_data[:, 2]
    VelocityAtImpact[i, :] = impact_data[:, 5]
    ErosionAtImpact[i, :] = B * (impact_data[:, 5])**3    ##  (cm/s) Lamb et al., 2008
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

#Process velocity array to average values over one scallop length
VelocityAvg = np.zeros_like(diam)
for r in range(len(diam)):
    VelocityAvg[r] = -np.average(VelocityAtImpact[r, int(len(x0)/2):int(len(x0)/2+len(uScal))][VelocityAtImpact[r, int(len(x0)/2):int(len(x0)/2+len(uScal))]<0])

#Process erosion rate array to average values over one scallop length and normalize by number of impacts
ErosionSum = np.zeros_like(diam)
NormErosionAvg = np.zeros_like(diam)
NumberOfImpactsByGS = np.zeros_like(diam)
for r in range(len(diam)):
    ErosionSum[r] = -np.sum(ErosionAtImpact[r, int(len(x0)/2):int(len(x0)/2+len(uScal))][ErosionAtImpact[r, int(len(x0)/2):int(len(x0)/2+len(uScal))]<0]*1000*36*24*365.25)
    NumberPositives = len(ErosionAtImpact[r, int(len(x0)/2):int(len(x0)/2+len(uScal))][ErosionAtImpact[r, int(len(x0)/2):int(len(x0)/2+len(uScal))]<0])
    NumberOfImpactsByGS[r] = NumberPositives
    if NumberPositives > 0:
        NormErosionAvg[r] = ErosionSum[r]/NumberPositives
    else:
        NormErosionAvg[r] = 0

#####save all data
np.savetxt('./outputs2/VelocityAtImpact'+str(l32)+flow_regime+'.csv',VelocityAtImpact,delimiter=",")
np.savetxt('./outputs2/ImpactEnergyAvg'+str(l32)+flow_regime+'.csv',ImpactEnergyAvg,delimiter=",")
np.savetxt('./outputs2/VelocityAvg'+str(l32)+flow_regime+'.csv',VelocityAvg,delimiter=",")
np.savetxt('./outputs2/EnergyAtImpact'+str(l32)+flow_regime+'.csv',EnergyAtImpact,delimiter=",")
np.savetxt('./outputs2/XAtImpact'+str(l32)+flow_regime+'.csv',XAtImpact,delimiter=",")
np.savetxt('./outputs2/ZAtImpact'+str(l32)+flow_regime+'.csv',ZAtImpact,delimiter=",")
np.savetxt('./outputs2/ErosionAtImpact'+str(l32)+flow_regime+'.csv',ErosionAtImpact,delimiter=",")
np.savetxt('./outputs2/AverageVelocities'+str(l32)+flow_regime+'.csv',AverageVelocities,delimiter=",")
np.savetxt('./outputs2/MaxVelocities'+str(l32)+flow_regime+'.csv',MaxVelocities,delimiter=",")
np.savetxt('./outputs2/diam'+str(l32)+flow_regime+'.csv',diam,delimiter=",")
np.savetxt('./outputs2/TotalImpactEnergy'+str(l32)+flow_regime+'.csv',TotalImpactEnergy,delimiter=",")
np.savetxt('./outputs2/ParticleDrag'+str(l32)+flow_regime+'.csv',ParticleDrag,delimiter=",")
np.savetxt('./outputs2/ParticleReynolds'+str(l32)+flow_regime+'.csv',ParticleReynolds,delimiter=",")
np.savetxt('./outputs2/NormErosionAvg'+str(l32)+flow_regime+'.csv',NormErosionAvg,delimiter=",")

####plot results, all plotting schemes available in scallopplotlib.py
pars, stdevs, res, fig, axs = spl.average_velocities_plot(rho_quartz, rho_water, diam, l32, VelocityAvg)
plt.show()

# fig, axs, axins = spl.abrasion_and_dissolution_plot(x0)
# plt.draw()
# plt.show()
