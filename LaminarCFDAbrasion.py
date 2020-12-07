#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import darthabrader as da

# ## assumptions
# 
# 1. sediment concentration uniform in x, one grain begins at each x-index location at height of bedload
# 2. particle velocity influenced by water velocity, gravity, and viscosity
# 

# In[2]:



##variable declarations
nx = 101
ny = 101
nt = 10
nit = 50 
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
x = np.linspace(0, 5, nx)
y = np.linspace(0, 5, ny)
X, Y = np.meshgrid(x, y)


##physical variables
rho = 1000
nu = 4
F = 1
dt = .000001

#initial conditions
u = np.zeros((ny, nx)) + 60
un = np.zeros((ny, nx))

v = np.zeros((ny, nx))
vn = np.zeros((ny, nx))

p = np.ones((ny, nx))
pn = np.ones((ny, nx))

b = np.zeros((ny, nx))


numsc = 1
w = da.scallop_array_one(x, numsc)

udiff = 1
stepcount = 0

while udiff > 0.00001:
    un = u.copy()
    vn = v.copy()

    b = da.build_up_b(rho, dt, dx, dy, u, v)
    p = da.pressure_poisson_periodic(p, dx, dy, nit, b)

    u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt / dx * 
                    (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                     vn[1:-1, 1:-1] * dt / dy * 
                    (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                     dt / (2 * rho * dx) * 
                    (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                     nu * (dt / dx**2 * 
                    (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                     dt / dy**2 * 
                    (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])) + 
                     F * dt)

    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt / dx * 
                    (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                     vn[1:-1, 1:-1] * dt / dy * 
                    (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                     dt / (2 * rho * dy) * 
                    (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                     nu * (dt / dx**2 *
                    (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                     dt / dy**2 * 
                    (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

    # Periodic BC u @ x = -1     
    u[1:-1, -1] = (un[1:-1, -1] - un[1:-1, -1] * dt / dx * 
                  (un[1:-1, -1] - un[1:-1, -2]) -
                   vn[1:-1, -1] * dt / dy * 
                  (un[1:-1, -1] - un[0:-2, -1]) -
                   dt / (2 * rho * dx) *
                  (p[1:-1, 0] - p[1:-1, -2]) + 
                   nu * (dt / dx**2 * 
                  (un[1:-1, 0] - 2 * un[1:-1,-1] + un[1:-1, -2]) +
                   dt / dy**2 * 
                  (un[2:, -1] - 2 * un[1:-1, -1] + un[0:-2, -1])) + F * dt)

    # Periodic BC u @ x = 0
    u[1:-1, 0] = (un[1:-1, 0] - un[1:-1, 0] * dt / dx *
                 (un[1:-1, 0] - un[1:-1, -1]) -
                  vn[1:-1, 0] * dt / dy * 
                 (un[1:-1, 0] - un[0:-2, 0]) - 
                  dt / (2 * rho * dx) * 
                 (p[1:-1, 1] - p[1:-1, -1]) + 
                  nu * (dt / dx**2 * 
                 (un[1:-1, 1] - 2 * un[1:-1, 0] + un[1:-1, -1]) +
                  dt / dy**2 *
                 (un[2:, 0] - 2 * un[1:-1, 0] + un[0:-2, 0])) + F * dt)

    # Periodic BC v @ x = -1
    v[1:-1, -1] = (vn[1:-1, -1] - un[1:-1, -1] * dt / dx *
                  (vn[1:-1, -1] - vn[1:-1, -2]) - 
                   vn[1:-1, -1] * dt / dy *
                  (vn[1:-1, -1] - vn[0:-2, -1]) -
                   dt / (2 * rho * dy) * 
                  (p[2:, -1] - p[0:-2, -1]) +
                   nu * (dt / dx**2 *
                  (vn[1:-1, 0] - 2 * vn[1:-1, -1] + vn[1:-1, -2]) +
                   dt / dy**2 *
                  (vn[2:, -1] - 2 * vn[1:-1, -1] + vn[0:-2, -1])))

    # Periodic BC v @ x = 0
    v[1:-1, 0] = (vn[1:-1, 0] - un[1:-1, 0] * dt / dx *
                 (vn[1:-1, 0] - vn[1:-1, -1]) -
                  vn[1:-1, 0] * dt / dy *
                 (vn[1:-1, 0] - vn[0:-2, 0]) -
                  dt / (2 * rho * dy) * 
                 (p[2:, 0] - p[0:-2, 0]) +
                  nu * (dt / dx**2 * 
                 (vn[1:-1, 1] - 2 * vn[1:-1, 0] + vn[1:-1, -1]) +
                  dt / dy**2 * 
                 (vn[2:, 0] - 2 * vn[1:-1, 0] + vn[0:-2, 0])))
    
    
    # Wall BC: no-slip on scallop surface
    u[:16, 0] = 0
    u[:15, 1] = 0
    u[:15, 2] = 0
    u[:14, 3] = 0
    u[:13, 4] = 0
    u[:13, 5] = 0
    u[:12, 6] = 0
    u[:12, 7] = 0
    u[:11, 8] = 0
    u[:10, 9] = 0
    u[:10, 10] = 0
    u[:9, 11] = 0
    u[:9, 12] = 0
    u[:8, 13] = 0
    u[:8, 14] = 0
    u[:7, 15] = 0
    u[:7, 16] = 0
    u[:6, 17] = 0
    u[:6, 18] = 0
    u[:5, 19] = 0
    u[:5, 20] = 0
    u[:4, 21] = 0
    u[:4, 22] = 0
    u[:3, 23] = 0
    u[:3, 24] = 0
    u[:3, 25] = 0
    u[:2, 26] = 0
    u[:2, 27] = 0
    u[:2, 28] = 0
    u[:1, 29] = 0
    u[:1, 30] = 0
    u[:1, 31] = 0
    u[:1, 32] = 0
    u[:1, 33] = 0
    u[0, 34] = 0
    u[0, 35] = 0
    u[0, 36] = 0
    u[0, 37] = 0
    u[0, 38] = 0
    u[0, 39] = 0
    u[0, 40] = 0
    u[0, 41] = 0
    u[0, 42] = 0
    u[0, 43] = 0
    u[0, 44] = 0
    u[0, 45] = 0
    u[:1, 46] = 0
    u[:1, 47] = 0
    u[:1, 48] = 0
    u[:1, 49] = 0
    u[:1, 50] = 0
    u[:1, 51] = 0
    u[:2, 52] = 0
    u[:2, 53] = 0
    u[:2, 54] = 0
    u[:3, 55] = 0
    u[:3, 56] = 0
    u[:3, 57] = 0
    u[:4, 58] = 0
    u[:4, 59] = 0
    u[:4, 60] = 0
    u[:5, 61] = 0
    u[:5, 62] = 0
    u[:5, 63] = 0
    u[:6, 64] = 0
    u[:6, 65] = 0
    u[:6, 66] = 0
    u[:7, 67] = 0
    u[:7, 68] = 0
    u[:8, 69] = 0
    u[:8, 70] = 0
    u[:8, 71] = 0
    u[:9, 72] = 0
    u[:9, 73] = 0
    u[:9, 74] = 0
    u[:10, 75] = 0
    u[:10, 76] = 0
    u[:10, 77] = 0
    u[:11, 78] = 0
    u[:11, 79] = 0
    u[:11, 80] = 0
    u[:12, 81] = 0
    u[:12, 82] = 0
    u[:12, 83] = 0
    u[:12, 84] = 0
    u[:13, 85] = 0
    u[:13, 86] = 0
    u[:13, 87] = 0
    u[:13, 88] = 0
    u[:14, 89] = 0
    u[:14, 90] = 0
    u[:14, 91] = 0
    u[:14, 92] = 0
    u[:14, 93] = 0
    u[:15, 94] = 0
    u[:15, 95] = 0
    u[:15, 96] = 0
    u[:15, 97] = 0
    u[:15, 98] = 0
    u[:16, 99] = 0
    u[:17, 100] = 0
    v[:16, 0] = 0
    v[:15, 1] = 0
    v[:15, 2] = 0
    v[:14, 3] = 0
    v[:13, 4] = 0
    v[:13, 5] = 0
    v[:12, 6] = 0
    v[:12, 7] = 0
    v[:11, 8] = 0
    v[:10, 9] = 0
    v[:10, 10] = 0
    v[:9, 11] = 0
    v[:9, 12] = 0
    v[:8, 13] = 0
    v[:8, 14] = 0
    v[:7, 15] = 0
    v[:7, 16] = 0
    v[:6, 17] = 0
    v[:6, 18] = 0
    v[:5, 19] = 0
    v[:5, 20] = 0
    v[:4, 21] = 0
    v[:4, 22] = 0
    v[:3, 23] = 0
    v[:3, 24] = 0
    v[:3, 25] = 0
    v[:2, 26] = 0
    v[:2, 27] = 0
    v[:2, 28] = 0
    v[:1, 29] = 0
    v[:1, 30] = 0
    v[:1, 31] = 0
    v[:1, 32] = 0
    v[:1, 33] = 0
    v[0, 34] = 0
    v[0, 35] = 0
    v[0, 36] = 0
    v[0, 37] = 0
    v[0, 38] = 0
    v[0, 39] = 0
    v[0, 40] = 0
    v[0, 41] = 0
    v[0, 42] = 0
    v[0, 43] = 0
    v[0, 44] = 0
    v[0, 45] = 0
    v[:1, 46] = 0
    v[:1, 47] = 0
    v[:1, 48] = 0
    v[:1, 49] = 0
    v[:1, 50] = 0
    v[:1, 51] = 0
    v[:2, 52] = 0
    v[:2, 53] = 0
    v[:2, 54] = 0
    v[:3, 55] = 0
    v[:3, 56] = 0
    v[:3, 57] = 0
    v[:4, 58] = 0
    v[:4, 59] = 0
    v[:4, 60] = 0
    v[:5, 61] = 0
    v[:5, 62] = 0
    v[:5, 63] = 0
    v[:6, 64] = 0
    v[:6, 65] = 0
    v[:6, 66] = 0
    v[:7, 67] = 0
    v[:7, 68] = 0
    v[:8, 69] = 0
    v[:8, 70] = 0
    v[:8, 71] = 0
    v[:9, 72] = 0
    v[:9, 73] = 0
    v[:9, 74] = 0
    v[:10, 75] = 0
    v[:10, 76] = 0
    v[:10, 77] = 0
    v[:11, 78] = 0
    v[:11, 79] = 0
    v[:11, 80] = 0
    v[:12, 81] = 0
    v[:12, 82] = 0
    v[:12, 83] = 0
    v[:12, 84] = 0
    v[:13, 85] = 0
    v[:13, 86] = 0
    v[:13, 87] = 0
    v[:13, 88] = 0
    v[:14, 89] = 0
    v[:14, 90] = 0
    v[:14, 91] = 0
    v[:14, 92] = 0
    v[:14, 93] = 0
    v[:15, 94] = 0
    v[:15, 95] = 0
    v[:15, 96] = 0
    v[:15, 97] = 0
    v[:15, 98] = 0
    v[:16, 99] = 0
    v[:17, 100] = 0
    
    #water surface BC, du/dy = 0 @ y = -1, v = 0 @ y = -2
    u[-1, :] = u[-2, :] 
    v[-1, :]=0
    
    udiff = np.abs(np.sum(u) - np.sum(un)) / np.sum(u)
    stepcount += 1


    
print(stepcount)

fig = plt.figure(figsize=(11, 7), dpi=100)
plt.ylim(0,5)
#plt.quiver(X[::, ::], Y[::, ::], u[::, ::], v[::, ::]);
plt.contourf(X, Y, np.sqrt(u**2 + v**2), alpha = 0.5)
plt.colorbar()
#plt.contour(X, Y, u)
#plt.plot(x , w, 'b')
#plt.title('Velocity magnitude (cm/s) with streamlines')
#plt.streamplot(X, Y, u, v)
plt.xlabel('X')
plt.ylabel('Z');


# In[3]:


fig = plt.figure(figsize=(11, 7), dpi=100)
#plt.plot(x0[:101], z0[:101])
plt.contourf(X, Y, v, alpha = 0.7, vmin = -40, vmax = 40)
plt.colorbar()
plt.title('Z-directed velocity, laminar flow')
plt.xlabel('X')
plt.ylabel('Z');


# In[4]:


# six-scallop long velocity matrix

u_water = np.empty(shape=(101,601))
w_water = np.empty(shape=(101,601))

a = 6
b = len(u) - 1
c = len(u) - 1

for i in range(a):
    for j in range(b):   
        u_water[:, i*b + j] = u[:, j]
        w_water[:, i*b + j] = v[:, j]
        


# In[5]:


# building the initial scallop array

xmax = 6  #number of scallops
dx = 0.01
xScal = np.arange(0, xmax+dx,dx)  #x-array for scallop field
uScal = np.arange(0,1+dx,dx)  #x-array for a single scallop


x0, z0 = da.scallop_array(xScal, uScal, xmax)   #initial scallop profile, dimensions in centimeters
dzdx = np.gradient(z0, x0)
theta2 = np.arctan(dzdx)  #slope angle at each point along scalloped profile


# In[6]:


# definitions and parameters

Hf_gravel = 7.92    #distance of fall for 60-mm gravel (Lamb et al., 2008)
Hf_sand = 3.84     #distance of fall for 1-mm sand (Lamb et al., 2008)
D_gravel = 6
D_sand = 0.1
rho_quartz = 2.65  # g*cm^-3
rho_water = 1
Re = 23300     #Reynold's number from scallop formation experiments (Blumberg and Curl, 1974)
drag_coef = (-490.546/Re) + (57.87e4/(Re*Re)) + 0.46
mu_water = 0.013  # g*cm^-1*s^-1
L = 5    # cm, Sauter-mean crest-to-crest scallop length

Hf_grain = 6
D_grain = 4.5


# In[7]:


Re = 23300
(-490.546/Re) + (57.87e4/(Re*Re)) + 0.46


# In[8]:



# In[9]:


u_w0 = (Re * mu_water) / (L * rho_water)   # cm/s, assume constant downstream, x-directed velocity equal to average velocity of water as in Curl (1974)

w_s_gravel = da.settling_velocity(rho_quartz, rho_water, drag_coef, D_gravel, Hf_gravel)
mass_gravel = np.pi * rho_quartz * D_gravel**3 / 6
w_s_sand = da.settling_velocity(rho_quartz, rho_water, drag_coef, D_sand, Hf_sand)
mass_sand = np.pi * rho_quartz * D_sand**3 / 6

mass_grain = np.pi * rho_quartz * D_grain**3 / 6
w_s_grain = da.settling_velocity(rho_quartz, rho_water, drag_coef, D_grain, Hf_grain)


# In[10]:


print(w_s_sand)
print(w_s_gravel)


gravel_energy, gravel_w_i, impact_data = da.sediment_saltation(x0, z0, Hf_gravel, w_water, u_water, u_w0, w_s_gravel, mass_gravel, dx, theta2, drag_coef)

fig, axs = plt.subplots(nrows = 3, ncols = 1, figsize = (11,22))    
axs[0].set_xlim(10, 25)
axs[0].plot (x0, z0, 'grey')
axs[0].scatter (x0, gravel_energy)
axs[0].set_ylabel('KE (Joules)')
axs[0].set_xlabel('x (cm)')
axs[0].set_title('Kinetic energy of impacting 60 mm gravel on floor scallops, D/L = 1.2')
axs[1].set_xlim(10, 25)
axs[1].plot (x0, z0, 'grey')
axs[1].scatter(x0, z0, (gravel_energy)/10**5)
axs[1].set_ylabel('z (cm)')
axs[1].set_xlabel('x (cm)')
axs[1].set_title('Kinetic energy of impacting 60 mm gravel on floor scallops, dot size scales with energy, D/L = 1.2')
axs[2].set_xlim(10, 25)
axs[2].plot (x0, z0, 'grey')
axs[2].scatter (x0, gravel_w_i)
axs[2].set_title ('60 mm Gravel velocity normal to surface at point of impact (cm/s)')
axs[2].set_xlabel ('x (cm)')
axs[2].set_ylabel ('w_i (cm/s)');


# In[14]:


B = 9.4075*10**-12
max_E = np.zeros_like(x0)
for a in range(len(gravel_w_i)):
    if gravel_w_i[a] <= 0:
        max_E[a] = -(315576000 * B * gravel_w_i[a]**3)
    elif gravel_w_i[a] > 0:
        max_E[a] = 0

plt.plot (x0, z0*40, 'grey')
plt.scatter (x0, max_E)
plt.ylabel('E (mm/year)')
plt.xlabel('x (cm)')
plt.title('Maximum possible erosion rate, 60 mm gravel on scallops, no alluvial cover');


# In[15]:


sand_energy, sand_w_i, impact_data= da.sediment_saltation(x0, z0, Hf_sand, w_water, u_water, u_w0, w_s_sand, mass_sand, dx, theta2, drag_coef)

fig, axs = plt.subplots(nrows = 3, ncols = 1, figsize = (11,22))    
#axs[0].set_xlim(10, 25)
axs[0].plot (x0, z0, 'grey')
axs[0].scatter (x0, sand_energy)
axs[0].set_ylabel('KE (ergs)')
axs[0].set_xlabel('x (cm)')
axs[0].set_title('Kinetic energy of impacting 1 mm sand on floor scallops, D/L = 0.02')
#axs[1].set_xlim(10, 25)
axs[1].plot (x0, z0, 'grey')
axs[1].scatter(x0, z0, (sand_energy)*10**2)
axs[1].set_ylabel('z (cm)')
axs[1].set_xlabel('x (cm)')
axs[1].set_title('Kinetic energy of impacting 1 mm sand on floor scallops, dot size scales with energy, D/L = 0.02')
#axs[2].set_xlim(10, 25)
axs[2].plot (x0, z0, 'grey')
axs[2].scatter (x0, sand_w_i)
axs[2].set_title ('1 mm Sand velocity normal to surface at point of impact (cm/s)')
axs[2].set_xlabel ('x (cm)')
axs[2].set_ylabel ('w_i (cm/s)');


# In[ ]:


B = 9.4075*10**-12
for a in range(len(sand_w_i)):
    if sand_w_i[a] <= 0:
        max_E[a] = -(315576000 * B * sand_w_i[a]**3)
    elif sand_w_i[a] > 0:
        max_E[a] = 0


plt.plot (x0, z0*40, 'grey')
plt.scatter (x0, max_E)
plt.ylabel('E (mm/year)')
plt.xlabel('x (cm)')
plt.title('Maximum possible erosion rate, 1 mm sand on scallops, no alluvial cover');


# In[ ]:


grain_energy, grain_w_i, impact_data = da.sediment_saltation(x0, z0, Hf_grain, w_water, u_water, u_w0, w_s_grain, mass_grain, dx, theta2, drag_coef)

fig, axs = plt.subplots(nrows = 3, ncols = 1, figsize = (11,22))    
axs[0].set_xlim(10, 25)
axs[0].plot (x0, z0, 'grey')
axs[0].scatter (x0, grain_energy*10**-7)
axs[0].set_ylabel('KE (Joules)')
axs[0].set_xlabel('x (cm)')
axs[0].set_title('Kinetic energy of impacting 45 mm gravel on floor scallops, D/L = 0.8')
axs[1].set_xlim(10, 25)
axs[1].plot (x0, z0, 'grey')
axs[1].scatter(x0, z0, (grain_energy)/10**4)
axs[1].set_ylabel('z (cm)')
axs[1].set_xlabel('x (cm)')
axs[1].set_title('Kinetic energy of impacting 45 mm gravel on floor scallops, dot size scales with energy, D/L = 0.8')
axs[2].set_xlim(10, 25)
axs[2].plot (x0, z0, 'grey')
axs[2].scatter (x0, grain_w_i)
axs[2].set_title ('45 mm gravel velocity normal to surface at point of impact (cm/s)')
axs[2].set_xlabel ('x (cm)')
axs[2].set_ylabel ('w_i (cm/s)');


# In[ ]:


B = 9.4075*10**-12

for a in range(len(grain_w_i)):
    if grain_w_i[a] <= 0:
        max_E[a] = -(315576000 * B * grain_w_i[a]**3)
    elif grain_w_i[a] > 0:
        max_E[a] = 0


plt.plot (x0, z0*40, 'grey')
plt.scatter (x0, max_E)
plt.ylabel('E (mm/year)')
plt.xlabel('x (cm)')
plt.title('Maximum possible erosion rate, 45 mm gravel on scallops, no alluvial cover');


# In[ ]:


fig, ax = plt.subplots()

line1, = ax.plot(x0, gravel_w_i, '--', label = '60 mm')
line2, = ax.plot(x0,grain_w_i, 'o',label = '45 mm')
line3, = ax.plot(x0, sand_w_i, '^', label = '1 mm')

ax.legend()
plt.show()






