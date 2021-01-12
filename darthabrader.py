#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 14:07:04 2020

@author: dylanward

This is a first crack at a library for scallop abrasion functions
Based on Jupyter notebooks by Rachel Bosch
"""
import numpy as np
from dragcoeff import dragcoeff
#import pdb

#CFD laminar flow field

def __init__(self):
    pass

def build_up_b(rho, dt, dx, dy, u, v):
    b = np.zeros_like(u)
    b[1:-1, 1:-1] = (rho * (1 / dt * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) +
                                      (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                            ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                            2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                                 (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx))-
                            ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))
    
    # Periodic BC Pressure @ x = 2
    b[1:-1, -1] = (rho * (1 / dt * ((u[1:-1, 0] - u[1:-1,-2]) / (2 * dx) +
                                    (v[2:, -1] - v[0:-2, -1]) / (2 * dy)) -
                          ((u[1:-1, 0] - u[1:-1, -2]) / (2 * dx))**2 -
                          2 * ((u[2:, -1] - u[0:-2, -1]) / (2 * dy) *
                               (v[1:-1, 0] - v[1:-1, -2]) / (2 * dx)) -
                          ((v[2:, -1] - v[0:-2, -1]) / (2 * dy))**2))

    # Periodic BC Pressure @ x = 0
    b[1:-1, 0] = (rho * (1 / dt * ((u[1:-1, 1] - u[1:-1, -1]) / (2 * dx) +
                                   (v[2:, 0] - v[0:-2, 0]) / (2 * dy)) -
                         ((u[1:-1, 1] - u[1:-1, -1]) / (2 * dx))**2 -
                         2 * ((u[2:, 0] - u[0:-2, 0]) / (2 * dy) *
                              (v[1:-1, 1] - v[1:-1, -1]) / (2 * dx))-
                         ((v[2:, 0] - v[0:-2, 0]) / (2 * dy))**2))
    
    return b

def pressure_poisson_periodic(p, dx, dy, nit, b):
    pn = np.empty_like(p)
    
    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 +
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                         (2 * (dx**2 + dy**2)) -
                         dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1])

        # Periodic BC Pressure @ x = 2
        p[1:-1, -1] = (((pn[1:-1, 0] + pn[1:-1, -2])* dy**2 +
                        (pn[2:, -1] + pn[0:-2, -1]) * dx**2) /
                       (2 * (dx**2 + dy**2)) -
                       dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, -1])

        # Periodic BC Pressure @ x = 0
        p[1:-1, 0] = (((pn[1:-1, 1] + pn[1:-1, -1])* dy**2 +
                       (pn[2:, 0] + pn[0:-2, 0]) * dx**2) /
                      (2 * (dx**2 + dy**2)) -
                      dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 0])
        
        # Wall boundary conditions, pressure
        p[-1, :] =p[-2, :]  # dp/dy = 0 at y = 2
        p[0, :] = p[1, :]  # dp/dy = 0 at y = 0
    
    return p

# Define scallop surface
def scallop_array_one(x_array, number_of_scallops):
    
    y = np.ones(x_array.shape)
    T = np.int_((len(x_array) - 1) / number_of_scallops)
   
    one_period = np.linspace(0, 1, T + 1) #x-array for a single scallop
    period = len(one_period)-1
    n = number_of_scallops 
           
    for i in range(n):
        for j in range(period):
            v =1 -((0.112 * np.sin(np.pi * one_period)) + (0.028 * np.sin(2 * np.pi * one_period)) - (0.004 * np.sin(3 * np.pi * one_period)))
            y[i*period + j] = v[j]    
        y = y*5 - 4.37
   
    return y

def scallop_array(x_array, one_period, number_of_scallops, crest_to_crest_scallop_length):
    
    l32 = crest_to_crest_scallop_length
    z = np.zeros(x_array.shape)
    period = len(one_period)-1
    n = number_of_scallops 
           
    for i in range(n):
        for j in range(period):
            v =-((0.112 * np.sin(np.pi * one_period)) + (0.028 * np.sin(2 * np.pi * one_period)) - (0.004 * np.sin(3 * np.pi * one_period)))    #(Blumberg and Curl, 1974)
            z[i*period + j] = v[j]
    
    x = x_array *l32
    z = z*l32

    return x, z

def turbulent_flowfield(CFD_dataset, x_array, one_period, number_of_scallops, x_directed_flow_velocity, z_directed_flow_velocity):
    #restructure STAR-CCM+ turbulent flow data set
    for i in range(len(CFD_dataset)):
        x_index = np.int(CFD_dataset[i, 0])
        z_index = np.int(CFD_dataset[i, 1])
        x_directed_flow_velocity[z_index, x_index] = CFD_dataset[i, 2]
        z_directed_flow_velocity[z_index, x_index] = CFD_dataset[i, 3]
     
    if np.any(x_directed_flow_velocity == -9999):    #identify holes in data set and patch them    
        holes_and_wall_u = np.where(x_directed_flow_velocity == -9999)
        ux_zero = np.array(holes_and_wall_u[1])
        uz_zero = np.array(holes_and_wall_u[0])
        
        for j in range(len(ux_zero)-1):
            hole_z = np.int(uz_zero[j])
            hole_x = np.int(ux_zero[j])
            d = 8
            uN = x_directed_flow_velocity[hole_z + 1, hole_x]
            wN = z_directed_flow_velocity[hole_z + 1, hole_x]
            if uN == 0 or uN == -9999:
                uN = 0
                wN = 0
                d = d - 1
            uNE = x_directed_flow_velocity[hole_z + 1, hole_x + 1]
            wNE = z_directed_flow_velocity[hole_z + 1, hole_x + 1]
            if uNE == 0 or uNE == -9999:
                uNE = 0
                wNE = 0
                d = d - 1
            uE = x_directed_flow_velocity[hole_z, hole_x + 1]
            wE = z_directed_flow_velocity[hole_z, hole_x + 1]
            if uE == 0 or uE == -9999:
                uE = 0
                wE = 0
                d = d - 1
            uSE = x_directed_flow_velocity[hole_z - 1, hole_x + 1]
            wSE = z_directed_flow_velocity[hole_z - 1, hole_x + 1]
            if uSE == 0 or uSE == -9999:
                uSE = 0
                wSE = 0
                d = d - 1
            uS = x_directed_flow_velocity[hole_z - 1, hole_x]
            wS = z_directed_flow_velocity[hole_z - 1, hole_x]
            if uS == 0 or uS == -9999:
                uS = 0
                wS = 0
                d = d - 1
            uSW = x_directed_flow_velocity[hole_z - 1, hole_x - 1]
            wSW = z_directed_flow_velocity[hole_z - 1, hole_x - 1]
            if uSW == 0 or uSW == -9999:
                uSW = 0
                wSW = 0
                d = d - 1
            uW = x_directed_flow_velocity[hole_z, hole_x - 1]
            wW = z_directed_flow_velocity[hole_z, hole_x - 1]
            if uW == 0 or uW == -9999:
                uW = 0
                wW = 0
                d = d - 1
            uNW = x_directed_flow_velocity[hole_z + 1, hole_x - 1]
            wNW = z_directed_flow_velocity[hole_z + 1, hole_x - 1]
            if uNW == 0 or uNW == -9999:
                uNW = 0
                wNW = 0
                d = d - 1
            x_directed_flow_velocity[hole_z, hole_x] = ((uN + uNE + uE + uSE + uS + uSW + uW + uNW)/d)
            z_directed_flow_velocity[hole_z, hole_x] = ((wN + wNE + wE + wSE + wS + wSW + wW + wNW)/d) 
            
    x_directed_flow_velocity[:, -1] = x_directed_flow_velocity[:, 0]
    z_directed_flow_velocity[:, -1] = z_directed_flow_velocity[:, 0]
    
    
    # six-scallop long velocity matrix
    
    u_water = np.empty(shape=(int(len(one_period)),int(len(x_array))))
    w_water = np.empty(shape=(int(len(one_period)),int(len(x_array))))
    
    b = len(x_directed_flow_velocity) - 1
    
    for i in range(number_of_scallops):
        for j in range(b):   
            u_water[:, i*b + j] = x_directed_flow_velocity[:, j]
            w_water[:, i*b + j] = z_directed_flow_velocity[:, j]
            
    return u_water, w_water
    
    
    

def settling_velocity(rho_sediment, rho_fluid, grain_size):
    R = (rho_sediment/rho_fluid - 1)
    g = 981 # cm*s^-2
    D = grain_size
    nu = 0.01307  # g*cm^-1*s^-1
    C_1 = 18
    C_2 = 1
    w_s = -(R*g*D**2)/((C_1*nu)+np.sqrt(0.75*C_2*R*g*D**3))
    
    return w_s

def particle_reynolds_number(D,urel,mu_kin):
    # Grain diameter, relative velocity (settling-ambient), kinematic viscosity
    return 2*D*np.abs(urel)/mu_kin

def sediment_saltation(x0, scallop_elevation, w_water, u_water, u_w0, w_s, D, dx, theta2, mu_kin, crest_height, scallop_length):
    ### define constants and parameters
    rho_w = 1
    rho_s = 2.65
    drag = (3 * rho_w/(rho_w + 2 * rho_s))  ##### velocity factor for sphere transported by fluid (Landau and Lifshitz, 1995)
    g = 981
    m = np.pi * rho_s * D**3 / 6
    
    #calculate bedload height as function of grain size (Wilson, 1987)
    xi = np.linspace(0, 1, 5)
    delta = crest_height + (0.5 + 3.5 * xi)*D
    Hf = delta[1]

        
    l_ds = -(3 * Hf * u_w0) / (2 * w_s)  # length of saltation hop for trajectory calculation above CFD flow field (Lamb et al., 2008)
    impact_data = np.zeros(shape=(len(x0), 9))  # 0 = time, 1 = x, 2 = z, 3 = u, 4 = w, 5 = |Vel|, 6 = KE, 7 = Re_p, 8 = drag coefficient; one row per particle
    dt = dx / u_w0
    location_data = []
    # define machine epsilon threshold
    eps2=np.sqrt( u_w0*np.finfo(float).eps )

    
    for i in range(len(x0)):    #begin one particle at rest at each x-position at its fall height (Hf per Wilson, 1987)
        h = 0
        t = 0
        OOB_FLAG = False
        sediment_location = np.zeros(shape=(1, 5))  
        sediment_location[0, :] = [t, x0[i], Hf, 0, 0]
           #initial position for ith particle, # 0 = time, 1 = x, 2 = z, 3 = u, 4 = w 
        
        # upper level of fall, ignores turbulent flow structure
        while sediment_location[h, 2] >= (0.99 * scallop_length) :
            h += 1
            t += dt
            if i+h < (x0.size - 1):
                pi_x = x0[i + h]    # new horizontal step location
            else:
                #print(' out of bounds in upper zone' )
                OOB_FLAG = True
                break
                
            pi_z = (-(Hf/(l_ds)**2)*(pi_x - x0[i])**2) + Hf 
            pi_u = drag * u_w0
            pi_w = -(Hf - pi_z)/t            
            sediment_location = np.append(sediment_location, [[t, pi_x, pi_z, pi_u, pi_w]], axis = 0)  
            x_idx = (i + h)
            z_idx = np.rint((pi_z/0.05))
            
        # near-ground portion, with drag
        while not OOB_FLAG and h < x0.size and sediment_location[h, 2] > scallop_elevation[h]:        #while that particle is in transport in the water
            t += dt
            # get current location with respect to computational mesh at time = t - dt
            x_idx = np.rint((sediment_location[h, 1]/0.05))                
            if sediment_location[h, 2] <= (0.99 * scallop_length):
                z_idx = np.rint((sediment_location[h, 2]/0.05))
            else:
                z_idx = scallop_length/.05
            
            wrel = sediment_location[h, 4] - w_water[int(z_idx), int(x_idx)]
            
            # make sure result of squaring wrel is above machine precision
            if np.abs(wrel) > eps2:                                       
                Re_p = particle_reynolds_number(D, wrel, mu_kin)
                drag_coef = dragcoeff(Re_p)
                a = -(1 - (rho_w/rho_s)) * g - ((3 * rho_w * drag_coef) * (wrel**2) /(4 * rho_s * D))  
            else:
                a = 0
            
            pi_x = sediment_location[h, 1] + sediment_location[h, 3] * dt                        # particle i x-position at time = t
            pi_z = sediment_location[h, 2] + sediment_location[h, 4] * dt + 0.5 * a * dt**2      # particle i z-position at time = t  
            pi_u = drag * u_water[int(z_idx), int(x_idx)]                                        # particle i x-velocity at time = t
            pi_w = sediment_location[h, 4] + (drag * w_water[int(z_idx), int(x_idx)]) + (a * dt) # particle i z-velocity at time = t
            if pi_w < w_s:
                pi_w = w_s
            sediment_location = np.append(sediment_location, [[t, pi_x, pi_z, pi_u, pi_w]], axis = 0)

            
            # projected next 
            try:
                next_x_idx = np.int(np.rint((pi_x/0.05)))
            except:
                print('NaN in pi_x. this is fixed and should never happen again!')
                next_x_idx = -9999
                raise Exception
                
            #print ('next_x', next_x_idx)
            if next_x_idx >= x0.size or next_x_idx < 0:
                #print('out of bounds in lower zone!')
                OOB_FLAG = True
                break                        

            
            if next_x_idx > 0 and pi_z <= scallop_elevation[int(next_x_idx)]:
                impact_data[i, :5] = sediment_location[h+1]
                #print('impact!')
                break
            
                            
            if pi_z <= 5:
                z_idx = np.int(np.rint((pi_z/0.05)))
            else:
                z_idx = scallop_length/.05
            
            h+=1
            #print('h',h)

            
    
        if impact_data[i,3] != 0:
            theta1 = np.arctan(impact_data[i, 4]/impact_data[i, 3])             
        else:
            #print('div/0 or other error in theta1')
            theta1 = 0
            
        alpha = np.pi - theta1 - theta2[i]          # angle of impact
            
        impact_data[i, 5] = (np.sqrt(impact_data[i, 4]**2 + impact_data[i, 3]**2))*np.sin(alpha)
        if impact_data[i, 5] <= 0:          
            impact_data[i, 6] += 0.5 * m * impact_data[i, 5]**2
        else:
            impact_data[i, 6] += 0 
        
# for intuitive-looking trajectory plotting, draw the trajectories through the scallops:
        while not OOB_FLAG and h < x0.size and sediment_location[h, 2] > 0:        #while that particle is in transport in the water
          
            t += dt
        # get current indices -  this should be the previous h, above
            x_idx = np.rint((sediment_location[h, 1]/0.05))                
                
            if sediment_location[h, 2] <= 0.8*scallop_length:
                z_idx = np.rint((sediment_location[h, 2]/0.05))
            else:
                z_idx = scallop_length/.05
            
            wrel = sediment_location[h, 4] - w_water[int(z_idx), int(x_idx)]
            
        # make sure result of squaring wrel is above machine precision
            if np.abs(wrel) > eps2:                                       
                Re_p = particle_reynolds_number(D, wrel, mu_kin)
                drag_coef = dragcoeff(Re_p)
                a = -(1 - (rho_w/rho_s)) * g - ((3 * rho_w * drag_coef) * (wrel**2) /(4 * rho_s * D))  
            else:
                a = 0
            
            pi_x = sediment_location[h, 1] + sediment_location[h, 3] * dt
            pi_z = sediment_location[h, 2] + sediment_location[h, 4] * dt + 0.5 * a * dt**2   
            pi_u = drag * u_water[int(z_idx), int(x_idx)]
            pi_w = sediment_location[h, 4] + (drag * w_water[int(z_idx), int(x_idx)]) + (a * dt)
            if pi_w < w_s:
                pi_w = w_s
            sediment_location = np.append(sediment_location, [[t, pi_x, pi_z, pi_u, pi_w]], axis = 0)
        
        # projected next 
            try:
                next_x_idx = np.int(np.rint((pi_x/0.05)))
            except:
                print('NaN in pi_x. this is fixed and should never happen again!')
                next_x_idx = -9999
                raise Exception
                
        #print ('next_x', next_x_idx)
            if next_x_idx >= x0.size or next_x_idx < 0:
            #print('out of bounds in lower zone!')
                OOB_FLAG = True
                break                        

            
            if next_x_idx > 0 and pi_z <= 0:
                break
            
                            
            if pi_z <= 5:
                z_idx = np.int(np.rint((pi_z/0.05)))
            else:
                z_idx = scallop_length/.05
            
            h+=1
            #print('h',h)

        location_data.append(sediment_location)   # store trajectory for plotting        
        #print('bedload thickness = ', Hf)
        
    return impact_data, location_data
       

if __name__ == "__main__":
    print('tests not implemented')