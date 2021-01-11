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


def settling_velocity(rho_sediment, rho_fluid, drag_coef, grain_size, fall_distance):
    R = (rho_sediment/rho_fluid - 1)
    g = 981 # cm*s^-2
    D = grain_size
    nu = 0.01307  # g*cm^-1*s^-1
    #C = drag_coef
    C_1 = 18
    C_2 = 1
    #w_s = np.sqrt((4*R*g*D)/(3*C))
    w_s = -(R*g*D**2)/((C_1*nu)+np.sqrt(0.75*C_2*R*g*D**3))
    
    return w_s

def particle_reynolds_number(D,urel,mu_kin):
    # Grain diameter, relative velocity (settling-ambient), kinematic viscosity
    return 2*D*np.abs(urel)/mu_kin

def sediment_saltation(x0, scallop_elevation, w_water, u_water, u_w0, w_s, D, dx, theta2, mu_kin, crest_height):
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
        while sediment_location[h, 2] >= 4:
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
            # get current location with respect to computational mesh -  this should be the previous h, above
            x_idx = np.rint((sediment_location[h, 1]/0.05))                
                
            if sediment_location[h, 2] <= 5:
                z_idx = np.rint((sediment_location[h, 2]/0.05))
            else:
                z_idx = 100
            
            wrel = sediment_location[h, 4] - w_water[int(z_idx), int(x_idx)]
            
            # make sure result of squaring wrel is above machine precision
            if np.abs(wrel) > eps2:                                       
                Re_p = particle_reynolds_number(D, wrel, mu_kin)
                drag_coef = dragcoeff(Re_p)
                #print('wrel', wrel, 'drag_coef', drag_coef)
                a = -(1 - (rho_w/rho_s)) * g - ((3 * rho_w * drag_coef) * (wrel**2) /(4 * rho_s * D))  
            else:
                a = 0
            
           # print('sediment_location[h, 1]', sediment_location[h, 1],'sediment_location[h, 3]', sediment_location[h, 3])               
            pi_x = sediment_location[h, 1] + sediment_location[h, 3] * dt
            #print('sediment_location[h, 3]', sediment_location[h, 3])
            pi_z = sediment_location[h, 2] + sediment_location[h, 4] * dt + 0.5 * a * dt**2   
            
            #print('x_idx= ', x_idx, ' z_idx= ', z_idx, 'pi_x', pi_x, 'pi_z= ', pi_z)
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

            
            if next_x_idx > 0 and pi_z <= scallop_elevation[int(next_x_idx)]:
                impact_data[i, :5] = sediment_location[h+1]
                #print('impact!')
                break
            
                            
            if pi_z <= 5:
                z_idx = np.int(np.rint((pi_z/0.05)))
            else:
                z_idx = 100
            
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
                
            if sediment_location[h, 2] <= 5:
                z_idx = np.rint((sediment_location[h, 2]/0.05))
            else:
                z_idx = 100
            
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
                z_idx = 100
            
            h+=1
            #print('h',h)

        location_data.append(sediment_location)   # store trajectory for plotting        
        #print('bedload thickness = ', Hf)
        
    return impact_data, location_data
       

if __name__ == "__main__":
    print('tests not implemented')