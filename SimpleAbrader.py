# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 10:42:39 2020

@author: rachelbosch

This is a simpliflied, streamlined scallop abrasion calculator to recreate
motivational simulation justifying the comparison of abrasion and dissolution

Based on darthabrader.py by Dylan Ward, based on Jupyter notebooks by Rachel Bosch
"""
import numpy as np

# Define scallop surface
def scallop_array(x_array, one_period, number_of_scallops):
    
    z = np.zeros(x_array.shape)
    period = len(one_period)-1
    n = number_of_scallops 
           
    for i in range(n):
        for j in range(period):
            v =-((0.112 * np.sin(np.pi * one_period)) + (0.028 * np.sin(2 * np.pi * one_period)) - (0.004 * np.sin(3 * np.pi * one_period)))    #(Blumberg and Curl, 1974)
            z[i*period + j] = v[j]
    
    x = (x_array/number_of_scallops) * 30
    z = 0.7 + z*5

    return x, z

def settling_velocity(rho_sediment, rho_fluid, grain_size, fall_distance):
    R = (rho_sediment/rho_fluid - 1)
    g = 981 # cm*s^-2
    D = grain_size
    nu = 0.01307  # g*cm^-1*s^-1
    C_1 = 20
    C_2 = 1.1
    w_s = -(R*g*D**2)/((C_1*nu)+np.sqrt(0.75*C_2*R*g*D**3)) #(Ferguson and Church, 2004, using conservative Dietrich parameters)
    
    return w_s

def sediment_saltation(x0, scallop_elevation, u_w0, w_s, D, dx, theta2, mu_kin):
    ### define constants and parameters
    rho_w = 1
    rho_s = 2.65
    drag = (3 * rho_w/(rho_w + 2 * rho_s))  ##### velocity factor for sphere transported by fluid (Landau and Lifshitz, 1995)
    m = np.pi * rho_s * D**3 / 6
    
    #calculate bedload height as function of grain size (Wilson, 1987)
    xi = np.linspace(0, 1, 5)
    delta = 0.7 + (0.5 + 3.5 * xi)*D
    Hf = delta[1]

        
    l_ds = -(3 * Hf * u_w0) / (2 * w_s)  # length of saltation hop for trajectory calculation above CFD flow field (Lamb et al., 2008)
    impact_data = np.zeros(shape=(len(x0), 7))  # 0 = time, 1 = x, 2 = z, 3 = u, 4 = w, 5 = |Vel|, 6 = KE; one row per particle
    dt = dx / u_w0
    location_data = []

    



    
    for i in range(len(x0)):    #begin one particle at reast at each x-position at its fall height (Hf per Lamb et al., 2008)
        OOB_FLAG = False
        x_idx = 0
        h = 0
        t = 0
        sediment_location = np.zeros(shape=(1, 5))  
        sediment_location[0, :] = [t, x0[i], Hf, 0, 0]
           #initial position for ith particle, # 0 = time, 1 = x, 2 = z, 3 = u, 4 = w 

        while not OOB_FLAG and sediment_location[h-1, 2] > scallop_elevation[int(x_idx)]:           
            # this simulation ignores turbulent flow structure
            t += dt
            if i+h < (x0.size - 1):
                pi_x = x0[i + h]    # new horizontal step location
            else:
                OOB_FLAG = True
                break
            x_idx = np.rint((sediment_location[h, 1]/0.05)) 
               
            pi_z = (-(Hf/(l_ds)**2)*(pi_x - x0[i])**2) + Hf 
            pi_u = drag * u_w0
            pi_w = -(Hf - pi_z)/t 
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
                
            if next_x_idx >= x0.size or next_x_idx < 0:
                OOB_FLAG = True
                break     
                    
            if next_x_idx > 0 and pi_z <= scallop_elevation[int(next_x_idx)]:
                print('i = ', i, ' h = ', h, ' next_x_idx = ', next_x_idx)
                impact_data[i, :5] = sediment_location[h]
                print('impact!')
                break
            
    
            h+=1
            #print('h',h)
    
                
    
        if impact_data[i,3] != 0:
            theta1 = np.arctan(impact_data[i, 4]/impact_data[i, 3])             
        else:
            print('div/0 or other error in theta1')
            theta1 = 0
            
        alpha = np.pi - theta1 - theta2[i]          # angle of impact
            
        impact_data[i, 5] = (np.sqrt(impact_data[i, 4]**2 + impact_data[i, 3]**2))*np.sin(alpha)
        if impact_data[i, 5] <= 0:          
            impact_data[i, 6] += 0.5 * m * impact_data[i, 5]**2
        else:
            impact_data[i, 6] += 0 
        
        location_data.append(sediment_location)   # store trajectory for plotting
        
        print('bedload thickness = ', Hf)
        
    return impact_data, location_data
       

if __name__ == "__main__":
    print('tests not implemented')