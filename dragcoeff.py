#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 16:27:29 2019

@author: dylanward

Calculates the drag coefficient as a piecewise function of the Reynolds number. 
Reynolds number must be provided as a single numerical value.
Returns Cd as a FLOAT.

If run standalone, prints two example calculations and, if NumPy and PyPlot are
available, plots the drag coefficient as a function of Reynolds number over the 
range [1e-2,1e7].

"""

from math import exp

try:
    # if numpy and  matplotlib are available, use it to plot this function if called directly
    from numpy import logspace, zeros
    import matplotlib.pyplot as plt
    plotting = True
except: 
    # otherwise turn the plotting code off
    plotting = False

__all__ = ['dragcoeff']


def main():
    """
    Code to demonstrate the dragcoeff function in this module
    
    e.g. 
        dragcoeff(0.2) -> 115.0825
        dragcoeff(5e6) -> 0.38
        
    If PyPlot is available, also returns a plot of CD as a function of Re
    """
    
    print("The drag coefficient for Re=0.2 is:", dragcoeff(0.2))
    print("The drag coefficient for Re=5e6 is:", dragcoeff(5e6))
    
    if plotting:
        res = logspace(-2,7,1e3)
        dragmtx = zeros(len(res))
        for i, re in enumerate(res):
            dragmtx[i] = dragcoeff(re)

        plt.plot(res,dragmtx,'b-')
        plt.title('Drag coefficient vs. Reynolds Number',Fontsize=18)
        plt.ylabel('$C_D$',Fontsize=16)
        plt.xlabel('$Re$',Fontsize=16)
        plt.xscale('log')
        plt.yscale('log')
        plt.show()



def dragcoeff(re):
    """
    Calculates the drag coefficient as a piecewise function of the Reynolds number.
    Reynolds number must be provided as a single numerical value.
    Returns Cd as a FLOAT.
    """
    
    if (re == 0):
        cd = 0
    elif (re < 0.1):
        cd = 24/re
    elif (re <= 1.0):
        cd = (22.73/re) - (0.0903/(re*re)) + 3.690
    elif (re <= 10.0):
        cd = (29.1667/re) - (3.8889/(re*re)) + 1.222
    elif (re <= 100):
        cd = (46.5/re) - (116.67/(re*re)) + .6167
    elif ( re <= 1000.0):
        cd = (98.33/re) - (2778.0/(re*re)) + .3644
    elif (re <= 5000.0):
        cd = (148.62/re) - (4.75e4/(re*re)) + 0.357
    elif (re <= 10000.0):
        cd = (-490.546/re) + (57.87e4/(re*re)) + 0.46
    elif (re <= 50000.0):
        cd = (-1662.5/re) + (5.4167e6/(re*re)) + 0.5191
    elif (re <=120000.0):
        cd = 0.5191
    elif (re <= 2000000.0):
        cd = 0.38 + (0.14*exp(-(re-1.2e5)/0.8e5)) - (1.3e-8 * ((re-1.2e5)**1.5)*exp(-(re-1.2e5)/1.5e5))
    elif (re <= 10000000.0):
        cd = 0.38
    else:
        cd = 0.38

    return cd



if __name__ == "__main__":
    main()
    
