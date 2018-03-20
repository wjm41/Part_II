# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 11:31:15 2017

@author: user
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

def Cornu_point(start,end):
    cos_func= lambda x: np.cos(np.pi*(x**2)/2)
    sin_func= lambda x: np.sin(np.pi*(x**2)/2)
    C=integrate.quad(cos_func,start,end)[0]
    S=integrate.quad(sin_func,start,end)[0]
    x_y=np.array([C,S])
    return x_y 

def Generate_Spiral(start=-10,end=10,step=0.01):
    integral=[]
    i=start
    while i<=end:
        x_y=Cornu_point(0,i)
        integral.append(x_y)
        i+=step
    return integral

def Plot_Spiral():
    Raw_Data=Generate_Spiral()
    Plot_Data=np.array(Raw_Data) #changing 'Raw_Data' from (x1,y1),(x2,y2),... pair form to (x1,x2,...)(y1,y2,...) form for plotting purposes
    x,y=Plot_Data.T
    plt.plot(x,y)
    plt.xlabel("C(u)")
    plt.ylabel("S(u)")
    plt.title("Cornu Spiral")
    # From here
    ax = plt.gca()  
    ax.xaxis.set_label_coords(0.8, 0.4)
    ax.yaxis.set_label_coords(0.4, 0.8)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0)) 
    #To here, code just for setting origin of coordinate axes at (0,0)
    plt.savefig("cornu_spiral.eps")    
    plt.show()
    
def Amplitude_Data(Lambda=0.01,d=0.1,D=0.3,distance=25,num_points=3000):
    Scale=np.sqrt(2/(Lambda*D))
    x_y=[]
    Phasor=Cornu_point(-Scale*d/2,Scale*d/2)
    I_0=np.sqrt(np.sum(np.multiply(Phasor,Phasor)))
    distance_points=np.linspace(-distance,distance,num_points)
    for i in distance_points:
        Phasor=Cornu_point(-Scale*(d/2+i),Scale*(d/2-i))
        Rel_Amplitude=np.sqrt(np.sum(np.multiply(Phasor,Phasor)))/I_0
        x_y.append([i,Rel_Amplitude])
    return x_y

def Amplitude_Plot(Lambda=0.01,d=0.1,D=0.3,distance=25,num_points=3000):
    Plot_Data=np.array(Amplitude_Data(Lambda,d,D,distance,num_points)) 
    #changing from (x1,y1),(x2,y2),... pair form to 
    #(x1,x2,...)(y1,y2,...) form for plotting purposes
    x,y=Plot_Data.T
    plt.plot(x,y)
    plt.xlabel("Distance from Center of Pattern /cm")
    plt.ylabel("Relative Amplitude /arb. units")
    plt.title("Relative Amplitude of Diffraction Pattern for D="+str(D)+"cm")
    # From here
    ax = plt.gca() 
    ax.yaxis.set_label_coords(0.4, +0.7)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0)) 
    #To here, code just for setting origin of coordinate axes at (0,0)
    plt.savefig("diffraction_amplitude_"+str(D)+".eps")      
    plt.show()
    
def Phase_Data(Lambda=0.01,d=0.1,D=0.3,distance=25,num_points=3000):
    Scale=np.sqrt(2/(Lambda*D))
    x_y=[]
    Phasor=Cornu_point(-Scale*d/2,Scale*d/2)
    Phi_0=np.arctan(Phasor[1]/Phasor[0])
    distance_points=np.linspace(-distance,distance,num_points)
    for i in distance_points:
        Phasor=Cornu_point(-Scale*(d/2+i),Scale*(d/2-i))
        Rel_Phase=np.arctan(Phasor[1]/Phasor[0])-Phi_0
        x_y.append([i,Rel_Phase])
    return x_y

def Phase_Plot(Lambda=0.1,d=0.1,D=0.3,distance=25,num_points=3001):
    Plot_Data=np.array(Phase_Data(Lambda,d,D,distance,num_points))
    #changing from (x1,y1),(x2,y2),... pair form to 
    #(x1,x2,...)(y1,y2,...) form for plotting purposes
    x,y=Plot_Data.T
    plt.plot(x,y,lw=0.7)
    plt.xlabel("Distance from Center of Pattern /cm")
    plt.ylabel("Relative Phase /rad")
    plt.title("Relative Phase of Diffraction Pattern for D="+str(D)+"cm")
    # From here
    ax = plt.gca()  
    ax.yaxis.set_label_coords(0.4, +0.3)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0)) 
    #To here, code just for setting origin of coordinate axes at (0,0)
    plt.savefig("diffraction_phase_"+str(D)+".eps")
    plt.show()
    
print("Plotting graphs, should be done in a minute or two!")

Plot_Spiral()
lamd=1
d=10
D_list=[30,50,100]
for D in D_list:
    Amplitude_Plot(1,10,D,25,3001)
    Phase_Plot(1,10,D,25,3001)    