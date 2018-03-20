# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 15:31:56 2017

@author: user
"""
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

#Defining the form of the ODE for use with odeint
def ODE(y,t,g,l,Omega,q,F):
    return [y[1], -(g/l)*np.sin(y[0])-q*y[1]+F*np.sin(Omega*t)]

#Defining system parameters
g=1
l=1
Omega=1.0*2/3
tau=2*np.pi

def Test_ODE(y0,T,q,F,xlim,N):
    '''
    Tests the result of the ODE integrator versus the theoretical prediction
    of the undamped, undriven motion by plotting the theoretical prediction 
    of the pendulum displacement on top of that of the ODE integrator.
    N oscillations from the xlim-th oscillation are plotted with the inital
    angular displacements and velocities specified in y0, and using the given
    values of q and F. 1000*T points and T periods are used in the calculation.
    '''
    y0=[y0, 0.0] #set initial conditions
    t=np.linspace(0.0, tau*T, 1000*T) 
    
    #integrate the system over the time specified in t
    y=scipy.integrate.odeint(ODE,y0,t,args=(g,l,Omega,q,F,))
    
    #Plot the results
    fig=plt.figure()
    ax1 = fig.add_subplot(211)    
    ax1.plot(t/tau,y[:,0],lw=3.4,label="Displacement (rad)", color="red")
    ax1.plot(t/tau,y0[0]*np.cos(t),lw=1.6,label="theory", color="blue")
    ax1.set_ylabel("Displacement/rad")
    title_name=r"$\theta_0$ = "+str(y0[0])+" q = "+str(q)+" F = "+str(F)
    ax1.set_title(title_name)
    ax1.legend(loc="upper right")
    ax1.set_xlim([xlim,xlim+N]) #restrict the x range to N oscillations
    
    ax2 = fig.add_subplot(212)    
    ax2.plot(t/tau,y[:,1],lw=3.4,label="Angular Velocity", color="red")
    ax2.plot(t/tau,-y0[0]*np.sin(t),lw=1.6,label="theory", color="blue")
    ax2.set_xlabel("Number of oscillations")
    ax2.set_ylabel("Angular Velocity /rad$s^{-1}$")
    ax2.legend(loc="upper right")
    ax2.set_xlim([xlim,xlim+5]) 
    plot_name="Task_2_check.eps"
    plt.savefig(plot_name,bbox_inches="tight",transparent=True)
    plt.show()
    
def Energy_Conservation(y0,T,q,F):
    '''
    Plots the fraction of energy in the system against time. The inital
    angular displacements and velocities specified in y0, with the given
    values of q and F. 1000*T points and T periods are used in the calculation.
    '''
    #Set initial conditions
    y0=[y0, 0.0] 
    t=np.linspace(0.0, 2*np.pi*T, 1000*T) 
    
    #integrate the system over the time specified in t    
    y=scipy.integrate.odeint(ODE,y0,t,args=(g,l,Omega,q,F,))
    
    #Calculates total energy by summing kintetic and potential energies
    KE=np.square(y[:,1])
    PE=np.square(y[:,0])
    Energy=(KE+PE)/2 
    Energy_frac=Energy/Energy[0] #Divides by the inital value to get fraction
    
    #Make a straight-line fit of the plot
    coeffs=np.polyfit(t/tau,Energy_frac,1)
    z=coeffs[0]*t/tau+coeffs[1]
    
    #Plot the results
    fig, ax1 = plt.subplots()    
    ax1.plot(t/tau,Energy_frac,lw=1.0,label="Energy", color="red")
    ax1.plot(t/tau,z,lw=1.0,label="Straight line fit", color="blue")
    ax1.set_xlabel("Number of oscillations")
    ax1.set_ylabel("Energy fraction")
    ax1.set_title(r"Change in energy with number of oscillations $\theta_0$ = "+str(y0[0]))
    ax1.legend(loc="upper right")
    plot_name="Task_2_Energy_Conservation.eps"
    plt.savefig(plot_name,bbox_inches="tight",transparent=True)
    print("The best fit for the line is y = " + str("{:8.4g}".format(coeffs[0])) + "x +" + str("{:8.4g}".format(coeffs[1])))
    plt.show()
    
def Amplitude_Period(ymax,N_amp,N_period,q,F):
    '''
    Plots a graph of oscillation period against amplitude, where amplitude 
    varies from 0 to ymax with N_amp points. The period is calculated by taking
    twice the average of the time taken for the angular velocity to cross 
    the origin N_period times.
    '''
    #Choose a large number of natural oscillations to ensure 
    #N_period oscillations contained within range.
    T=3*N_period 
    t=np.linspace(0.0, 2*np.pi*T, 1000*T)
    y_init=np.linspace(0.01,ymax-0.01,N_amp) #0 and ymax are not included
    
    #Calulates the periods of the pendulum for an array of inital conditions
    #Periods stored as an array
    period_list=[]
    for i in y_init:
        y0=[i,0.0]
        y=scipy.integrate.odeint(ODE,y0,t,args=(g,l,Omega,q,F,))
        
        #Counting periods
        time_indices=np.where(np.diff(np.sign(y[:,1])))
        times=t[time_indices]-t[time_indices][0]
        period=2*np.average(np.diff(times))
        period_list.append(period)
    period_list=np.array(period_list) #Turn period_list into a numpy array
    
    #Plot the results
    fig, ax1 = plt.subplots()    
    ax1.plot(y_init,period_list/tau,lw=3.4, color="red")
    ax1.set_xlabel(r"$\theta_0$ /rad")
    ax1.set_ylabel("Period/2$\pi$")
    title_name="Amplitude-Period plot for q = "+str(q)+" F = "+str(F)
    ax1.set_title(title_name)
    ax1.legend(loc="upper right")
    plot_name="Task_2_Amplitude_Period.eps"
    plt.savefig(plot_name,bbox_inches="tight",transparent=True)
    plt.show()
    
    #Directly calculate the period for pi/2 inital amplitude
    y0=[np.pi/2,0.0]
    y=scipy.integrate.odeint(ODE,y0,t,args=(g,l,Omega,q,F,))
    time_indices=np.where(np.diff(np.sign(y[:,1])))
    times=t[time_indices]-t[time_indices][0]
    period=2*np.average(np.diff(times))
    print("The period of oscillation when the initial amplitude is pi/2 is "+str(period))

def Compare_ODE(y0,y1,T,q,F,R):
    '''
    Compares the result of the ODE integrator for two different initial amplitudes
    y0 and y1; here R*T points are plotted to illustrate the divergence/convergence
    of the solutions. R*T points and T periods are plotted.
    '''
    #Set initial conditions
    y0=[y0, 0.0] 
    y1=[y1, 0.0]
    t=np.linspace(0.0, tau*T, R*T)
    
    #Integrate the system for the two sets of initial conditions
    y=scipy.integrate.odeint(ODE,y0,t,args=(g,l,Omega,q,F,))
    z=scipy.integrate.odeint(ODE,y1,t,args=(g,l,Omega,q,F,))
    
    #Plot the results
    fig=plt.figure()
    ax1=fig.add_subplot(211)
    ax1.plot(t/tau,y[:,0],lw=1.6,label=r"$\theta_0$ = "+str(y0[0]), color="red")
    ax1.plot(t/tau,z[:,0]*np.cos(t),lw=1.6,label=r"$\theta_0$ = "+str(y1[0]), color="blue")
    ax1.set_xlabel("Number of oscillations")
    ax1.set_ylabel("Displacement/rad")
    title_name="Investigation of Butterfly Effect for q = "+str(q)+" F = "+str(F)
    ax1.set_title(title_name)
    ax1.legend(loc="lower right")    
    ax2=fig.add_subplot(212)
    ax2.plot(t/tau,y[:,1],lw=1.6,label=r"$\theta_0$ = "+str(y0[0]), color="red")
    ax2.plot(t/tau,z[:,1]*np.cos(t),lw=1.6,label=r"$\theta_0$ = "+str(y1[0]), color="blue")
    ax2.set_xlabel("Number of oscillations")
    ax2.set_ylabel("Angular Velocity/rad$s^{-1}$")
    ax2.legend(loc="lower right")
    
    plot_name="Task_2_Butterfly.eps"
    plt.savefig(plot_name,bbox_inches="tight",transparent=True)
    plt.show()
    
def Chaos(y0,y1,T,q,F,R):
    '''
    Compares the result of the ODE integrator for two different initial amplitudes
    y0 and y1; here R*T points are plotted to illustrate the divergence/convergence
    of the solutions. 
    '''
    #Set initial conditions
    y0=[y0, 0.0] 
    y1=[y1, 0.0]
    t=np.linspace(0.0, tau*T, R*T) 
    
    #Integrate the system and store the results as a 4x4 array
    y=np.zeros([4,4,R*T,2])
    z=np.zeros([4,4,R*T,2])
    q_count=0
    for q_i in q:
        F_count=0
        for F_i in F:
            y[q_count,F_count]=scipy.integrate.odeint(ODE,y0,t,args=(g,l,Omega,q_i,F_i,))
            z[q_count,F_count]=scipy.integrate.odeint(ODE,y1,t,args=(g,l,Omega,q_i,F_i,))
            F_count+=1
        q_count+=1
        
    #Plot the results as a 4x4 grid of subplots
    fig=plt.figure()
    for i in range(len(q)):
        for j in range(len(F)):
            ax=fig.add_subplot(len(q),len(F),1+j+i*len(F))
            ax.plot(y[i,j,:,0],y[i,j,:,1],lw=1.6,label=r"$\theta_0$ = "+str(y0[0]), color="red")
            ax.plot(z[i,j,:,0],z[i,j,:,1],lw=1.6,label=r"$\theta_0$ = "+str(y1[0]), color="blue")
            ax.tick_params(axis='x', labelsize=6)
            ax.tick_params(axis='y', labelsize=6)
            
            #Show labels on top and left hand side for values of q abd F
            if i==0:
                ax.set_title("F = "+str(F[j]))
            if j==0:
                ax.set_ylabel("q = "+str(q[i]))
    ax.legend(loc='upper center', bbox_to_anchor=(-1,-0.25))
    plt.suptitle("Angular Velocity vs Angle for a variety of q and F",fontsize=17)
    plot_name="Task_2_Chaos.eps"
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    plt.savefig(plot_name,bbox_inches="tight",transparent=True)
    plt.show()

#Core Task 1
Test_ODE(0.01,10,0,0,0,5)
Test_ODE(0.01,100,0,0,50,5)
Test_ODE(0.01,1000,0,0,500,5)

Energy_Conservation(0.01,10000,0,0)

Amplitude_Period(np.pi,100,100,0,0)

Energy_Conservation(0.1,1000,0,0)
Energy_Conservation(0.2,1000,0,0)
Energy_Conservation(np.pi,1000,0,0)

#Core Task 2
Test_ODE(0.01,10,0.5,0,0,5)
Test_ODE(0.01,10,1,0,0,5)
Test_ODE(0.01,10,10,0,0,5)

F=[0.5, 1.2, 1.44, 1.465]
for f in F:
    Test_ODE(0.01,10,0.5,f,0,10)
    Amplitude_Period(np.pi,100,100,0.5,f)

#Supplementary Task 1
Compare_ODE(0.2,0.20001,10000,0.5,1.2,1)


#Supplementary Task 2
q=[0, 0.1, 0.24, 0.5]
F=[0, 0.5, 1.2, 1.44]
Chaos(0.1,0.3,40,q,F,100)