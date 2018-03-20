# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 15:31:56 2017

@author: user
"""
# Solve the ODE for a conducting ring spinning in a magnetic field.
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
# This function evaluates the derivatives for the equation
# d^2 theta/dt^2 = - (2/tau) * sin^2(theta) * d theta/dt
# We work in the transformed variables y[0] = theta, y[1] = d(theta)/dt

def derivatives(y,t,tau):
    return [y[1], -(2.0/tau)*np.sin(y[0])**2*y[1]]


# Main code starts here
t=np.linspace(0.0, 20.0, 200)
y0=[0.0, 10.0]
tau=2.0
y=scipy.integrate.odeint(derivatives,y0,t,args=(tau,))

for i in range(len(y)):
    print("{:8.4g} {:8.4g} {:8.4g}".format(t[i],y[i,0],y[i,1]))


# Plot angular speed and analytical approximation
fig, ax1 = plt.subplots()
ax1.plot(t,y[:,1],lw=2,label="angular speed (rad/s)")
ax1.plot(t,10*np.exp(-t/2.0),lw=2,label="exponential")
ax1.set_xlabel("Time/s")
ax1.set_ylabel("Angular Speed")
ax1.set_title("Evolution of spinning ring in a magnetic field")
ax1.legend(loc="lower right")
# Angular position plotted on same plot with second set of axes
ax2=ax1.twinx()
ax2.plot(t,y[:,0],lw=2,label="angle (rad)",color="red")
ax2.set_ylabel("Position")
ax2.legend(loc="upper right")
plt.savefig("ode_ring1_scipy.pdf",bbox_inches="tight",transparent=True)
plt.show()