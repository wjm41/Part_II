# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 15:34:52 2017

@author: wjm41
"""

"""Task 1 - Integration and Random Numbers"""

import numpy as np
import matplotlib.pyplot as plt

def Error(Array,N):
    '''Calculates the standard deviation of N Monte-Carlo estimates.'''
    mean_squared=np.sum(np.multiply(Array,Array))/N
    SD=np.sqrt((mean_squared-np.mean(Array)**2)/N)
    return SD
    
def Monte_Carlo(N):
    '''Returns the Monte-Carlo estimate of the value of the integral and its
    theoretical error for N sample points.'''
    #vectorised code for calculating the value of the integrand
    func=np.sin(np.sum(np.random.rand(N,8)*np.pi/8, axis=1)) 
    mean=np.mean(func)
    SD=Error(func,N)
    Volume=(np.pi/8)**8
    return 1e6*Volume*mean,1e6*Volume*SD


def Generate_Estimates(N_start=100,N_end=1000,num_points=25,num_int=25):
    '''Using num_points from N_start to N_end, generates the standard deviation
    of the Monte-Carlo estimates, the theoretical errors, and the actual error
    as (N,error) pairs. Also prints out the value and standard deviation of the 
    Monte-Carlo estimate for every N value.'''
    Measured=[]
    Theory=[]
    Actual=[]
    N_values=np.logspace(N_start,N_end,num_points)
    for j in N_values:
        Integral_value=np.empty([num_int,1])
        Theory_error=np.empty([num_int,1])
        for i in range(num_int):
            Integral_value[i],Theory_error[i]=Monte_Carlo(int(j))
        SD=Error(Integral_value,num_int)
        Measured.append([j,SD])
        Theory.append([j,np.mean(Theory_error)])
        Actual.append([j,abs(np.mean(Integral_value)-537.1873411)])
        print("The value of the integral for N="+str(j)+" is "+str(np.mean(Integral_value))+"+-"+str(SD))
    return np.array(Measured),np.array(Theory), np.array(Actual)


print("Generating Monte Carlo plot...")

#SD of Monte-Carlo values
Raw_Error,Theory_Error,Actual_Error=Generate_Estimates(3,6,25,25)
x,y=Raw_Error.T
plt.loglog(x,y)

#Curve Fit of SD
coeffs=np.polyfit(np.log(x),np.log(y),1) 
z=np.exp(coeffs[1])*np.exp(coeffs[0]*np.log(x))
print('The equation of the straight line fit is y='+str(coeffs[0])+'x + '+str(coeffs[1]))
plt.loglog(x,z)

#Theoretical Estimate of Errors
i,j=Theory_Error.T
plt.loglog(i,j)

#Actual Error
p,q,=Actual_Error.T
plt.loglog(p,q)

plt.legend(['Measured error', 'Curve fit','Theoretical error','Actual Error'], loc='upper right')
plt.xlabel("Number of Monte-Carlo sample points")
plt.ylabel("Estimate of error in integral")
plt.title("log-log plot of Monte-Carlo error against the number of sample points used")
plt.savefig("monte_carlo.eps")     
plt.show()
