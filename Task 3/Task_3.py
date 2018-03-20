# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 15:31:56 2017

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker

#Defining constants - permeability and coil current
mu_0 = 1E-7*4*np.pi
I=1/mu_0


#class OOMFormatter(matplotlib.ticker.ScalarFormatter):
#    '''This is here only for aesthetic purposes, so that the colour maps
#    in my contour plots can have scientific notation. Not important!'''
#    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
#        self.oom = order
#        self.fformat = fformat
#        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
#    def _set_orderOfMagnitude(self, nothing):
#        self.orderOfMagnitude = self.oom
#    def _set_format(self, vmin, vmax):
#        self.format = self.fformat
#        if self._useMathText:
#            self.format = '$%s$' % matplotlib.ticker._mathdefault(self.format)

def Generate_2Dpoints(x_num,x_distance,y_num,y_distance):
    '''Generates an (x_num+1,y_num+1,3)-shaped array that stores the vector
    positions of every points in '''
    x_axis = np.linspace(-x_distance, x_distance, x_num+1)
    x_vector = np.array([1,0,0])
    #Generate a (x_num+1,3) array of vector points
    R=x_axis[np.newaxis,:].T*x_vector[np.newaxis,:]
    
    Z=R
    #Use a for-loop to generate a ((x_num+1)*(y_num+1)+(x_num+1),3) 
    #array of points
    for i in np.linspace(-y_distance,y_distance,y_num+1):
        y_axis = i*np.ones(x_num+1)
        y_vector=np.array([0,1,0])
        y_values=y_axis[np.newaxis,:].T*y_vector[np.newaxis,:]
        Z=np.append(Z,R+y_values,axis=0)    
    #Remove extra initial term so we have a (x_num+1)*(y_num+1),3) 
    #array of points
    return Z[x_num+1:len(Z):1] 


def Line_element(x1,x2,R):
    '''Returns the value of the magnetic field at points in matrix R due 
    to a line element that runs from vector positions x1 to x2.'''
    
    #Length vector = dl vector in Biot-Savart law
    length_vector=x2-x1
    
    #Calculate mid_point vector of element so we can calculate vector 
    #distance r from element to calculation point
    mid_point=(x1+x2)/2
    r=mid_point-R
    
    #Calculation of B-field using vectorised numpy commands for the cross product
    #np.cross, as well as np.linalg.norm for calculating the magntitude of r
    B=mu_0*I/(4*np.pi*np.linalg.norm(r,axis=-1)[np.newaxis,:].T**3)*np.cross(length_vector,r)

    return B

def Coil(x,N,R,a):
    '''Calculates the value of the magnetic field at points in matrix R due
    to a coil of radius "a" centered at vector position x by treating the circular coil
    as N line elements symmetrically placed around the mid-point of the coil
    such that the total lengths of the elements equal the circumference of the 
    coil.'''
    
    Length=2*np.pi*a/N
    
    #Define a rotation matrix for calculating positions of mid-points of line elements
    theta = 2*np.pi/N
    c, s = np.cos(theta), np.sin(theta)    
    Z=np.zeros((N,3))
    Z[0]=x+np.array([0,0,a])
    for i in range(1,N):
        Z[i][0]=Z[0][0]
        Z[i][1]=c*Z[i-1][1]-s*Z[i-1][2]
        Z[i][2]=s*Z[i-1][1]+c*Z[i-1][2]

    T=np.zeros((N,3))
    T[0]=np.array([0,1,0]) #Defining the initial line element length unit vector
    for i in range(1,N):
        T[i][0]=T[0][0]
        T[i][1]=c*T[i-1][1]-s*T[i-1][2]
        T[i][2]=s*T[i-1][1]+c*T[i-1][2]    

    #Vector positions of line elements 
    x1=Z-T*Length/2
    x2=Z+T*Length/2
    
    B=np.zeros_like(R)
    #Call Line_element function to calculate magnetic field values
    for i in range(N):
        B+=Line_element(x1[i],x2[i],R)
    return B

def Place_Coils(x,a,distance,num_segments,num_coils,Z,x_num,y_num):
    '''Returns the vector B field values at the points in matrix Z from num_coils
    coils equally spaced within a length 'distance'. Achieves this by calling the
    Coil function at the specified coil positions using a for-loop.'''
    #Calls the coil function for the first coil at vector position x
    B=Coil(x,num_segments,Z,a)
    #When there is more than 1 coil, call the coil function num_coils-1 times
    #with the coils evenly spaced from x to x-distance
    if num_coils!=1:
        for j in range(1,num_coils):
            y=x-distance*j*np.array([1,0,0])/(num_coils-1)
            B+=Coil(y,num_segments,Z,a) #Fields add vectorially 
    #Return the resultant field as a (3,y_num+1,x_num+1) array where
    #the 3 rows along the 0-axis correspond to the (x,y,z) components of B
    #within the 2D grid
    return np.reshape(B.T,(3,y_num+1,x_num+1))  

def Generate_Grid(x_num,x_distance,y_num,y_distance):
    '''Generates an (Y,X) grid of points uniformly spaced in
    (-y_distance,y_distance) and (-x_distance,x_distance). Uses the np.mgrid
    numpy function.'''
    #Because of the way the B-field data is generated as well as the way
    #that np.mgrid works, it is easier to generate a (Y,X) grid rather than
    #(X,Y)
    Y, X = np.mgrid[-y_num/2:y_num/2+1, -x_num/2:x_num/2+1]
    #Now convert the grid to distances
    Y=Y*2*y_distance/y_num
    X=X*2*x_distance/x_num

    return Y,X    

def Task_1(x,a,num_segments,x_num,x_distance,y_num,y_distance):
    '''This function plots the B-field on axis from -x_distance to x_distance
    for a single coil with "num_segments" of line elements, comparing the 
    amplitude and percentage difference with the theoretical result using 
    a plot with 1000 points. A vector arrow plot of the B-field within a 
    2D grid of (y_num+1,x_num+1) points uniformly spaced about a rectange 
    of size (2*y_distance,2*x_distance) is made using the quiver function.'''
    
    #Generate data points along x-axis with y=0
    x_values = np.linspace(x[0]-x_distance, x[0]+x_distance, 1000)
    y_values = np.array([1,0,0])
    R=x_values[np.newaxis,:].T*y_values[np.newaxis,:]
    
    #Calculate B-field values by calling Coil on points in R
    B=Coil(x,num_segments,R,a)
    
    #Take transpose to separate (x,y,z) components of B-field
    B_plot=B.T 
    
    #Plot the x-component of B against the x-axis
    plt.plot(R.T[0],B_plot[0],lw=6,label='Simulation')
    #Generate the theoretical values of the B-field
    Theory=mu_0*I*a**2/(2*(a**2+R.T[0]**2)**(3/2))
    plt.plot(R.T[0],Theory,lw=2.4,label='Theory')
    #Configure the plot with appropriate title and axis labels
    plt.legend(loc='upper right')
    plt.suptitle("Theoretical and Simulated magnitudes of B field on axis")
    plt.title('For N='+str(num_segments)+" segments",fontsize=10,ha='center')
    plt.xlabel("x/m")
    plt.ylabel("B/T")
    plt.savefig("On_Axis_magnitude.eps")
    plt.show()
    
    #Plot the % difference between the analytical and simulated field strengths
    plt.plot(R.T[0],(Theory-B_plot[0])/Theory, lw=2.4,label="Difference")
    #Configure the plot with appropriate title and axis labels
    plt.suptitle("% Difference between simulated and theoretical values of B field on axis")
    plt.title('For N='+str(num_segments)+" segments",fontsize=10,ha='center')
    plt.xlabel("x/m")
    plt.ylabel(r"%$\frac{\Delta B}{B}$")
    plt.savefig("On_Axis_diff.eps")
    plt.show()    
    
    #Use Place_Coils function to calculate the B-field due to a 
    #single coil at x on an array of points Z
    Z=Generate_2Dpoints(x_num,x_distance,y_num,y_distance)
    B=Place_Coils(x,a,0,num_segments,1,Z,x_num,y_num)
    #Generate 2D grid for vector plot
    Y,X=Generate_Grid(x_num,x_distance,y_num,y_distance)
    
    #Makes a vector arrow plot of the B-field in the x-y plane
    plt.quiver(X,Y,B[0],B[1],pivot='mid')
    #Plots the location and current direction of the coil
    plt.plot(0,1,'ro')
    plt.plot(0,-1,'rx')
    #Configure the plot with appropriate title and axis labels
    plt.suptitle('Magnetic field vector plot')
    plt.title('for N='+str(num_segments)+' segments')
    plt.xlabel("x/m")
    plt.ylabel("y/m")
    plt.savefig("Task_1_vector.eps")
    plt.show()
    
def Task_2(x,a,num_segments,x_num,x_distance,y_num,y_distance):
    '''This function calculates the B-field within a 2D grid of 
    (y_num+1,x_num+1) points uniformly spaced about a rectange of 
    size (2*y_distance,2*x_distance), using the Place_Coil function to
    place two coils at x and x-a. A vector arrow plot of the field magnitude
    is made using quiver, whie a contour plot of the % difference of the field
    from that at the origin was made using contourf.'''    

    #Use Place_Coils function to calculate the B-field due to a 
    #single coil at x on an array of points Z    
    Z=Generate_2Dpoints(x_num,x_distance,y_num,y_distance)
    B=Place_Coils(x,a,a,num_segments,2,Z,x_num,y_num)

    #Generate 2D grid for vector plot - multiply by 100 to convert to cm
    Y, X =Generate_Grid(x_num,x_distance,y_num,y_distance)
    Y*=100
    X*=100
    
    #Makes a vector arrow plot of the B-field in the x-y plane    
    plt.quiver(X,Y,B[0],B[1],pivot='mid')
    #Configure the plot with appropriate title and axis labels
    plt.suptitle('Magnetic field vector plot for Helmholtz Coil')
    plt.title('using N='+str(num_segments)+' segments')
    plt.xlabel("x/cm")
    plt.ylabel("y/cm")
    plt.ylim([-y_distance*1.1*100,y_distance*1.1*100])
    plt.xlim([-x_distance*1.1*100,x_distance*1.1*100])
    plt.savefig("Task_2_vector.eps")
    plt.show()

    #Calculate mid points of x and y to calculate the B field strength at the 
    #origin of the system, and calculate the % deviance of the field within 
    #the grid from the origin value
    mid_x=int(x_num/2)
    mid_y=int(x_num/2)    
    Origin=np.sqrt(B[0][mid_y][mid_x]**2+B[1][mid_y][mid_x]**2)
    Deviance=(np.sqrt(B[0]**2+B[1]**2)-Origin)*100/Origin
 
    #Makes a contour plot of the % deviance with a nice colormap
    plt.contourf(X,Y,Deviance,cmap='jet')
    #Configure the plot with appropriate title and axis labels
    plt.suptitle('Helmholtz Coil contour plot of B-field')
    plt.title('using N='+str(num_segments)+' segments')
    plt.xlabel("x/cm")
    plt.ylabel("y/cm")
    #Configure colorbar of contour plot using function found online
#    cbar=plt.colorbar(format=OOMFormatter(-4,mathText=True)) 
    cbar=plt.colorbar()
    cbar.set_label(r'%$\Delta$B from B(0,0,0) ', rotation=270)
    plt.savefig("Task_2_contour.eps",bbox_inches = 'tight')
    plt.show()
    
def Supplementary(x,a,num_segments,num_coils,x_num,x_distance,y_num,y_distance):
    '''This function calculates the B-field within a 2D grid of 
    (y_num+1,x_num+1) points uniformly spaced about a rectange of 
    size (2*y_distance,2*x_distance), using the Place_Coil function to
    place "num_coils" coils at uniformly spaced from x to x-10*a. A vector 
    arrow plot of the field magnitude is made using quiver, whie a contour 
    plot of the % difference of the field from that at the origin was made 
    using contourf. The % difference between the on-axis field strength and
    that of an ideal theoretical solenoid is also plotted.'''    

    #Use Place_Coils function to calculate the B-field due to a 
    #single coil at x on an array of points Z        
    Z=Generate_2Dpoints(x_num,x_distance,y_num,y_distance)
    B=Place_Coils(x,a,10*a,num_segments,num_coils,Z,x_num,y_num)

    #Generate 2D grid for vector plot - multiply by 100 to convert to cm
    Y,X=Generate_Grid(x_num,x_distance,y_num,y_distance)

    #Makes a vector arrow plot of the B-field in the x-y plane    
    plt.quiver(X,Y,B[0],B[1],pivot='mid')
    
    #Plots the location and current direction of the coils
    coil_x=np.arange(-5,6,10/(num_coils-1))
    coil_y=np.ones_like(coil_x)
    plt.plot(coil_x,coil_y,'ro')
    plt.plot(coil_x,-coil_y,'rx')
    
    #Configure the plot with appropriate title, axis labels etc
    plt.suptitle('Magnetic field vector plot for N='+str(num_coils)+' coils')
    plt.title('using N='+str(num_segments)+' segments')
    plt.xlabel("x/m")
    plt.ylabel("y/m")
    plt.ylim([-1.1,1.1])
    plt.xlim([-5.4,5.4])
    plt.savefig("Supplementary_vector.eps")
    plt.show()    
    
    #Calculate mid points of x and y to calculate the B field strength at the 
    #origin of the system, and calculate the % deviance of the field within 
    #the grid from the origin value    
    mid_x=int(x_num/2)
    mid_y=int(x_num/2)
    Origin=np.sqrt(B[0][mid_y][mid_x]**2+B[1][mid_y][mid_x]**2)
    Deviance=(np.sqrt(B[0]**2+B[1]**2)-Origin)*100/Origin

    #Makes a contour plot of the % deviance with a nice colormap    
    plt.contourf(X,Y,Deviance,cmap='jet')
    #Configure the plot with appropriate title, axis labels etc
    plt.suptitle('Contour plot of B-field for N='+str(num_coils)+' coils')
    plt.title('using N='+str(num_segments)+' segments')
    plt.xlabel("x/m")
    plt.ylabel("y/m")
    #Configure colorbar of contour plot using function found online
#    cbar=plt.colorbar(format=OOMFormatter(0,mathText=True)) 
    cbar=plt.colorbar()
    cbar.set_label(r'%$\Delta$B from B(0,0,0) ', rotation=270)
    plt.savefig("Supplementary_contour.eps",bbox_inches = 'tight')
    plt.show()    
    
    #Generate data points along x-axis with y=0
    x_values = np.linspace(-x_distance, +x_distance, 1001)
    y_values = np.array([1,0,0])
    R=x_values[np.newaxis,:].T*y_values[np.newaxis,:]
    
    #Calculate B-field values by calling Coil on points in R
    #using a for-loop to include the contribution of all the coils
    B=Coil(x,num_segments,R,a)
    for j in range(1,num_coils):
        y=x-10*a*j*np.array([1,0,0])/(num_coils-1)
        B+=Coil(y,num_segments,R,a)
    B_plot=B.T
    
    #Calculates the theoretical field strength for a solenoid and plots the %
    #deviance between the two along the x-axis - the origin is marked with a 
    #red dot
    Theory=mu_0*I*num_coils/(10*a)
    plt.plot(R.T[0],(Theory-B_plot[0])*100/Theory, lw=2.4,label="Difference")
    plt.plot(0,0,'ro')
    #Configure the plot with appropriate title, axis labels etc
    plt.suptitle("% Difference between simulated and theoretical values of B field on axis")
    plt.title('For N='+str(num_segments)+" segments",fontsize=10,ha='center')
    plt.xlabel("x/m")
    plt.ylabel(r"%$\frac{\Delta B}{B}$")
    plt.savefig("Supplementary_compare.eps")
    plt.show()
    
x=np.array([0,0,0])
Task_1(x,1,32,10,0.8,10,0.8)

x=np.array([0.5,0,0])
Task_2(x,1,32,50,0.05,50,0.05)

x=np.array([5,0,0])
Supplementary(x,1,32,8,20,5,20,0.8)
Supplementary(x,1,32,32,20,5,20,0.8)