import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

# Spatial grid definition
N = 5120 # number of grid points
L = 1.0  # lenght of spatial grid
dx = np.float64(L/N) # uniform grid
Tmax = 10 #maximum time of iteraction

# init array
rho = np.ones(N, np.float64) # density
v = np.zeros(N, np.float64) #velocity
eps =np.zeros(N, np.float64) #energy density
p = np.zeros(N, np.float64) #pressure
rhoN = np.zeros(N, np.float64)
vN = np.zeros(N, np.float64)
epsN = np.zeros(N, np.float64)

gamma = 5./3. # ratio of specific heats at constant pressure and volume
rho[:] = 1.
v[:] = 0.
Eblast = 1.0 #blast energy
vol = 20*dx #volume of blast domain, tipically between 10*dx and L/2
p[:] = 1e-4
p[1:21] = (gamma-1.0)*(Eblast/vol) #high pressure in the first cells
eps[:] = p/(gamma-1) + 0.5*rho*v**2

fig=plt.figure()
frames = []
frame1 = plt.gca()

t=0
dt=0
iter=0
E = []
M = []
time=[]

with open("dat.txt","w") as external_file:
    while t < Tmax:
        
        # plot the pressure every 20 iterations
        if iter%20==0:
            plt.xlabel('Shock radius')
            plt.ylabel('Density')
            plt.ylim(0,4)
            #HERE YOU CAN DECIDE WHAT TO PLOT CHANGING FIRST ARGUMENT
            prho,=plt.plot(rho,'k')
            frames.append( [prho] )
        
        # computing dt in order to be sure that the solution is stable
        c = np.sqrt(np.max(gamma*p/rho)) # max sound speed
        CFL = 0.4 #Courant number (MUST BE MINUS THAT 1)
        dt = CFL*dx/c #time interval
        
        # LAX-FRIEDRICHS method to solve 1D-hydro equations
        rhoN[1:-1] = 0.5*(rho[2:]+rho[:-2]) - 0.5*(dt/dx)*(rho[2:]*v[2:]-rho[:-2]*v[:-2])
        
        vN[1:-1] = 0.5*(rho[2:]*v[2:]+rho[:-2]*v[:-2]) -\
                   0.5*(dt/dx)*(rho[2:]*v[2:]**2 - rho[:-2]*v[:-2]**2) -\
                   0.5*(dt/dx)*(p[2:]-p[:-2])
        
        vN[1:-1] /= rhoN[1:-1]
        
        epsN[1:-1] = 0.5*(eps[2:]+eps[:-2]) -\
                     0.5*(dt/dx)*((eps[2:]+p[2:])*v[2:] - (eps[:-2]+p[:-2])*v[:-2])
            
        #boundary in x=0 (outflow)
        rhoN[0] = rhoN[1]
        vN[0] = vN[1]
        epsN[0] = epsN[1]
        
        #boundary in x=N (outflow)
        rhoN[-1] = rhoN[-2]
        vN[-1] = vN[-2]
        epsN[-1] = epsN[-2]

        # computing pressure
        p[1:-1] = (gamma-1.)*(epsN[1:-1]-0.5*rhoN[1:-1]*vN[1:-1]**2)

        #renaming variables
        rho[:] = rhoN[:]
        v[:] = vN[:]
        eps[:] = epsN[:]
        
        # printing on an external file t and rho values
        print('iteraction:',iter,'t=',t,'dt=',dt,'rhomin=',np.min(rho),'rhomax=',np.max(rho),'vmin=',np.min(v),'vmax=',np.max(v),file=external_file)
        t+=dt
        iter +=1
    external_file.close()
    
anim = animation.ArtistAnimation(fig, frames)

#save animation
anim.save('density_CFL0.4_300dpi.gif', fps=6, dpi=300)
plt.show()

