import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from mpi4py import MPI
import time as tm

comm = MPI.COMM_WORLD #access to the default communicator
size = comm.Get_size() #number of process in communication
rank = comm.Get_rank() #rank of the process in communication
print(f'Task {rank} running...\n')

# spatial grid
N = 48000 # number of grid points
local_N = (N // size) + 2 # domain decomposition
L = 1.0 #length of grid
dx = np.float64(L/N) #uniform grid
Tmax = 0.1

gamma = 5./3. #ratio between specific heats at constant volume
vol = 20*dx #tipically it should have value between 10*dx and L/2
Eblast = 1.0

# initialization of local variables
local_rho = np.zeros(local_N, dtype=np.float64)
local_v = np.zeros(local_N, dtype=np.float64)
local_eps = np.zeros(local_N, dtype=np.float64)
local_p = np.zeros(local_N, dtype=np.float64)
	
# initialization of updated local variables
local_rhoN = np.ones(local_N, dtype=np.float64)
local_vN = np.zeros(local_N, dtype=np.float64)
local_epsN = np.zeros(local_N, dtype=np.float64)

local_x = np.arange(local_N-2, dtype=np.float64)*dx+rank*local_N*dx

#definition of function to initialize pres and other quantities
def init_pres(x):
    result = np.ones_like(x) * 1e-4
    condition = x < (dx * 20)
    result[condition] = (gamma - 1.0) * (Eblast / vol)
    return result

local_rho[1:-1] = 1.  # Inizializza tutti gli elementi da 1 a N-2 con il valore 1.
local_v[1:-1] = 0.  # Inizializza tutti gli elementi da 1 a N-2 con il valore 0.
local_p[1:-1] = init_pres(local_x).flatten()
local_eps[1:-1] = local_p[1:-1]/(gamma-1) + 0.5*local_rho[1:-1]*local_v[1:-1]**2  # Calcola e inizializza gli elementi da 1 a N-2.

'''
def init_pres(x):
	if x< (dx*20):
		return (gamma-1.0)*(Eblast/vol)
	else:
		return 1e-4
		
for i in range (local_N-2):
	local_rho[i+1] = 1.
	local_v[i+1] = 0.
	local_p[i+1] = init_pres(local_x[i])
	local_eps[i+1] = local_p[i+1]/(gamma-1) + 0.5*local_rho[i+1]*local_v[i+1]**2

'''	
#print('rank=',rank,'p=',local_p,'eps=',local_eps)
fig=plt.figure()
frames = []
frame1 = plt.gca()

t=0
dt=0
iter=0
global_cmax = 0
E = []
M = []
time=[]

execution_time_comm = 0
execution_time_eq = 0
execution_time_sound = 0
execution_time_total = 0

start_time_total = tm.time()
while t < Tmax:
	#plotting quantity every 20 iterations
	'''
	if iter % 100 == 0:
		# need to communicate to root zero all spatial domain of variable
		gathered_rho = comm.gather(local_rho[1:-1], root=0)
		if rank == 0:
			rho = np.concatenate(gathered_rho)  # Concatenate the gathered arrays
			x_values = np.arange(N, dtype=np.float64) * dx  # Global x-values
			plt.xlabel('Shock radius')
			plt.ylabel('Velocity')
			prho, = plt.plot(x_values, rho, 'k')  # Adjust x-values for each rank
			frames.append([prho])
	'''		
	#print('iter=',iter,'rank=',rank,'rho=',local_rho,'p=',local_p,'global_cmax=',global_cmax,'t=',t)
	
	start_time_sound = tm.time()
	# compute a stable dt (CFL 'must' be <1)
	local_c = np.sqrt(np.max(gamma*local_p[1:-1]/local_rho[1:-1])) #each core computes max speed of sound
	
	#communication to rank=0 of all local_c to determine the higher
	global_cmax = np.zeros(1,dtype=np.float64)
	comm.Reduce(local_c,global_cmax,op=MPI.MAX,root=0) #calcola il max in root=0
	
	# Only rank=0 computes the global maximum
	global_cmax = comm.bcast(global_cmax, root=0)
	CFL = 0.4
	dt = CFL*dx/global_cmax
	end_time_sound = tm.time()
	execution_time_sound += (end_time_sound - start_time_sound)
	
	
	'''
	#communications and left BC in core 0
	if rank==0:		
		comm.Send(local_rho[-2:-1],dest=rank+1,tag=10) #send the last element of array
		comm.Send(local_v[-2:-1],dest=rank+1,tag=20)
		comm.Send(local_eps[-2:-1],dest=rank+1,tag=30)
		
		comm.Recv(local_rho[-1:],source=rank+1,tag=11)
		comm.Recv(local_v[-1:],source=rank+1,tag=21)
		comm.Recv(local_eps[-1:],source=rank+1,tag=31)
		
		#boundary in x=0 (outflow)
		local_rho[0] = local_rho[1]
		local_v[0] = local_v[1]
		local_eps[0] = local_eps[1]
		
	#communications and right BC in the last core
	elif rank==size-1:
		comm.Send(local_rho[1:2],dest=rank-1,tag=11)
		comm.Send(local_v[1:2],dest=rank-1,tag=21)
		comm.Send(local_eps[1:2],dest=rank-1,tag=31)
		
		comm.Recv(local_rho[0:1],source=rank-1,tag=10)
		comm.Recv(local_v[0:1],source=rank-1,tag=20)
		comm.Recv(local_eps[0:1],source=rank-1,tag=30)
		
		#boundary in x=N (outflow)
		local_rho[-1] = local_rho[-2]
		local_v[-1] = local_v[-2]
		local_eps[-1] = local_eps[-2]
		
	#communications between other cores
	else:
		comm.Send(local_rho[-2:-1],dest=rank+1,tag=10)
		comm.Send(local_v[-2:-1],dest=rank+1,tag=20)
		comm.Send(local_eps[-2:1],dest=rank+1,tag=30)
		
		comm.Send(local_rho[1:2],dest=rank-1,tag=11)
		comm.Send(local_v[1:2],dest=rank-1,tag=21)
		comm.Send(local_eps[1:2],dest=rank-1,tag=31)
		
		comm.Recv(local_rho[-1:],source=rank+1,tag=11)
		comm.Recv(local_v[-1:],source=rank+1,tag=21)
		comm.Recv(local_eps[-1:],source=rank+1,tag=31)
		
		comm.Recv(local_rho[0:1],source=rank-1,tag=10)
		comm.Recv(local_v[0:1],source=rank-1,tag=20)
		comm.Recv(local_eps[0:1],source=rank-1,tag=30)
	'''		
	
	if rank == 0:
		#boundary in x=0 (outflow)
		local_rho[0] = local_rho[1]
		local_v[0] = local_v[1]
		local_eps[0] = local_eps[1]
		
	if rank == size-1:
		#boundary in x=N (outflow)
		local_rho[-1] = local_rho[-2]
		local_v[-1] = local_v[-2]
		local_eps[-1] = local_eps[-2]
	

	start_time_comm = tm.time()
	if rank%2 == 0:
		#print('iter=',iter,'rank=',rank,'local_rho[1:2]=',local_rho[1:2])
		if rank>0:
			comm.Send(local_rho[1:2],dest=rank-1,tag=11)
			comm.Send(local_v[1:2],dest=rank-1,tag=21)
			comm.Send(local_eps[1:2],dest=rank-1,tag=31)
			comm.Send(local_p[1:2],dest=rank-1,tag=41)
		if rank<size-1:			
			comm.Send(local_rho[-2:-1],dest=rank+1,tag=10)
			comm.Send(local_v[-2:-1],dest=rank+1,tag=20)
			comm.Send(local_eps[-2:-1],dest=rank+1,tag=30)
			comm.Send(local_p[-2:-1],dest=rank+1,tag=40)
	else:
		if rank>0:
			comm.Recv(local_rho[0:1],source=rank-1,tag=10)
			comm.Recv(local_v[0:1],source=rank-1,tag=20)
			comm.Recv(local_eps[0:1],source=rank-1,tag=30)
			comm.Recv(local_p[0:1],source=rank-1,tag=40)
		if rank<size-1:
			comm.Recv(local_rho[-1:],source=rank+1,tag=11)
			comm.Recv(local_v[-1:],source=rank+1,tag=21)
			comm.Recv(local_eps[-1:],source=rank+1,tag=31)
			comm.Recv(local_p[-1:],source=rank+1,tag=41)
			
	if rank%2 !=0:
		if rank >0:			
			comm.Send(local_rho[1:2],dest=rank-1,tag=110)
			comm.Send(local_v[1:2],dest=rank-1,tag=210)
			comm.Send(local_eps[1:2],dest=rank-1,tag=310)
			comm.Send(local_p[1:2],dest=rank-1,tag=410)
		if rank<size-1:
			comm.Send(local_rho[-2:-1],dest=rank+1,tag=100)
			comm.Send(local_v[-2:-1],dest=rank+1,tag=200)
			comm.Send(local_eps[-2:-1],dest=rank+1,tag=300)
			comm.Send(local_p[-2:-1],dest=rank+1,tag=400)
	else:
		if rank>0:
			comm.Recv(local_rho[0:1],source=rank-1,tag=100)
			comm.Recv(local_v[0:1],source=rank-1,tag=200)
			comm.Recv(local_eps[0:1],source=rank-1,tag=300)
			comm.Recv(local_p[0:1],source=rank-1,tag=400)
		if rank<size-1:
			comm.Recv(local_rho[-1:],source=rank+1,tag=110)
			comm.Recv(local_v[-1:],source=rank+1,tag=210)
			comm.Recv(local_eps[-1:],source=rank+1,tag=310)
			comm.Recv(local_p[-1:],source=rank+1,tag=410)

	end_time_comm = tm.time()
	execution_time_comm += (end_time_comm - start_time_comm)
	
	start_time_eq = tm.time()
	#solve 1D Hydrodynamic equations with Lax-Friedrich method
	local_rhoN[1:-1] = 0.5*(local_rho[2:]+local_rho[:-2]) - 0.5*(dt/dx)*(local_rho[2:]*local_v[2:]-local_rho[:-2]*local_v[:-2])
	
	local_vN[1:-1] = 0.5*(local_rho[2:]*local_v[2:]+local_rho[:-2]*local_v[:-2]) - 0.5*(dt/dx)*(local_rho[2:]*local_v[2:]**2 - local_rho[:-2]*local_v[:-2]**2) - 0.5*(dt/dx)*(local_p[2:]-local_p[:-2])
	
	local_vN[1:-1] /= local_rhoN[1:-1]
	
	local_epsN[1:-1] = 0.5*(local_eps[2:]+local_eps[:-2]) - 0.5*(dt/dx)*((local_eps[2:]+local_p[2:])*local_v[2:] - (local_eps[:-2]+local_p[:-2])*local_v[:-2])
	
	local_p[1:-1] = (gamma-1.)*(local_epsN[1:-1]-0.5*local_rhoN[1:-1]*local_vN[1:-1]**2)
	end_time_eq = tm.time()
	execution_time_eq += (end_time_eq - start_time_eq)

	#rename variables
	local_rho[:] = local_rhoN[:]
	local_v[:] = local_vN[:]
	local_eps[:] = local_epsN[:]
	
	
	t+=dt #aggiorno t
	#fine iterazione numerica
	iter +=1
	
end_time_total = tm.time()
execution_time_total += (end_time_total - start_time_total)

print(f"N={N},{rank},{execution_time_eq} eq, {execution_time_comm} comm, {execution_time_sound} sound, {execution_time_total} TOT")

#anim = animation.ArtistAnimation(fig, frames)
#plt.savefig('density_CFL0.4_paralla_density.png',dpi=1200)
#anim.save('animazione_paralla_density.gif', fps=12, dpi=72)

#anim.save('Lauriano_new_density_scaled_CFL0.4_300dpi_symmetric_14-12-23.gif', fps=6, dpi=300)
#plt.show()

