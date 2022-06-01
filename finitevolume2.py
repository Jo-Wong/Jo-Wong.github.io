import matplotlib.pyplot as plt
import numpy as np

"""
Create Your Own Finite Volume Fluid Simulation (With Python) Part 2: 
Boundary Conditions and Source Terms
Philip Mocz (2021), @PMocz

Simulate the Raleigh-Taylor Instability with the Finite Volume Method. 
Demonstrates gravity source term and Reflecting boundary condition

"""


def getConserved( rho, vx, vy, vz, P, gamma, vol ):
	"""
    Calculate the conserved variable from the primitive
	rho      is matrix of cell densities
	vx       is matrix of cell x-velocity
	vy       is matrix of cell y-velocity
	vz		 is matrix of cell z-velocity
	P        is matrix of cell pressures
	gamma    is ideal gas gamma
	vol      is cell volume
	Mass     is matrix of mass in cells
	Momx     is matrix of x-momentum in cells
	Momy     is matrix of y-momentum in cells
	Momz	 is matrix of z-momentum in cells
	Energy   is matrix of energy in cells
	"""
	Mass   = rho * vol
	Momx   = rho * vx * vol
	Momy   = rho * vy * vol
	Momz   = rho * vz * vol
	Energy = (P/(gamma-1) + 0.5*rho*(vx**2+vy**2))*vol
	
	return Mass, Momx, Momy, Momz, Energy


def getPrimitive( Mass, Momx, Momy, Momz, Energy, gamma, vol ):
	"""
    Calculate the primitive variable from the conservative
	Mass     is matrix of mass in cells
	Momx     is matrix of x-momentum in cells
	Momy     is matrix of y-momentum in cells
	Momz 	 is matrix of z-momentum in cells
	Energy   is matrix of energy in cells
	gamma    is ideal gas gamma
	vol      is cell volume
	rho      is matrix of cell densities
	vx       is matrix of cell x-velocity
	vy       is matrix of cell y-velocity
	vz		 is matrix of cell z-velocity
	P        is matrix of cell pressures
	"""
	rho = Mass / vol
	vx  = Momx / rho / vol
	vy  = Momy / rho / vol
	vz  = Momz / rho / vol
	P   = (Energy/vol - 0.5*rho * (vx**2+vy**2)) * (gamma-1)
	
	rho, vx, vy, vz, P = setGhostCells(rho, vx, vy, vz, P)	
	
	return rho, vx, vy, vz, P


def getGradient(f, dx):
	"""
    Calculate the gradients of a field
	f        is a matrix of the field
	dx       is the cell size
	f_dx     is a matrix of derivative of f in the x-direction
	f_dy     is a matrix of derivative of f in the y-direction
	f_dz	 is a matrix of derivative of f in the z-direction
	"""
	# directions for np.roll() 
	R = -1   # right
	L = 1    # left
	
	f_dx = ( np.roll(f,R,axis=0) - np.roll(f,L,axis=0) ) / (2*dx)
	f_dy = ( np.roll(f,R,axis=1) - np.roll(f,L,axis=1) ) / (2*dx)
	f_dz = ( np.roll(f,R,axis=2) - np.roll(f,L,axis=2) ) / (2*dx)
	
	f_dx, f_dy, f_dz = setGhostGradients(f_dx, f_dy, f_dz)
	
	return f_dx, f_dy, f_dz

def setAGhostLaplacian( f_ddx, f_ddy, f_ddz ):
	# directions for np.roll() 
	R = -1   # right
	L = 1    # left

	f_ddx[0,:,:]  = np.roll(f_ddx,R,axis=0)[0,:,:] 
	f_ddx[-1,:,:] = np.roll(f_ddx,L,axis=0)[-1,:,:] 
	f_ddy[:,0,:]  = np.roll(f_ddy,R,axis=1)[:,0,:] 
	f_ddy[:,-1,:] = np.roll(f_ddy,L,axis=1)[:,-1,:]
	f_ddz[:,:,0]  = np.roll(f_ddz,R,axis=2)[:,:,0]
	f_ddz[:,:,-1] = np.roll(f_ddz,L,axis=2)[:,:,-1]

	return f_ddx, f_ddy, f_ddz 

def getALaplacian(f, dx):
	"""
	Calculate the Laplace derivative of a field
	f		 is a matrix of the field
	dx		 is the cell size
	f_dd	 is a matrix of the Laplacian of the field
	"""
	# directions for np.roll() 
	R = -1   # right
	L = 1    # left

	f_ddx = ( np.roll(f,R,axis=0) - 2*f + np.roll(f,L,axis=0) ) / (dx**2)
	f_ddy = ( np.roll(f,R,axis=1) - 2*f + np.roll(f,L,axis=1) ) / (dx**2)
	f_ddz = ( np.roll(f,R,axis=2) - 2*f + np.roll(f,L,axis=2) ) / (dx**2)

	#f_ddx, f_ddy, f_ddz = setAGhostLaplacian(f_ddx, f_ddy, f_ddz)

	f_ddx[0,:,:]  = ( f[0,:,:]  - 2*f[1,:,:]  + f[2,:,:]  ) / (dx**2)
	f_ddx[-1,:,:] = ( f[-1,:,:] - 2*f[-2,:,:] + f[-3,:,:] ) / (dx**2)
	f_ddy[:,0,:]  = ( f[:,0,:]  - 2*f[:,1,:]  + f[:,2,:]  ) / (dx**2)
	f_ddy[:,-1,:] = ( f[:,-1,:] - 2*f[:,-2,:] + f[:,-3,:] ) / (dx**2)
	#f_ddz[:,:,0]  = ( f[:,:,0]  - 2*f[:,:,1]  + f[:,:,2]  ) / (dx**2)   # assuming width_z = 0
	#f_ddz[:,:,-1] = ( f[:,:,-1] - 2*f[:,:,-2] + f[:,:,-3] ) / (dx**2)   # assuming width_z = 0

	'''
	f_dx, f_dy, f_dz = findAGrad(f, dx)

	f_ddx, _,     _     = findAGrad(f_dx, dx)
	_,     f_ddy, _     = findAGrad(f_dy, dx)
	_,     _,     f_ddz = findAGrad(f_dz, dx)
	'''

	return f_ddx + f_ddy + f_ddz


def slopeLimit(f, dx, f_dx, f_dy, f_dz):
	"""
    Apply slope limiter to slopes
	f        is a matrix of the field
	dx       is the cell size
	f_dx     is a matrix of derivative of f in the x-direction
	f_dy     is a matrix of derivative of f in the y-direction
	f_dz 	 is a matrix of derivative of f in the z-direction
	"""
	# directions for np.roll() 
	R = -1   # right
	L = 1    # left
	
	f_dx = np.maximum(0., np.minimum(1., ( (f-np.roll(f,L,axis=0))/dx)/(f_dx + 1.0e-8*(f_dx==0)))) * f_dx
	f_dx = np.maximum(0., np.minimum(1., (-(f-np.roll(f,R,axis=0))/dx)/(f_dx + 1.0e-8*(f_dx==0)))) * f_dx
	f_dy = np.maximum(0., np.minimum(1., ( (f-np.roll(f,L,axis=1))/dx)/(f_dy + 1.0e-8*(f_dy==0)))) * f_dy
	f_dy = np.maximum(0., np.minimum(1., (-(f-np.roll(f,R,axis=1))/dx)/(f_dy + 1.0e-8*(f_dy==0)))) * f_dy
	f_dz = np.maximum(0., np.minimum(1., ( (f-np.roll(f,L,axis=2))/dx)/(f_dz + 1.0e-8*(f_dz==0)))) * f_dz
	f_dz = np.maximum(0., np.minimum(1., (-(f-np.roll(f,R,axis=2))/dx)/(f_dz + 1.0e-8*(f_dz==0)))) * f_dz
	
	return f_dx, f_dy, f_dz


def extrapolateInSpaceToFace(f, f_dx, f_dy, f_dz, dx):
	"""
    Calculate the gradients of a field
	f        is a matrix of the field
	f_dx     is a matrix of the field x-derivatives
	f_dy     is a matrix of the field y-derivatives
	f_dz	 is a matrix of the field z-derivatives
	dx       is the cell size
	f_XL     is a matrix of spatial-extrapolated values on `left' face along x-axis 
	f_XR     is a matrix of spatial-extrapolated values on `right' face along x-axis 
	f_YR     is a matrix of spatial-extrapolated values on `left' face along y-axis 
	f_YR     is a matrix of spatial-extrapolated values on `right' face along y-axis 
	f_ZR     is a matrix of spatial-extrapolated values on `left' face along z-axis 
	f_ZR     is a matrix of spatial-extrapolated values on `right' face along z-axis 
	"""
	# directions for np.roll() 
	R = -1   # right
	L = 1    # left
	
	f_XL = f - f_dx * dx/2
	f_XL = np.roll(f_XL,R,axis=0)
	f_XR = f + f_dx * dx/2
	
	f_YL = f - f_dy * dx/2
	f_YL = np.roll(f_YL,R,axis=1)
	f_YR = f + f_dy * dx/2

	f_ZL = f - f_dz * dx/2
	f_ZL = np.roll(f_ZL,R,axis=2)
	f_ZR = f + f_dz * dx/2
	
	return f_XL, f_XR, f_YL, f_YR, f_ZL, f_ZR
	

def applyFluxes(F, flux_F_X, flux_F_Y, flux_F_Z, dx, dt):
	"""
    Apply fluxes to conserved variables
	F        is a matrix of the conserved variable field
	flux_F_X is a matrix of the x-dir fluxes
	flux_F_Y is a matrix of the y-dir fluxes
	flux_F_Z is a matrix of the z-dir fluxes
	dx       is the cell size
	dt       is the timestep
	"""
	# directions for np.roll() 
	R = -1   # right
	L = 1    # left
	
	# update solution
	F += - dt * dx * flux_F_X
	F +=   dt * dx * np.roll(flux_F_X,L,axis=0)
	F += - dt * dx * flux_F_Y
	F +=   dt * dx * np.roll(flux_F_Y,L,axis=1)
	F += - dt * dx * flux_F_Z
	F +=   dt * dx * np.roll(flux_F_Z,L,axis=2)
	
	return F


def getFlux(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, vz_L, vz_R, P_L, P_R, gamma):
	"""
    Calculate fluxed between 2 states with local Lax-Friedrichs/Rusanov rule 
	rho_L        is a matrix of left-state  density
	rho_R        is a matrix of right-state density
	vx_L         is a matrix of left-state  x-velocity
	vx_R         is a matrix of right-state x-velocity
	vy_L         is a matrix of left-state  y-velocity
	vy_R         is a matrix of right-state y-velocity
	vz_L	     is a matrix of left-state  z-velocity
	vz_R		 is a matrix of right-state z-velocity
	P_L          is a matrix of left-state  pressure
	P_R          is a matrix of right-state pressure
	gamma        is the ideal gas gamma
	flux_Mass    is the matrix of mass fluxes
	flux_Momx    is the matrix of x-momentum fluxes
	flux_Momy    is the matrix of y-momentum fluxes
	flux_Energy  is the matrix of energy fluxes
	"""
	
	# left and right energies
	en_L = P_L/(gamma-1)+0.5*rho_L * (vx_L**2+vy_L**2+vz_L**2)
	en_R = P_R/(gamma-1)+0.5*rho_R * (vx_R**2+vy_R**2+vz_R**2)

	# compute star (averaged) states
	rho_star  = 0.5*(rho_L + rho_R)
	momx_star = 0.5*(rho_L * vx_L + rho_R * vx_R)
	momy_star = 0.5*(rho_L * vy_L + rho_R * vy_R)
	momz_star = 0.5*(rho_L * vz_L + rho_R * vz_R)
	en_star   = 0.5*(en_L + en_R)
	
	P_star = (gamma-1)*(en_star-0.5*(momx_star**2+momy_star**2+momz_star**2)/rho_star)
	
	# compute fluxes (local Lax-Friedrichs/Rusanov)
	flux_Mass   = momx_star
	flux_Momx   = momx_star**2/rho_star + P_star
	flux_Momy   = momx_star * momy_star/rho_star
	flux_Momz	= momx_star * momz_star/rho_star
	flux_Energy = (en_star+P_star) * momx_star/rho_star
	
	# find wavespeeds
	C_L = np.sqrt(gamma*P_L/rho_L) + np.abs(vx_L)
	C_R = np.sqrt(gamma*P_R/rho_R) + np.abs(vx_R)
	C = np.maximum( C_L, C_R )
	
	# add stabilizing diffusive term
	flux_Mass   -= C * 0.5 * (rho_L - rho_R)
	flux_Momx   -= C * 0.5 * (rho_L * vx_L - rho_R * vx_R)
	flux_Momy   -= C * 0.5 * (rho_L * vy_L - rho_R * vy_R)
	flux_Momz	-= C * 0.5 * (rho_L * vz_L - rho_R * vz_R)
	flux_Energy -= C * 0.5 * ( en_L - en_R )

	return flux_Mass, flux_Momx, flux_Momy, flux_Momz, flux_Energy

def addGhostCells( rho, vx, vy, vz, Ax, Ay, Az, P ):
	"""
    Add ghost cells to the top and bottom
	rho      is matrix of cell densities
	vx       is matrix of cell x-velocity
	vy       is matrix of cell y-velocity
	vz		 is matrix of cell z-velocity
	Ax		 is matrix of cell Ax
	Ay		 is matrix of cell Ay
	Az		 is matrix of cell Az
	P        is matrix of cell pressures
	"""
	rho = np.hstack((rho[:,0:1,:], rho, rho[:,-1:,:]))
	vx  = np.hstack(( vx[:,0:1,:],  vx,  vx[:,-1:,:]))
	vy  = np.hstack(( vy[:,0:1,:],  vy,  vy[:,-1:,:]))
	vz  = np.hstack(( vz[:,0:1,:],  vz,  vz[:,-1:,:]))
	P   = np.hstack((  P[:,0:1,:],   P,   P[:,-1:,:]))

	#dif_x = Ax[:,1:2,:] - Ax[:,0:1,:]
	#dif_y = Ay[:,1:2,:] - Ay[:,0:1,:]
	#dif_z = Az[:,1:2,:] - Az[:,0:1,:]
	#Ax  = np.hstack(( Ax[:,0:1,:] - dif_x,  Ax,  Ax[:,-1:,:] + dif_x))
	#Ay  = np.hstack(( Ay[:,0:1,:] - dif_y,  Ay,  Ay[:,-1:,:] + dif_y))
	#Az  = np.hstack(( Az[:,0:1,:] - dif_z,  Az,  Az[:,-1:,:] + dif_z))

	return rho, vx, vy, vz, Ax, Ay, Az, P
	
def setGhostCells( rho, vx, vy, vz, P ):
	"""
    Set ghost cells at the top and bottom
	rho      is matrix of cell densities
	vx       is matrix of cell x-velocity
	vy       is matrix of cell y-velocity
	vz		 is matrix of cell z-velocity
	P        is matrix of cell pressures
	"""
	
	rho[:,0,:]  = rho[:,1,:]
	vx[:,0,:]   =  vx[:,1,:]
	vy[:,0,:]   = -vy[:,1,:]
	vz[:,0,:]   =  vz[:,1,:]
	P[:,0,:]    =   P[:,1,:]
	
	rho[:,-1,:] = rho[:,-2,:]
	vx[:,-1,:]  =  vx[:,-2,:]
	vy[:,-1,:]  = -vy[:,-2,:]
	vz[:,-1,:]  =  vz[:,-2,:]
	P[:,-1,:]   =   P[:,-2,:]
	
	return rho, vx, vy, vz, P
	
def setGhostGradients( f_dx, f_dy, f_dz ):
	"""
    Set ghost cell y-gradients at the top and bottom to be reflections
	f_dx     is a matrix of derivative of f in the x-direction
	f_dy     is a matrix of derivative of f in the y-direction
	f_dz	 is a matrix of derivative of f in the z-direction
	"""
	
	f_dy[:,0,:]  = -f_dy[:,1,:]  
	f_dy[:,-1,:] = -f_dy[:,-2,:] 
	
	return f_dx, f_dy, f_dz

def addSourceTerm( Mass, Momx, Momy, Momz, Bx, By, Bz, Energy, g, vol, dx, dt ):
	"""
    Add gravitational source term to conservative variables
	Mass     is matrix of mass in cells
	Momx     is matrix of x-momentum in cells
	Momy     is matrix of y-momentum in cells
	Energy   is matrix of energy in cells
	g        is strength of gravity
	Y        is matrix of y positions of cells
	dt       is timestep to progress solution
	"""
	
	Bf_x, Bf_y, Bf_z = findMagneticForce(Bx, By, Bz, vol, dx)
	#print('f: ', np.max(Bf_x), np.max(Bf_y), np.max(Bf_z))

	Energy += dt * Momy * g
	Momx += dt * Bf_x
	Momy += dt * Mass * g + dt * Bf_y
	Momz += dt * Bf_z
	
	return Mass, Momx, Momy, Momz, Energy

def setAGhostGradients( f_dx, f_dy, f_dz ):
	# directions for np.roll() 
	R = -1   # right
	L = 1    # left

	f_dx[0,:,:]  = np.roll(f_dx,R,axis=0)[0,:,:] 
	f_dx[-1,:,:] = np.roll(f_dx,L,axis=0)[-1,:,:] 
	f_dy[:,0,:]  = np.roll(f_dy,R,axis=1)[:,0,:] 
	f_dy[:,-1,:] = np.roll(f_dy,L,axis=1)[:,-1,:]
	f_dz[:,:,0]  = np.roll(f_dz,R,axis=2)[:,:,0]
	f_dz[:,:,-1] = np.roll(f_dz,L,axis=2)[:,:,-1]

	return f_dx, f_dy, f_dz

def findAGrad ( f, dx ):
	# directions for np.roll() 
	R = -1   # right
	L = 1    # left
	
	f_dx = ( np.roll(f,R,axis=0) - np.roll(f,L,axis=0) ) / (2*dx)
	f_dy = ( np.roll(f,R,axis=1) - np.roll(f,L,axis=1) ) / (2*dx)
	f_dz = ( np.roll(f,R,axis=2) - np.roll(f,L,axis=2) ) / (2*dx)
	
	#f_dx, f_dy, f_dz = setAGhostGradients(f_dx, f_dy, f_dz)

	f_dx[0,:,:]  = ( np.roll(f,R,axis=0)[0,:,:]  - f[0,:,:] )  / dx
	f_dx[-1,:,:] = ( f[-1,:,:] - np.roll(f,L,axis=0)[-1,:,:] ) / dx
	f_dy[:,0,:]  = ( np.roll(f,R,axis=1)[:,0,:]  - f[:,0,:] )  / dx
	f_dy[:,-1,:] = ( f[:,-1,:] - np.roll(f,L,axis=1)[:,-1,:] ) / dx
	f_dz[:,:,0]  = ( np.roll(f,R,axis=2)[:,:,0]  - f[:,:,0] )  / dx
	f_dz[:,:,-1] = ( f[:,:,-1] - np.roll(f,L,axis=2)[:,:,-1] ) / dx

	return f_dx, f_dy, f_dz

def findBField( Ax, Ay, Az, dx ):
	"""
	Calculate magnetic field by taking curl of vector potential
	Ax		 is matrix of x-direction of vector potential
	Ay		 is matrix of y-direction of vector potential
	Az 		 is matrix of z-direction of vector potential
	dx		 is the cell size
	"""
	# directions for np.roll() 
	R = -1   # right
	L = 1    # left

	#Ax_dx, Ax_dy, Ax_dz = getGradient(Ax, dx)
	#Ay_dx, Ay_dy, Ay_dz = getGradient(Ay, dx)
	#Az_dx, Az_dy, Az_dz = getGradient(Az, dx)

	Ax_dx, Ax_dy, Ax_dz = findAGrad(Ax, dx)
	Ay_dx, Ay_dy, Ay_dz = findAGrad(Ay, dx)
	Az_dx, Az_dy, Az_dz = findAGrad(Az, dx)

	Bx =  Az_dy - Ay_dz
	By = -Az_dx + Ax_dz
	Bz =  Ay_dx - Ax_dy

	return Bx, By, Bz

def findMagneticForce( Bx, By, Bz, vol, dx ):
	""" Calculate magnetic contribution to change in magnetic density
	Note: We set the magnetic permittivity (mu_0) to 1.
	Bx		 is matrix of x-direction of B field
	By 		 is matrix of y-direction of B field
	Bz		 is matrix of z-direction of B field
	dx		 is the cell size
	"""

	Bsq = Bx**2 + By**2 + Bz**2
	Bsq_dx, Bsq_dy, Bsq_dz = findAGrad(Bsq, dx)

	Bx_dx, Bx_dy, Bx_dz = findAGrad(Bx, dx)
	By_dx, By_dy, By_dz = findAGrad(By, dx)
	Bz_dx, Bz_dy, Bz_dz = findAGrad(Bz, dx)

	# Calculate magnetic pressure
	Bp_x = Bsq_dx / 2
	Bp_y = Bsq_dy / 2
	Bp_z = Bsq_dz / 2

	# Calculate magnetic tension
	Bt_x = Bx_dx * Bx + Bx_dy * By + Bx_dz * Bz
	Bt_y = By_dx * Bx + By_dy * By + By_dz * Bz
	Bt_z = Bz_dx * Bx + Bz_dy * By + Bz_dz * Bz

	# Calculate total magnetic force
	Bf_x = (Bt_x - Bp_x) * vol
	Bf_y = (Bt_y - Bp_y) * vol
	Bf_z = (Bt_z - Bp_z) * vol

	return Bf_x, Bf_y, Bf_z
	
def main():
	""" Finite Volume simulation """
	
    # USER CONTROL PARAMETERS #
	B0					   = 5     # MAGNETIC FIELD STRENGTH           
	sigma				   = 500   # ELECTRIC RESISTIVITY
	g 					   = -1    # GRAVITY	
	tEnd                   = 3.9   # SIMULATION TIME

	# Simulation parameters
	N                      = 64 # resolution N x 3N
	boxsizeX               = 0.5
	boxsizeY               = 1.5
	boxsizeZ               = boxsizeX / N
	gamma                  = 1.4 # ideal gas gamma                        
	courant_fac            = 0.4
	t                      = 0
	tOut                   = 0.01 # draw frequency
	useSlopeLimiting       = False
	plotRealTime = True # switch on for plotting as the simulation goes along
	
	# Mesh
	dx = boxsizeX / N
	vol = dx**2
	xlin = np.linspace(0.5*dx, boxsizeX-0.5*dx, N)
	ylin = np.linspace(0.5*dx, boxsizeY-0.5*dx, 3*N)
	zlin = np.linspace(0.5*dx, boxsizeZ-0.5*dx, 1)
	Y, X, Z = np.meshgrid( ylin, xlin, zlin )
	
	# Generate Initial Conditions - heavy fluid on top of light, with perturbation
	w0 = 0.0025
	P0 = 2.5
	rho = 1. + (Y > 0.75)
	vx = np.zeros(X.shape)
	vy = w0 * (1-np.cos(4*np.pi*X)) * (1-np.cos(4*np.pi*Y/3)) 
	vz = np.zeros(Z.shape)

	ylin = np.linspace(-0.5*dx, boxsizeY+0.5*dx, 3*N+2)
	YA, XA, ZA = np.meshgrid( ylin, xlin, zlin )
	Ax = np.zeros(XA.shape)                                                  
	Ay = np.zeros(YA.shape)                
	Az = B0 * (YA - boxsizeY/2)**2 * (XA - boxsizeX/2)**2
	P = P0 + g * (Y-0.75) * rho
	
	rho, vx, vy, vz, Ax, Ay, Az, P = addGhostCells(rho, vx, vy, vz, Ax, Ay, Az, P)

	# Get conserved variables
	Mass, Momx, Momy, Momz, Energy = getConserved( rho, vx, vy, vz, P, gamma, vol )

	# Get magnetic field
	Bx, By, Bz = findBField( Ax, Ay, Az, dx )
	
	# prep figure
	fig = plt.figure(figsize=(4,4), dpi=80)
	outputCount = 1
	
	# Simulation Main Loop
	while t < tEnd:

		# get Primitive variables
		rho, vx, vy, vz, P = getPrimitive( Mass, Momx, Momy, Momz, Energy, gamma, vol )
		
		# get time step (CFL) = dx / max signal speed
		dt = courant_fac * np.min( dx / (np.sqrt( gamma*P/rho ) + np.sqrt(vx**2+vy**2)) )
		plotThisTurn = False
		if t + dt > outputCount*tOut:
			dt = outputCount*tOut - t
			plotThisTurn = True
			#print([np.max(Bx), np.max(By), np.max(Bz)])
			#print([np.max(Ax_dd), np.max(Ay_dd), np.max(Az_dd)])
			#print([np.max(Ax), np.max(Ay), np.max(Az)])
			#print([np.max(vx), np.max(vy), np.max(vz)])
			#print([np.max(P_dx), np.max(P_dy), np.max(P_dz)])
		
		# Add Source (half-step)
		Mass, Momx, Momy, Momz, Energy = addSourceTerm( Mass, Momx, Momy, Momz, Bx, By, Bz, Energy, g, vol, dx, dt/2 )

		# get Primitive variables
		rho, vx, vy, vz, P = getPrimitive( Mass, Momx, Momy, Momz, Energy, gamma, vol )	
		
		# calculate gradients
		rho_dx, rho_dy, rho_dz = getGradient(rho, dx)
		vx_dx,  vx_dy,  vx_dz  = getGradient(vx,  dx)
		vy_dx,  vy_dy,  vy_dz  = getGradient(vy,  dx)
		vz_dx,  vz_dy,  vz_dz  = getGradient(vz,  dx)
		P_dx,   P_dy,   P_dz   = getGradient(P,   dx)
		Ax_dx,  Ax_dy,  Ax_dz  = findAGrad(Ax,  dx)
		Ay_dx,  Ay_dy,  Ay_dz  = findAGrad(Ay,  dx)
		Az_dx,  Az_dy,  Az_dz  = findAGrad(Az,  dx)

		Ax_dd = getALaplacian(Ax, dx)
		Ay_dd = getALaplacian(Ay, dx)
		Az_dd = getALaplacian(Az, dx)
		#print('laplace: ', np.max(abs(Ax_dd)), np.max(abs(Ay_dd)), np.max(abs(Az_dd)))
		
		# slope limit gradients
		if useSlopeLimiting:
			rho_dx, rho_dy, rho_dz = slopeLimit(rho, dx, rho_dx, rho_dy, rho_dz)
			vx_dx,  vx_dy,  vx_dz  = slopeLimit(vx , dx, vx_dx,  vx_dy,  vx_dz )
			vy_dx,  vy_dy,  vy_dz  = slopeLimit(vy , dx, vy_dx,  vy_dy,  vy_dz )
			vz_dx,  vz_dy,  vz_dz  = slopeLimit(vz , dx, vz_dx,  vz_dy,  vz_dz )
			P_dx,   P_dy,   P_dz   = slopeLimit(P  , dx, P_dx,   P_dy,   P_dz  )
			Ax_dx,  Ax_dy,  Ax_dz  = slopeLimit(Ax , dx, Ax_dx,  Ax_dy,  Ax_dz )
			Ay_dx,  Ay_dy,  Ay_dz  = slopeLimit(Ay , dx, Ay_dx,  Ay_dy,  Ay_dz )
			Az_dx,  Az_dy,  Az_dz  = slopeLimit(Az , dx, Az_dx,  Az_dy,  Az_dz )
			# need to add limiter for laplacian?
		
		# extrapolate half-step in time
		rho_prime = rho - 0.5*dt * ( vx * rho_dx + rho * vx_dx + vy * rho_dy + rho * vy_dy + vz * rho_dz + rho * vz_dz)
		vx_prime  = vx  - 0.5*dt * ( vx * vx_dx + vy * vx_dy + (1/rho) * P_dx + vz * vx_dz )
		vy_prime  = vy  - 0.5*dt * ( vx * vy_dx + vy * vy_dy + (1/rho) * P_dy + vz * vy_dz )
		vz_prime  = vz  - 0.5*dt * ( vx * vz_dx + vy * vz_dy + (1/rho) * P_dz + vz * vz_dz )
		P_prime   = P   - 0.5*dt * ( gamma*P * (vx_dx + vy_dy + vz_dz)  + vx * P_dx + vy * P_dy + vz * P_dz)

		Ax  = Ax - dt * ( vx * Ax_dx + vy * Ax_dy + vz * Ax_dz - Ax_dd / sigma )
		Ay  = Ay - dt * ( vx * Ay_dx + vy * Ay_dy + vz * Ay_dz - Ay_dd / sigma )
		Az  = Az - dt * ( vx * Az_dx + vy * Az_dy + vz * Az_dz - Az_dd / sigma )
		#print('b: ', np.max(Bx), np.max(By), np.max(Bz))
		#print('A: ', np.max(abs(Ax)), np.max(abs(Ay)), np.max(abs(Az)))

		# extrapolate in space to face centers
		rho_XL, rho_XR, rho_YL, rho_YR, rho_ZL, rho_ZR = extrapolateInSpaceToFace(rho_prime, rho_dx, rho_dy, rho_dz, dx)
		vx_XL,  vx_XR,  vx_YL,  vx_YR,  vx_ZL,  vx_ZR  = extrapolateInSpaceToFace(vx_prime,  vx_dx,  vx_dy,  vx_dz,  dx)
		vy_XL,  vy_XR,  vy_YL,  vy_YR,  vy_ZL,  vy_ZR  = extrapolateInSpaceToFace(vy_prime,  vy_dx,  vy_dy,  vy_dz,  dx)
		vz_XL,  vz_XR,  vz_YL,  vz_YR,  vz_ZL,  vz_ZR  = extrapolateInSpaceToFace(vz_prime,  vz_dx,  vz_dy,  vz_dz,  dx)
		P_XL,   P_XR,   P_YL,   P_YR,   P_ZL,   P_ZR   = extrapolateInSpaceToFace(P_prime,   P_dx,   P_dy,   P_dz,   dx)
		
		# compute fluxes (local Lax-Friedrichs/Rusanov)
		flux_Mass_X, flux_Momx_X, flux_Momy_X, flux_Momz_X, flux_Energy_X = getFlux(rho_XL, rho_XR, vx_XL, vx_XR, vy_XL, vy_XR, vz_XL, vz_XR, P_XL, P_XR, gamma)
		flux_Mass_Y, flux_Momy_Y, flux_Momz_Y, flux_Momx_Y, flux_Energy_Y = getFlux(rho_YL, rho_YR, vy_YL, vy_YR, vz_YL, vz_YR, vx_YL, vx_YR, P_YL, P_YR, gamma)
		flux_Mass_Z, flux_Momz_Z, flux_Momx_Z, flux_Momy_Z, flux_Energy_Z = getFlux(rho_YL, rho_YR, vz_ZL, vz_ZR, vx_ZL, vx_ZR, vy_ZL, vy_ZR, P_ZL, P_ZR, gamma)
				
		# update solution
		Mass   = applyFluxes(Mass, flux_Mass_X, flux_Mass_Y, flux_Mass_Z, dx, dt)
		Momx   = applyFluxes(Momx, flux_Momx_X, flux_Momx_Y, flux_Momx_Z, dx, dt)
		Momy   = applyFluxes(Momy, flux_Momy_X, flux_Momy_Y, flux_Momy_Z, dx, dt)
		Momz   = applyFluxes(Momz, flux_Momz_X, flux_Momz_Y, flux_Momz_Z, dx, dt)
		Energy = applyFluxes(Energy, flux_Energy_X, flux_Energy_Y, flux_Energy_Z, dx, dt)

		Bx, By, Bz = findBField( Ax, Ay, Az, dx )
		
		# Add Source (half-step)
		Mass, Momx, Momy, Momz, Energy = addSourceTerm( Mass, Momx, Momy, Momz, Bx, By, Bz, Energy, g, vol, dx, dt/2 )
		
		# update time
		t += dt
		
		# plot in real time - color 1/2 particles blue, other half red
		if (plotRealTime and plotThisTurn) or (t >= tEnd):
			plt.cla()
			plt.imshow(rho[:,:,0].T) # only display z=0
			plt.clim(0.8, 2.2)
			ax = plt.gca()
			ax.invert_yaxis()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)	
			ax.set_aspect('equal')	
			plt.pause(0.001)
			outputCount += 1
			print(t)
			
	
	# Save figure
	plt.savefig('finitevolume2_mod.png',dpi=240)
	plt.show()
	    
	return 0



if __name__== "__main__":
  main()

