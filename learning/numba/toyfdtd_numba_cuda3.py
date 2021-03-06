# ToyFDTD1, version 1.03
#      The if-I-can-do-it-you-can-do-it FDTD!
# Copyright (C) 1998,1999 Laurie E. Miller, Paul Hayes, Matthew O'Keefe
#
# This program is free software; you can redistribute it and/or
#     modify it under the terms of the GNU General Public License
#     as published by the Free Software Foundation; either version 2
#     of the License, or any later version, with the following conditions
#     attached in addition to any and all conditions of the GNU
#     General Public License:
#     When reporting or displaying any results or animations created
#     using this code or modification of this code, make the appropriate
#     citation referencing ToyFDTD1 by name and including the version
#     number.
#
# This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty
#     of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#     See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
#     along with this program; if not, write to the Free Software
#     Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
#     02111-1307  USA
#
# Contacting the authors:
#
# Laurie E. Miller, Paul Hayes, Matthew O'Keefe
# Department of Electrical and Computer Engineering
#      200 Union Street S. E.
#      Minneapolis, MN 55455
#
# lemiller@borg.umn.edu
#
# http://www.borg.umn.edu/toyfdtd/
# http://www.borg.umn.edu/toyfdtd/ToyFDTD1.html
# http://www.toyfdtd.org/
#
# This code is here for everyone, but not everyone will need something
#      so simple, and not everyone will need to read all the comments.
# This file is over 700 lines long, but less than 400 of that is actually
#      code.
#
# This ToyFDTD1 is a stripped-down, minimalist 3D FDTD code.  It
#      illustrates the minimum factors that must be considered to
#      create a simple FDTD simulation.
#
# Python port by Ertugrul Karademir
#

import numpy as np
import matplotlib.pyplot as plt
from numba import jit, vectorize, cuda
from numba import *
from timeit import default_timer as timer

# Physicsl constants
# Speed of light
C = 299792458.0 #meters/second
C2 = 89875517873681764.0 #meters^2/second^2
# Vacuum Permeability
MU0 = 1.2566370614359172953850573533118011536788677597500423283899778369231265625144835994512139301368468271e-6
# Vacuum Permittivity
EP0 = 8.8541878176203898505365630317107502606083701665994498081024171524053950954599821142852891607182008932e-12


# Total number of timesteps
MAXIMUM_ITERATION = 100
# The program will output 3D data every PLOT_MODULUS timestep.
PLOT_MODULUS = 5
# Frequency of the stimulus
FREQUENCY = C/532e-9

GUIDE_WIDTH  = 2.0e-6 #meters
GUIDE_HEIGHT = 2.0e-6 #meters

# Length of the wg in wls of stimulus
LENGTH_IN_WAVELENGTHS = 5.0

# Min number of grid cells per wl in x,y,z
CELLS_PER_WAVELENGTH = 25.0

# @cuda.jit('float32(float32,float32,float32,float32,float32,float32,float32)', inline=True)


# @vectorize(["float32(float32,float32,float32,float32,float32,float32,float32)"], target='cpu')
@cuda.jit(argtypes=[f4, f4, f4, f4, f4, f4, f4], device=True, inline=True)
def calc_fields(Hi, c1, Ej1, Ej2, c2, Ek1,Ek2):
    return  Hi + c1*(Ej1-Ej2) - c2*(Ek1-Ek2)


# @cuda.jit(argtypes=[f4,f4,f4,f4,f4,f4,f4], device=True, inline=True)
# def difference_operation(Hi, c1, Ej1, Ej2, c2, Ek1,Ek2):
#     return calc_fields(Hi, Ej1, Ej2, c2, Ek1, Ek2)

@vectorize(["float32,float32,float32,float32,float32,float32(float32,float32,float32,float32,float32,float32,float32,"\
+"float32,float32,float32,float32,float32,float32,float32,"\
+"float32,float32,float32,float32,float32,float32,float32,"\
+"float32,float32,float32,float32,float32,float32,float32,"\
+"float32,float32,float32,float32,float32,float32,float32,"\
+"float32,float32,float32,float32,float32,float32,float32)"], target='cuda')
def update_fields(
        Hx, dtmudz1, Ey11, Ey12, dtmudy1, Ez11,Ez12, \
        Hy, dtmudx2, Ez21, Ez22, dtmudz2, Ex21,Ex22, \
        Hz, dtmudy3, Ex31, Ex32, dtmudx3, Ey31,Ey32, \
        Ex, dtepsdy1, Hz11, Hz12, dtepsdz1, Hy11, Hy12, \
        Ey, dtepsdz2, Hx21, Hx22, dtepsdx2, Hz21, Hz22, \
        Ez, dtepsdx3, Hy31, Hy32, dtepsdy3, Hx31, Hx32, \
    ):
    Hx = Hx + dtmudz1*(Ey11-Ey12) - dtmudy1*(Ez11-Ez12)
    Hy = Hy + dtmudx2*(Ez21-Ez22) - dtmudz2*(Ex21-Ex22)
    Hz = Hz + dtmudy3*(Ex31-Ex32) - dtmudx3*(Ey31-Ey32)
    Ex = Ex + dtepsdy1*(Hz11-Hz12) - dtepsdz1*(Hy11-Hy12)
    Ey = Ey + dtepsdz2*(Hx21-Hx22) - dtepsdx2*(Hz21-Hz22)
    Ez = Ez + dtepsdx3*(Hy31-Hy32) - dtepsdy3*(Hx31-Hx32)

    return Hx, Hy, Hz, Ex, Ey, Ez


def main():
    iteration = 0 # time step
    stimulus = 0.0 # value of the stimulus at the time step
    currentSimulatedTime = 0.0 # time in secs simulated
    totalSimulatedTime = 0.0 # time in secs to be simulated

    # wavelength in meters
    lmd = C / FREQUENCY
    # angular feq in rad/sec
    omega = 2.0*np.pi*FREQUENCY

    # set ny and dy
    # start with small ny
    ny = 3
    # calculate dy from the guide width and ny
    dy = GUIDE_WIDTH/ny
    # until dy is less than a twenty-fifth of a wavelength
    #   increment ny and recalculate dy
    while dy >= lmd/CELLS_PER_WAVELENGTH:
        ny = ny + 1
        dy = GUIDE_WIDTH/ny

    # set nz and dz:
    # start with a small nz:
    nz = 3
    # calculate dz from the guide height and nz:
    dz = GUIDE_HEIGHT/nz;
    # until dz is less than a twenty-fifth of a wavelength,
    #   increment nz and recalculate dz:
    while dz >= lmd/CELLS_PER_WAVELENGTH:
        nz = nz + 1
        dz = GUIDE_HEIGHT/nz

    # set dx, nx, and dt
    # set dx equal to dy or dz whichever is smaller
    dx = dy if dy < dz else dz
    nx = int(LENGTH_IN_WAVELENGTHS*lmd/dx)


    # +/- versions of ns
    nxpp, nxmm = nx+1, nx-1
    nypp, nymm = ny+1, ny-1
    nzpp, nzmm = nz+1, nz-1

    # chose dt for Courant stability:
    dt = 1.0/(C*np.sqrt(1.0/(dx*dx) + 1.0/(dy*dy) + 1.0/(dz*dz)))

    totalSimulatedTime = MAXIMUM_ITERATION*dt

    # constants used in the field update equations:
    dtmudx = dt/(MU0*dx);
    dtepsdx = dt/(EP0*dx);
    dtmudy = dt/(MU0*dy);
    dtepsdy = dt/(EP0*dy);
    dtmudz = dt/(MU0*dz);
    dtepsdz = dt/(EP0*dz);

    #initialize Fields
    Ex = np.zeros((nx,   nypp, nzpp), dtype='float32')
    # np.ascontiguousarray(Ex, dtype='float32')
    Ey = np.zeros((nxpp, ny,   nzpp), dtype='float32')
    # np.ascontiguousarray(Ey, dtype='float32')
    Ez = np.zeros((nxpp, nypp, nz), dtype='float32')
    # np.ascontiguousarray(Ez, dtype='float32')

    Hx = np.zeros((nxmm, ny,   nz), dtype='float32')
    # np.ascontiguousarray(Hx, dtype='float32')
    Hy = np.zeros((nx,   nymm, nz), dtype='float32')
    # np.ascontiguousarray(Hy, dtype='float32')
    Hz = np.zeros((nx,   ny,   nzmm), dtype='float32')
    # np.ascontiguousarray(Hz, dtype='float32')


    # MAIN LOOP
    for iteration in range(MAXIMUM_ITERATION):
        if iteration%PLOT_MODULUS == 0:
            ### Output part
            # pass
            print("Sim time (ns):",currentSimulatedTime/1e-9)
            print("Iteration:",iteration, "of", MAXIMUM_ITERATION)
            plt.imshow(Ez[:,:,15], interpolation='none')
            plt.savefig("./img/cuda3_Ez_{}.png".format(iteration))

        # Compute the stimulus: a plane wave emanates from the x=0 face:
        #     The length of the guide lies in the x-direction, the width of the
        #     guide lies in the y-direction, and the height of the guide lies
        #     in the z-direction.  So the guide is sourced by all the ez
        #     components on the stimulus face.

        stimulus = np.sin(omega*currentSimulatedTime)
        Ez[0,0:nypp,0:nz] = stimulus

        Hx, Hy, Hz, Ex, Ey, Ez = update_fields(
            Hx[0:nxmm,0:ny,0:nz],dtmudz, Ey[1:nx, 0:ny, 1:nzpp], Ey[1:nx, 0:ny, 0:nz],dtmudy, Ez[1:nx, 1:nypp, 0:nz],Ez[1:nx, 0:ny, 0:nz],
            Hy[0:nx,0:nymm,0:nz],dtmudx, Ez[1:nxpp, 1:ny, 0:nz], Ez[0:nx, 1:ny, 0:nz],dtmudz, Ex[0:nx, 1:ny, 1:nzpp],Ex[0:nx, 1:ny, 0:nz],
            Hz[0:nx,0:ny,0:nzmm],dtmudy, Ex[0:nx, 1:nypp, 1:nz], Ex[0:nx, 0:ny, 1:nz],dtmudx, Ey[1:nxpp, 0:ny, 1:nz],Ey[0:nx, 0:ny, 1:nz],
            Ex[0:nx,1:ny,1:nz], dtepsdy, Hz[0:nx, 1:ny, 0:nzmm], Hz[0:nx, 0:nymm, 0:nzmm], dtepsdz, Hy[0:nx, 0:nymm, 1:nz], Hy[0:nx, 0:nymm, 0:nzmm],
            Ey[1:nx,0:ny,1:nz], dtepsdz, Hx[0:nxmm, 0:ny, 1:nz], Hx[0:nxmm, 0:ny, 0:nz-1], dtepsdx, Hz[1:nx, 0:ny, 0:nzmm], Hz[0:nxmm, 0:ny, 0:nz-1],
            Ez[1:nx,1:ny,0:nz], dtepsdx, Hy[1:nx, 0:nymm, 0:nz], Hy[0:nxmm, 0:nymm, 0:nz], dtepsdy, Hx[0:nxmm, 1:ny, 0:nz], Hx[0:nxmm, 0:nymm, 0:nz]
        )

        # Compute the boundary conditions:

        # OK, so I'm yanking your chain on this one.  The PEC condition is
        # enforced by setting the tangential E field components on all the
        # faces of the mesh to zero every timestep (except the stimulus
        # face).  But the lazy/efficient way out is to initialize those
        # vectors to zero and never compute them again, which is exactly
        # what happens in this code.

        currentSimulatedTime = dt*float(iteration)
        #END OF MAIN LOOP


    # Output section

    # Some progress

if __name__ == '__main__':
    start = timer()
    main()
    print(timer()-start)
