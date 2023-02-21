import math as m
import numpy as np
#import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import os

# 1D Hybrid model of Ar plasma sheath: PIC model for electrons and kinetic approach for ions.

def getAcc(pos_e, Nx, boxsize, neff, Gmtx, Laptx, t, Vrf, w, Vdc):

    N = pos_e.shape[0]
    dx = boxsize / Nx
    j = np.floor(pos_e / dx).astype(int)
    jp1 = j + 1
    weight_j = (jp1 * dx - pos_e) / dx
    weight_jp1 = (pos_e - j * dx) / dx



    n = np.bincount(j[:, 0], weights=weight_j[:, 0], minlength=Nx + 1);
    n += np.bincount(jp1[:, 0], weights=weight_jp1[:, 0], minlength=Nx + 1);

    n = np.delete(n, Nx)
    # boundary conditions for Poisson equation
    n[0] = 0
    n[Nx - 1] = (Vdc - Vrf * np.sin(w * t)) / dx ** 2

    n *= neff * 0.018080 / dx  # [V / mkm^2] = [n counts]*[e C]/[eps0 F/mkm]/[dx mkm]/[1 mkm^2]

    # Solve Poisson's Equation: laplacian(phi) = -n
    phi_Pois_grid = spsolve(Laptx, n, permc_spec="MMD_AT_PLUS_A")


    # Apply Derivative to get the Electric field

    E_grid = - Gmtx @ phi_Pois_grid

    # Boundary electric field
    E_grid = np.hstack((E_grid, 0))

    # Interpolate grid value onto particle locations

    Ee = weight_j * E_grid[j] + weight_jp1 * E_grid[jp1]

    ae = -Ee * neff

    # Unit calibration [a mkm/ ns^2] = [a V/mkm/e.m.u.] * [e C] * [1E-12 mkm/ns^2] / [me kg/e.m.u.]

    ae = ae * 1.76E-1

    return ae

def main():

    N = 500000  # Number of particles. Need 500 000 real particles. Smodel = 10000 mkm2
    Nx = 10000  # Number of mesh cells Need dx <= 0.01 mkm
    t = 0  # current time of the simulation
    tEnd = 100  # time at which simulation ends [ns]
    dt = 0.01  # timestep [1ns]
    boxsize = 500  # periodic domain [0,boxsize] [mkm] 1000 mkm
    # neff = 100  # number of real particles corresponding to count particles
    neff = 1  # number of real particles corresponding to count particles
    vth = 1E-3  # m/s to mkm/ns
    Te = 2.3  # electron temperature
    Ti = 0.06  # ion temperature
    me = 1  # electron mass
    mi = 73000  # ion mass
    Energy_max = 5.0  # max electron energy
    deltaE = 100  # energy discretization
    w = 2 * np.pi * 0.01356  # frequency
    # C = 1.4E-20  # capacity C = C0[F]/(Selectr/Smodel) Smodel = 1 mkm^2, Selectr = 7.1e10 mkm^2, C0 = 1000 pF
    C = 1.4E-16  # capacity C = C0[F]/(Selectr/Smodel) Smodel = 10000 mkm^2, Selectr = 7.1e10 mkm^2, C0 = 1000 pF
    initials = True  # initial condition calculation or plasma parameters

    # Generate Initial Conditions
    np.random.seed(42)  # set the random number generator seed

    # Nh = int(N / 2)
    Te *= 1.7E12 / 9.1  # kT/me
    Ti *= 1.7E12 / 9.1 / mi  # kT/mi
    me *= neff
    mi *= neff

    # Construct matrix G to computer Gradient  (1st derivative) (BOUNDARY CONDITIONS)
    dx = boxsize / Nx
    e = np.ones(Nx)
    diags = np.array([-1, 1])
    vals = np.vstack((-e, e))
    Gmtx = sp.spdiags(vals, diags, Nx, Nx);
    Gmtx = sp.lil_matrix(Gmtx)
    Gmtx[0, 0] = -2
    Gmtx[0, 1] = 2
    Gmtx[Nx - 1, Nx - 1] = 2
    Gmtx[Nx - 1, Nx - 2] = -2
    Gmtx /= (2 * dx)
    Gmtx = sp.csr_matrix(Gmtx)

    # Construct matrix L to computer Laplacian (2nd derivative) for Poisson equation
    diags = np.array([-1, 0, 1])
    vals = np.vstack((e, -2 * e, e))
    Laptx = sp.spdiags(vals, diags, Nx, Nx);
    Laptx = sp.lil_matrix(Laptx)
    Laptx[0, 0] = -1
    Laptx[0, 1] = 1
    Laptx[Nx - 1, Nx - 2] = 0
    Laptx[Nx - 1, Nx - 1] = 1
    Laptx /= dx ** 2
    Laptx = sp.csr_matrix(Laptx)

    Vdc = 0
    Vrf = 15

    pos_e = np.random.rand(N, 1) * boxsize
    vel_e = vth * np.random.normal(0, m.sqrt(Te), size=(N, 1))
    acc_e = getAcc(pos_e, Nx, boxsize, neff, Gmtx, Laptx, t, Vrf, w, Vdc)
    vel_e += acc_e * dt / 2.0

    print(vel_e)

    return 0


if __name__ == "__main__":
    main()

