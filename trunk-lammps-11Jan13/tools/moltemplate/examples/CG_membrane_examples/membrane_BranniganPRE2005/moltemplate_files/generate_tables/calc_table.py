#!/usr/bin/env python

# Calculate a table of pairwise energies and forces between "INT" atoms
# in the lipid membrane model described in 
#   Brannigan et al, Phys Rev E, 72, 011915 (2005)
# The energy of this interaction U(r) = eps*(0.4*(sigma/r)^12 - 3.0*(sigma/r)^2)
# However it is truncated at rc2 = 22.5 (shifted upwards to maintain continuity)

# The previous version included the repulsive core term
def U(r, eps, sigma):
    return eps*   (0.4*pow((sigma/r),12)  -  3.0*sigma*sigma/(r*r))
def F(r, eps, sigma):
    return eps*(12*0.4*pow((sigma/r),13)/sigma - 2*3.0*sigma*sigma/(r*r*r))
# We don't want to do that.  Instead compute the core repulsion using a
# different pair_style and add the attractive term on top of it using the table.
# This way it the core repulsion acts as a default interaction with other atom
# types (using the new repulsive mixing rules).

epsilon = 2.75/4.184 # kCal/mole
sigma   = 7.5
Rmin    = 2.6
Rmax    = 22.6
rcut    = 22.5
N       = 1001

for i in range(0,N):
    r = Rmin + i*(Rmax-Rmin)/(N-1)
    U_r = U(r, epsilon, sigma) - U(rcut, epsilon, sigma)
    F_r = F(r, epsilon, sigma)
    if r > rcut:
        U_r = 0.0
        F_r = 0.0
    print(str(i+1)+' '+str(r)+' '+str(U_r)+' '+str(F_r))

