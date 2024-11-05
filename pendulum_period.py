# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 17:16:37 2022

@author: Hugo Cavalcante
"""

from numpy import arange, array, arccos, sin, cos, Inf, pi, sqrt, vectorize
from scipy.special import ellipkinc, ellipk
from matplotlib.pyplot import *
from numpy import meshgrid, zeros_like
from mpl_toolkits import mplot3d

E0 = 0.00001
Emax = 8.0
Et = 2.0
dE = 0.001

### this parameter limits the growth of the period, as ellipkinc diverges for m = k² = 1
#ellip_m_max = 1.1 #0.9999

E = arange(E0, Emax, dE)

m, g, l = 1.0, 1.0, 1.0


def Period(E):
    if E==0.0:
        T = Inf
    elif E<=Et:
        theta_max = arccos(1.0-E)
        #T = (4*sqrt(2.0/E))*ellipkinc(theta_max/2, min(2.0/E,ellip_m_max))
        #T = (1*sqrt(2.0/E))*ellipkinc(theta_max/2, min(2.0/E,ellip_m_max))
        T = 4*sqrt(l/g)*ellipk(sin(theta_max/2))
    else:
        T = pi*l*sqrt(2*m/(E-Et))
    return T

vPeriod = vectorize(Period)

T = vPeriod(E)

# theta_max = arccos(1.0-E)
# figure()
# plot(E, theta_max)
# xlabel('$E$')
# ylabel('$\\theta_{max}$')

# EllipIInc = ellipkinc(theta_max/2, min(2.0/E,ellip_m_max))
# figure()
# plot(E, EllipIInc)
# xlabel('$E$')
# ylabel('$I$')


figure()
plot(E, T)
xlabel('$E$')
ylabel('$T$')
ylim(0.0, 40)


omega = 2*pi/T
figure()
plot(E, omega)
xlabel('$E$')
ylabel('$\\omega$')

#figure()
# E1 = E.copy()
# E2 = E.copy()
# E_T = arange(E0, Emax, 0.2)
# omega1 = 2*pi/vPeriod(E1)
# omega2 = 2*pi/vPeriod(E2)

# X, Y = meshgrid(E1[::20], E2[::20])
# Z = 2*pi/vPeriod(X) - 2*pi/vPeriod(Y)
# fig, ax = subplots(1,1)
# #CP = ax.contourf(X, Y, Z)
# #CP = ax.contour(X, Y, Z)
# #fig.colorbar(CP)
# #fig = figure()
# #ax = axes(projection = '3d')
# #SP = ax.plot_surface(X, Y, Z)
# #SP = ax.plot_surface(X, Y, Z, cmap = 'viridis', edgecolor = 'none')
# #SP2 = ax.plot_surface(X, Y, zeros_like(Z))
# CP = ax.contour(X, Y, Z)
# ax.set_title("frequencies difference")
# ax.set_xlabel('E1')
# ax.set_ylabel('E2')
# show()

figure()
plot([E0, Emax], [0, 0], '--k', )
xlabel('$E_1$')
ylabel('$\\omega_1 -\\omega_2$')
#ylabel('$|\\omega_1 -\\omega_2|$')
E_T = arange(E0, Emax, 0.6)
for i in range(1,len(E_T)):
    E1 = arange(E0, E_T[i], dE)
    E2 = E_T[i] - E1
    omega1 = 2*pi/vPeriod(E1)
    omega2 = 2*pi/vPeriod(E2)
    plot(E1, omega1-omega2, label="ET ={:.2g}".format(E_T[i]))
    #plot(E1, abs(omega1-omega2))

legend(loc=1)

