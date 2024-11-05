# -*- coding: utf-8 -*-
"""
pendula_torus.py
Created on Wed Apr  3 21:34:03 2024

@author: hugo


Solução numérica das equações de movimento de pêndulos acoplados por plataforma
móvel. Desenhando toro a partir da fase dos pêndulos.

Numerical solutions for the equations of motion of two pendula coupled by a 
sliding cart.

After the work of Gustavo G. C. de A. Dias, J. R. Rios Leite, Josué S. da Fonseca, 
(DF-UFPE). 

"""

#%% Imports and parameters
from numpy import *
import numpy as np
from matplotlib.pyplot import *
from scipy.integrate import odeint, Radau, solve_ivp
from integrators import Ruth4_symplectic

### number of points in time series
#N = 10000

#Method = "my_odeint"
Method = "DOP853" # other choices: "RK45" (default), "RK23", "DOP853", "Radau", "BDF", "LSODA"
#Method = "Ruth4" # symplectic 4th order integrator
###time step
dt = 0.01
#tf = (N-1)*dt
#tf = 100
#tf = 800
#tf = 2621.430000001 
#tf = 10485.760000001 
tf = 200.0
N = int(tf/dt)+1

### Physical parameters
mc = 15.0
m1 = 1.0
m2 = 1.0
l1 = 1.0
l2 = 1.0
g = 1.0 #9.8 
M = m1+m2+mc
# gamma1 = 1e-6 #0.0 #5e-5 #0.001 #5e-5 #0.001
# gamma2 = 1e-6 #0.0 #5e-5 #0.001 #5e-5 #0.001
# gammac = 4e-6 #1e-3 #0.0 #0.005 #5e-3 #0.01
gamma1 = 0.0 #4e-6 #0.0 #5e-5 #0.001 #5e-5 #0.001
gamma2 = 0.0 #4e-6 #0.0 #5e-5 #0.001 #5e-5 #0.001
gammac = 0.0 #12*gamma1 #1e-3 #0.0 #0.005 #5e-3 #0.01


### parameters for the plot
### time step (in units of dt) in the plot (graph)
jump = 1 #10
### start time (in units of dt) in the plot (graph)
start_i = 0 #1900000


# initial conditions (q1, q2, xc, q1p, q2p, vc)[0]
#X0 = array([1.5, 1.48801, 2.0, -2, 0, 0])
#X0 = array([0.0, 0.0, 0.0, 20.0, 0.5, 0])
#X0 = array([0.0, 0.0, 0.0, 2.0, 0.5, 0])
#X0 = array([0.0, 0.0, 0.0, 20.0, 0.5, 0])
#X0 = array([pi/2, -pi/2, 0.0, 1.3, -1.30001, 0.0])
#X0 = array([0.1, -0.15, 0.0, 0.0, 0.0, 0.0])
#X0 = array([0.0, 0.0, 0.0, 3.8, -3.5, 0.0])
#X0 = array([1.5, -1.55, 0.0, 0.0, 0.0, 0.0])
#X0 = array([1.9, -1.95, 0.0, 0.0, 0.0, 0.0])
#X0 = array([0.7, -0.7, 0.0, -2.3*1.00001, 2.3, 0.0])

#X0 = array([0.7, -0.7*1.00001, 0.0, -0.8, 0.8, 0.0])
#X0 = array([0.7*1.00001, -0.7, 0.0, -0.8, 0.8, 0.0])
# X0 = array([0.7, -0.7*1.00001, 0.0, -0.7, 0.7, 0.0])
# X0 = array([0.7, -0.7*1.00001, 0.0, -0.9, 0.9, 0.0])
# X0 = array([0.7, -0.7*1.00001, 0.0, -1.0, 1.0, 0.0])
# X0 = array([0.7, -0.7*1.00001, 0.0, -1.1, 1.1, 0.0])
# X0 = array([0.7, -0.7*1.00001, 0.0, -1.2, 1.2, 0.0])
# X0 = array([0.7, -0.7*1.00001, 0.0, -1.3, 1.3, 0.0])
# X0 = array([0.7, -0.7*1.00001, 0.0, -1.4, 1.4, 0.0])
# X0[3:4] *= (1-1.0e-9)
# X0 = array([pi, 5*pi/6, 0.0, 0, 0, 0.0])
# X0 = array([pi/2, 3*pi/5, 0.0, 0, 0, 0.0])
# X0 = array([pi/2, 4*pi/5, 0.0, 0, 0, 0.0])
# X0 = array([pi, pi/10, 0.0, 0, 0, 0.0])
# X0 = array([pi, pi/2, 0.0, 0, 0, 0.0])
# X0 = array([pi, pi, 0.0, 0.1, -0.1, 0.0])
# X0 = array([pi, pi, 0.0, 1.0, -1.0, 0.0])
# X0 = array([pi, pi, 0.0, 3.0, -3.0, 0.0])
# X0 = array([pi, pi, 0.0, 2.5, -2.5, 0.0])
# X0 = array([0, 0, 0.0, 2.5, -2.5, 0.0])
# X0 = array([1e-9, 1e-9, 0.0, 2.9, -2.9, 0.0])
# X0 = array([1e-9, 1e-9, 0.0, 3.0, -3.0, 0.0])
# X0 = array([1e-9, 1e-9, 0.0, 2**0.5, -2**0.5, 0.0])
# X0 = array([1e-9, 1e-9, 0.0, 2.0, -2.0, 0.0])
#X0 = array([pi, pi, 0.0, 4.0, -4.0, 0.0])
#X0 = array([pi/3, -pi/3+0.01, 0.0, 0.0, 0.0, 0.0])
X0 = array([pi/2, -pi/2, 0.0, -0.5, 0.1, 0.0])
#X0 = array([pi/2, -pi/2, 0.0, -2.5, 2.1, 0.0])
#X0 = array([pi/2, -pi/2, 0.0, -3.5, 3.4, 0.0])
#X0 = array([pi/2, -pi/2, 0.0, -3.5, 2.4, 0.0])


#%% solving the ODE
print("Start")


def phase_wrap(theta):
    return ((theta+pi)%(2*pi))-pi

def pos_wrap(x):
    return ((x+W_stage/2) % W_stage) -W_stage/2

def angle_wrap(theta):
    return ((theta+pi/2)%(pi))-pi/2
    #return theta % pi


#def f(t, X):
def f_x_first(X, t):
    q1, q2, xc, qp1, qp2, vc = X[0], X[1], X[2], X[3], X[4], X[5]
    vcp = (m1*sin(q1)*(l1*qp1**2+g*cos(q1)) +m2*sin(q2)*(l2*qp2**2+g*cos(q2))+(gamma1/l1)*qp1*cos(q1)+(gamma2/l2)*qp2*cos(q2) -gammac*vc)/(M-m1*cos(q1)**2 -m2*cos(q2)**2)
    ### movement equations for absolute coordinates
    return array([
        qp1, 
        qp2, 
        vc,
        -(g/l1)*sin(q1) -vcp*cos(q1)/l1 - gamma1*qp1/(m1*l1**2),
        -(g/l2)*sin(q2) -vcp*cos(q2)/l2 - gamma2*qp2/(m2*l2**2),
        vcp
        ])
    
### odeint and solve_ivp have inverted the position of t and X in their function calls,
### the trick bellow fixes for compatibility with old code (odeint)
if Method == "my_odeint":
    def f(X,t):
        return f_x_first(X,t)
else:
    def f(t,X):
        return f_x_first(X,t)

def my_odeint(f, y0, t, *args):
    """Integrator of ordinary differential equations - Replacement for 
    scipy.integrate.odeint using 4th-order Runge-Kutta algorithm. Does not
    check accuracy or refines time step.
    
    Input arguments:
        f: callable(y,t0,...)
            Right-hand side the differential equation.
        y0: scalar of vector
            Initial condition.
        t: array 
            Times at which the solution is desired. Must be finely spaced.
        *args: additional (optional) arguments that will be passed to f.
    
    @author: Hugo L. D. de S. Cavalcante
    """
    #### n: number of points desired in the solution    
    n = len(t)
    #### m: number of dimensions in the phase space (solution). Equals the size 
    #### of the initial condition
    if () == np.shape(y0):
        m = len(np.array([y0]))
    else:
        m = len(y0)
    #### initialize the solution array with the initial condition y0
    y = np.zeros((n,m))
    y[0,:] = y0
    h = t[1:]-t[0:-1]
    for i in range(0,n-1):
        k1 = f(y[i,:], t[i], *args)
        k2 = f(y[i,:]+h[i]*k1/2, t[i]+h[i]/2, *args)
        k3 = f(y[i,:]+h[i]*k2/2, t[i]+h[i]/2, *args)
        k4 = f(y[i,:]+h[i]*k3, t[i]+h[i], *args)
        y[i+1,:] = y[i,:] +h[i]*(k1+2*(k2+k3)+k4)/6
    return y

### for the symplectic method, the differential equation needs to be separated
### in the coordinates and velocities f -> (F, G) 
# def F(X,V, t):
#     return f_x_first(concatenate((X,V)), t)[0:3]
# def G(X,V, t):
#     return f_x_first(concatenate((X,V)), t)[3:]
def F(X, V, t):
    q1, q2, xc, qp1, qp2, vc = X[0], X[1], X[2], V[0], V[1], V[2]
    #vcp = (m1*sin(q1)*(l1*qp1**2+g*cos(q1)) +m2*sin(q2)*(l2*qp2**2+g*cos(q2))+(gamma1/l1)*qp1*cos(q1)+(gamma2/l2)*qp2*cos(q2) -gammac*vc)/(M-m1*cos(q1)**2 -m2*cos(q2)**2)
    ### movement equations for absolute coordinates
    return array([
        qp1, 
        qp2, 
        vc
        ])
    return

def G(X, V, t):
    q1, q2, xc, qp1, qp2, vc = X[0], X[1], X[2], V[0], V[1], V[2]
    vcp = (m1*sin(q1)*(l1*qp1**2+g*cos(q1)) +m2*sin(q2)*(l2*qp2**2+g*cos(q2))+(gamma1/l1)*qp1*cos(q1)+(gamma2/l2)*qp2*cos(q2) -gammac*vc)/(M-m1*cos(q1)**2 -m2*cos(q2)**2)
    ### movement equations for absolute coordinates
    return array([
        -(g/l1)*sin(q1) -vcp*cos(q1)/l1 - gamma1*qp1/(m1*l1**2),
        -(g/l2)*sin(q2) -vcp*cos(q2)/l2 - gamma2*qp2/(m2*l2**2),
        vcp
        ])
    return


### eliminate transient
#t = linspace(0, 200.0, 20000)
#X = odeint(f, X0, t, atol = 1e-12)
#X0 = X[-1,:]

t = linspace(0.0, tf, N)
#X = odeint(f, X0, t, rtol = 1e-12, atol = 1e-12)
#X = odeint(f, X0, t, tfirst = True, rtol = 1e-12, atol = 1e-12)
#X = Radau(f, 0.0, X0, tf, rtol = 1e-12, atol = 1e-12)
#Solution = solve_ivp(f, (0.0, tf), X0, method = 'Radau', t_eval = t, rtol = 1e-16, atol = 1e-16)


### for use with the solve_ivp methods
#X = Solution["y"]


if Method == "Ruth4":
    X = Ruth4_symplectic(F, G, X0[0:3], X0[3:], t)
    print("Finished main loop.")
    q1, q2, xc = X[:,0], X[:,1], X[:,2]
    qp1, qp2, vc = X[:,3], X[:,4], X[:,5]
    
elif Method == "my_odeint":
    X = my_odeint(f, X0, t)
    print("Finished main loop.")
    ### When t is in the lines
    q1, q2, xc = X[:,0], X[:,1], X[:,2]
    qp1, qp2, vc = X[:,3], X[:,4], X[:,5]


else:
    Solution = solve_ivp(f, (0.0, tf), X0, method = Method, t_eval = t, rtol = 5e-14, atol = 1e-16)
    #Solution = solve_ivp(f, (0.0, tf), X0, method = Method, t_eval = t)
    message = Solution["message"]
    print(message)
    print("Finished main loop.")
    X = Solution["y"]
    ## When t is in the columns
    q1, q2, xc = X[0,:], X[1,:], X[2,:]
    qp1, qp2, vc = X[3,:], X[4,:], X[5,:]


#%% Post solution processing
# When t is in the lines
#q1, q2, xc = X[:,0], X[:,1], X[:,2]
#qp1, qp2, vc = X[:,3], X[:,4], X[:,5]
## When t is in the columns
#q1, q2, xc = X[0,:], X[1,:], X[2,:]
#qp1, qp2, vc = X[3,:], X[4,:], X[5,:]


### Energia total E = T+V
E = 0.5*((m1*(l1*qp1)**2)+(m2*(l2*qp2)**2)+M*vc**2)+vc*(m1*l1*qp1*cos(q1)+m2*l2*qp2*cos(q2))-(m1*l1*g*cos(q1)+m2*l2*g*cos(q2)) +m1*l1*g+m2*l2*g
qa = (q1-q2)/2
qs = (q1+q2)/2
#qs = (q1+q2)/2 -(m1+m2)/(m1+m2+mc)*xc
##plot(t, qa, label = '$q_a$')
##plot(t, qs, label = '$q_s$')
#plot(t, q1, label = '$q_1$')
#plot(t, q2, label = '$q_2$')
#plot(t, qc, label = '$q_c$')

### Calculando fases dos vetores nos subespaços de fase : phi = arctan2(qp1,q1) , theta = arctan2(qp2,q2)
## Equation for the single pendulum separatrix: theta_dot = l*sqrt(2*(1-cos(theta)))
q1_wrap, q2_wrap = phase_wrap(q1), phase_wrap(q2)
#q1_wrap, q2_wrap = angle_wrap(q1), angle_wrap(q2)
phi = arctan2(qp1, q1_wrap)
theta = arctan2(qp2, q2_wrap)
#phi = q1_wrap
#theta = q2_wrap

# phi = arctan2(qp1, q1)
# theta = arctan2(qp2, q2)
# phi[qp1 > l1*sqrt(2*(1-cos(q1)))] = qp1[qp1 > l1*sqrt(2*(1-cos(q1)))]*t[qp1 > l1*sqrt(2*(1-cos(q1)))]
# phi[qp1 < -l1*sqrt(2*(1-cos(q1)))] = -qp1[qp1 < -l1*sqrt(2*(1-cos(q1)))]*t[qp1 < -l1*sqrt(2*(1-cos(q1)))]
# theta[qp2 > l2*sqrt(2*(1-cos(q2)))] = qp2[qp2 > l2*sqrt(2*(1-cos(q2)))]*t[qp2 > l2*sqrt(2*(1-cos(q2)))]
# theta[qp2 < -l2*sqrt(2*(1-cos(q2)))] = -qp2[qp2 < -l2*sqrt(2*(1-cos(q2)))]*t[qp2 < -l2*sqrt(2*(1-cos(q2)))]
phi[qp1 > l1*sqrt(2*(1-cos(q1)))] = q1_wrap[qp1 > l1*sqrt(2*(1-cos(q1)))]
phi[qp1 < -l1*sqrt(2*(1-cos(q1)))] = q1_wrap[qp1 < -l1*sqrt(2*(1-cos(q1)))]
theta[qp2 > l2*sqrt(2*(1-cos(q2)))] = q2_wrap[qp2 > l2*sqrt(2*(1-cos(q2)))]
theta[qp2 < -l2*sqrt(2*(1-cos(q2)))] = q2_wrap[qp2 < -l2*sqrt(2*(1-cos(q2)))]

###### Esta conf. funciona para rotações, não librações
# q1_wrap, q2_wrap = phase_wrap(q1), phase_wrap(q2)
# phi = arctan2(qp1, q1_wrap)
# theta = arctan2(qp2, q2_wrap)
# phi[qp1 > l1*sqrt(2*(1-cos(q1)))] = q1_wrap[qp1 > l1*sqrt(2*(1-cos(q1)))]
# phi[qp1 < -l1*sqrt(2*(1-cos(q1)))] = q1_wrap[qp1 < -l1*sqrt(2*(1-cos(q1)))]
# theta[qp2 > l2*sqrt(2*(1-cos(q2)))] = q2_wrap[qp2 > l2*sqrt(2*(1-cos(q2)))]
# theta[qp2 < -l2*sqrt(2*(1-cos(q2)))] = q2_wrap[qp2 < -l2*sqrt(2*(1-cos(q2)))]
######

#%% plotting results (time series and Energy)
print("Plotting results.")
fig1, (ax1, ax2) = subplots(nrows = 2, ncols = 1, sharex = True, figsize=(10.67,6))
ax1.plot(t[start_i::jump], phase_wrap(q1[start_i::jump]), label = '$q_1$')
ax1.plot(t[start_i::jump], phase_wrap(q2[start_i::jump]), label = '$q_2$')
ax1.plot(t[start_i::jump], qp1[start_i::jump], label = '$\\dot{q}_1$')
ax1.plot(t[start_i::jump], qp2[start_i::jump], label = '$\\dot{q}_2$')
#ax1.plot(t[start_i::jump], phase_wrap(qs[start_i::jump]), label = '$q_s$')
#ax1.plot(t[start_i::jump], phase_wrap(qa[start_i::jump]), label = '$q_a$')
ax1.plot(t[start_i::jump], vc[start_i::jump], label = '$v_c$')
ax1.set_ylabel('$q_i$, $\\dot{q}_i$')
ax1.legend(loc=1)

ax2.plot(t[start_i::jump], E[start_i::jump], label = '$E$')
#plot(t[::10], E1[::10], label = '$E_A$')
#plot(t[::10], E2[::10], label = '$E_B$')
ax2.set_xlabel('time')
ax2.set_ylabel('$E$')
ax2.legend(loc=1)

### Finished solution and continuous-time plot

### Plotting a 3D torus
R = 5
r = 1
fig2 = figure("Torus", figsize=(6,6))
ax3 = fig2.add_subplot(projection='3d')

x = (R+r*cos(theta))*cos(phi)
y = (R+r*cos(theta))*sin(phi)
z = r*sin(theta)

ax3.plot(x, y, z, 'o', ms = 1.0, label = "torus")
xlabel("x")
ylabel("y")
ax3.set_zlabel("z")
xlim(-1.1*(R+r), 1.1*(R+r))
ylim(-1.1*(R+r), 1.1*(R+r))
ax3.set_zlim(-1.1*(R+r), 1.1*(R+r))

full_angle = linspace(0, 2*pi, 32)
theta_surf, phi_surf = meshgrid(full_angle, full_angle)
x_surf = (R+0.98*r*cos(theta_surf))*cos(phi_surf)
y_surf = (R+0.98*r*cos(theta_surf))*sin(phi_surf)
z_surf = r*0.98*sin(theta_surf)
ax3.plot_surface(x_surf, y_surf, z_surf, rstride = 1, cstride = 1, alpha = 0.1)


show()

#%% Discretization and other analysis

### Discretization
MAX_DISCRETE = 10000
Q = zeros((2, MAX_DISCRETE))
N_discrete = 0
#q1_wrap, q2_wrap = phase_wrap(q1), phase_wrap(q2)
for i in range(N-1):
    #if q1_wrap[i]*q1_wrap[i+1] < 0 and q1_wrap[i]>q1_wrap[i+1]:
    #    Q[:, N_discrete] = q2[i], xc[i]
    #    N_discrete += 1
    # if vc[i]*vc[i+1] < 0 and vc[i]>vc[i+1]:
    #     Q[:, N_discrete] = q1_wrap[i], q2_wrap[i]
    #     N_discrete += 1
    if (phi[i+1]>0) and (phi[i]<=0):
        Q[:, N_discrete] = x[i], z[i]
        N_discrete += 1

print(f"{N_discrete} eventos discretos encontrados.")
Q = Q[:,:N_discrete] 

fig3 = figure("Discrete time", figsize=(6,6))
plot(Q[0,:], Q[1,:], 'o', markersize = 0.8)
#xlabel("$Q_1$")
#ylabel("$Q_2$")
xlabel("$x[n]$")
ylabel("$z[n]$")

       #plt.show()
show()



#%% Fourier transform
TF = fft.rfft(qp1)
freqs = fft.rfftfreq(N)
### Com rfft, o passo de frequência freqs[1]-freqs[0] é 1/N, deveria ser 1/(N*ts)
freqs = freqs/dt
omegas = 2*pi*freqs

TF_mag = abs(TF)*2/N
TF_fase = angle(TF)

fig4 = figure("Spectrum", figsize=(6,6))
plot(freqs, 20*log10(TF_mag))
xlabel('$f$')
xscale('log')
ylabel('$|TF|$ (dB)')
#ylim([-5,amax(log10(TF_mag))+1])
ylim(-60 )

print("Bye!")