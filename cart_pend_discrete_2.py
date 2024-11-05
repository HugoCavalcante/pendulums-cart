# -*- coding: utf-8 -*-
"""
pendula_discrete_1.py

Solução numérica das equações de movimento de pêndulos acoplados por plataforma
móvel. Construindo Mapa para visualizar/classificar soluções.

Numerical solutions for the equations of motion of two pendula coupled by a 
sliding cart.

After the work of Gustavo G. C. de A. Dias, J. R. Rios Leite, Josué S. da Fonseca, 
(DF-UFPE). 

Created on Sun Mar 10 11:44:48 2024

@author: Hugo L. D. de S. Cavalcante
"""

#%% Imports and parameters
from numpy import *
import numpy as np
from matplotlib.pyplot import *
from scipy.integrate import odeint, Radau, solve_ivp
from integrators import Ruth4_symplectic


Plot_ts = False
Plot_discrete = True
Plot_spectrum = False

#Method = "my_odeint"
Method = "DOP853" # other choices: "RK45" (default), "RK23", "DOP853", "Radau", "BDF", "LSODA"
#Method = "Ruth4" # symplectic 4th order integrator
###time step
dt = 0.01
#tf = (N-1)*dt
tf = 2000.0
N = int(tf/dt)+1


### Physical parameters
mc = 5.0
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

def phase_wrap(theta):
    return ((theta+pi)%(2*pi))-pi

def pos_wrap(x):
    return ((x+W_stage/2) % W_stage) -W_stage/2

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


t = linspace(0.0, tf, N)

# initial conditions (q1, q2, xc, q1p, q2p, vc)[0]
# X0 = array([pi, 5*pi/6, 0.0, 0, 0, 0.0])
# X0 = array([pi/2, 3*pi/5, 0.0, 0, 0, 0.0])
# X0 = array([pi/2, 4*pi/5, 0.0, 0, 0, 0.0])
# X0 = array([1e-9, 1e-9, 0.0, 2.0, -2.0, 0.0])
#X0 = array([pi, pi, 0.0, 4.0, -4.0, 0.0])

N_series = 64
Velocity_radius = 1.8
V_angle = 2*pi/N_series
V_theta = arange(0.0, 2*pi, V_angle)
for theta_i in range(N_series):
    q10 = pi/2
    q20 = pi/2
    xc0 = -(m1*l1*sin(q10)+m2*l2*sin(q20))/M
    
    q1p0 = Velocity_radius*cos(V_theta[theta_i])
    q2p0 = Velocity_radius*sin(V_theta[theta_i])

    vc0 = -(m1*l1*cos(q10)*q1p0+m2*l2*cos(q20)*q2p0)/M
    E0 = 0.5*((m1*(l1*q1p0)**2)+(m2*(l2*q2p0)**2)+M*vc0**2)+vc0*(m1*l1*q1p0*cos(q10)+m2*l2*q2p0*cos(q20))-(m1*l1*g*cos(q10)+m2*l2*g*cos(q20)) +m1*l1*g+m2*l2*g
    X0 = array([q10, q20, xc0, q1p0, q2p0, vc0])
    print(f"E0 = {E0:.4g}")

    #%% solving the ODE
    #print("Start")
    print(f"Calculating solution {theta_i+1} of {N_series}")
    
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
        print("Finished solution.")
        X = Solution["y"]
        ## When t is in the columns
        q1, q2, xc = X[0,:], X[1,:], X[2,:]
        qp1, qp2, vc = X[3,:], X[4,:], X[5,:]
    

    # When t is in the lines
    #q1, q2, xc = X[:,0], X[:,1], X[:,2]
    #qp1, qp2, vc = X[:,3], X[:,4], X[:,5]
    ## When t is in the columns
    #q1, q2, xc = X[0,:], X[1,:], X[2,:]
    #qp1, qp2, vc = X[3,:], X[4,:], X[5,:]
    
    
    ### Energia total E = T+V
    #E = 0.5*((m1*(l1*qp1)**2)+(m2*(l2*qp2)**2)+M*vc**2)+vc*(m1*l1*qp1*cos(q1)+m2*l2*qp2*cos(q2))-(m1*l1*g*cos(q1)+m2*l2*g*cos(q2)) +m1*l1*g+m2*l2*g
    #qa = (q1-q2)/2
    #qs = (q1+q2)/2
    #qs = (q1+q2)/2 -(m1+m2)/(m1+m2+mc)*xc
    ##plot(t, qa, label = '$q_a$')
    ##plot(t, qs, label = '$q_s$')
    #plot(t, q1, label = '$q_1$')
    #plot(t, q2, label = '$q_2$')
    #plot(t, qc, label = '$q_c$')


    #%% plotting results (time series and Energy)
    if Plot_ts:
        print("Plotting time series (continuous time).")
        E = 0.5*((m1*(l1*qp1)**2)+(m2*(l2*qp2)**2)+M*vc**2)+vc*(m1*l1*qp1*cos(q1)+m2*l2*qp2*cos(q2))-(m1*l1*g*cos(q1)+m2*l2*g*cos(q2)) +m1*l1*g+m2*l2*g
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
        ax2.plot([t[start_i], t[-1]], [E0, E0], lw = 0.5, label = '$E_0$')
        #plot(t[::10], E1[::10], label = '$E_A$')
        #plot(t[::10], E2[::10], label = '$E_B$')
        ax2.set_xlabel('time')
        ax2.set_ylabel('$E$')
        ax2.legend(loc=1)
        show()
        ### Finished solution and continuous-time plot


    #%% Discretization and other analysis
    
    ### Discretization
    MAX_DISCRETE = 10000
    Q = zeros((2, MAX_DISCRETE))
    N_discrete = 0
    q1_wrap, q2_wrap = phase_wrap(q1), phase_wrap(q2)
    for i in range(N-1):
        # if q1_wrap[i]*q1_wrap[i+1] < 0 and q1_wrap[i]<q1_wrap[i+1]:
        #     Q[:, N_discrete] = q2[i], xc[i]
        #     N_discrete += 1
        # if vc[i]*vc[i+1] < 0 and vc[i]>vc[i+1]:
        #     Q[:, N_discrete] = q1_wrap[i], q2_wrap[i]
        #     N_discrete += 1
        if q1_wrap[i]*q1_wrap[i+1] < 0 and q1_wrap[i]<q1_wrap[i+1]:
            Q[:, N_discrete] = q2_wrap[i], qp2[i]
            N_discrete += 1


    print(f"{N_discrete} eventos discretos encontrados.")
    Q = Q[:,:N_discrete] 
    
    if Plot_discrete:
        fig2 = figure("Discrete time", figsize=(6,6))
        plot(Q[0,:], Q[1,:], 'o', markersize = 0.8)
        xlabel("$Q_2$")
        ylabel("$\\dot{Q}_2$")
               #plt.show()
        show()

    if Plot_spectrum:
        #%% Fourier transform
        TF = fft.rfft(qp1)
        freqs = fft.rfftfreq(N)
        ### Com rfft, o passo de frequência freqs[1]-freqs[0] é 1/N, deveria ser 1/(N*ts)
        freqs = freqs/dt
        omegas = 2*pi*freqs
        
        TF_mag = abs(TF)*2/N
        TF_fase = angle(TF)
        
        fig3 = figure("Spectrum", figsize=(6,6))
        plot(freqs, 20*log10(TF_mag))
        xlabel('$f$')
        xscale('log')
        ylabel('$|TF|$ (dB)')
        #ylim([-5,amax(log10(TF_mag))+1])
        ylim(-60 )
        show()

print("Bye!")