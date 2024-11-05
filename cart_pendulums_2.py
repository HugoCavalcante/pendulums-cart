# -*- coding: utf-8 -*-
"""
pendula_2.py

Solução numérica das equações de movimento de pêndulos acoplados por plataforma
móvel. 

Soluções com pêndulos não-idênticos (as massas e comprimentos podem ser diferentes).

Numerical solutions for the equations of motion of two pendula coupled by a 
sliding cart.

After the work of Gustavo G. C. de A. Dias, J. R. Rios Leite, Josué S. da Fonseca, 
(DF-UFPE). 

Created on Wed Jun 22 16:18:48 2022

@author: Hugo L. D. de S. Cavalcante
"""

from numpy import *
import numpy as np
from matplotlib.pyplot import *
from scipy.integrate import odeint, Radau, solve_ivp


### number of points in time series
#N = 10000

#Method = "my_odeint"
Method = "DOP853" # other choices: "RK45" (default), "RK23", "DOP853", "Radau", "BDF", "LSODA"
###time step
dt = 0.01
#tf = (N-1)*dt
#tf = 100
tf = 500
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
gamma1 = 1e-4 #4e-6 #0.0 #5e-5 #0.001 #5e-5 #0.001
gamma2 = 1e-4 #4e-6 #0.0 #5e-5 #0.001 #5e-5 #0.001
gammac = 0.0 #12*gamma1 #1e-3 #0.0 #0.005 #5e-3 #0.01



### parameters for the plot
### time step (in units of dt) in the plot (graph)
jump = 1 #10
### start time (in units of dt) in the plot (graph)
start_i = 0 #1900000

### parameters for the animation
jump_anim = 10
### time window for the history plot (memory trace) in the animation
index_window = int(10/dt/jump_anim)
### width of the animation plot "wrapper" (limits large displacements of xc)
W_stage = 6.2


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
X0 = array([0.7, -0.7, 0.0, -2.3*1.00001, 2.3, 0.0])

print("Start")


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

if Method == "my_odeint":
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


print("Plotting results.")
fig1, (ax1, ax2) = subplots(nrows = 2, ncols = 1, sharex = True, figsize=(10.67,6))
#ax1.plot(t[start_i::jump], phase_wrap(q1[start_i::jump]), label = '$q_1$')
#ax1.plot(t[start_i::jump], phase_wrap(q2[start_i::jump]), label = '$q_2$')
#ax1.plot(t[start_i::jump], qp1[start_i::jump], label = '$\\dot{q}_1$')
#ax1.plot(t[start_i::jump], qp2[start_i::jump], label = '$\\dot{q}_2$')
ax1.plot(t[start_i::jump], phase_wrap(qs[start_i::jump]), label = '$q_s$')
ax1.plot(t[start_i::jump], phase_wrap(qa[start_i::jump]), label = '$q_a$')
ax1.plot(t[start_i::jump], vc[start_i::jump], label = '$v_c$')
ax1.set_ylabel('$q_i$, $\\dot{q}_i$')
ax1.legend(loc=1)

ax2.plot(t[start_i::jump], E[start_i::jump], label = '$E$')
#plot(t[::10], E1[::10], label = '$E_A$')
#plot(t[::10], E2[::10], label = '$E_B$')
ax2.set_xlabel('time')
ax2.set_ylabel('$E$')
ax2.legend(loc=1)

### Finished solution and static plot


### animation
import matplotlib.animation as animation
(x1,y1) = xc +l1*(sin(q1)-1.1),   l1*(1.2-cos(q1))
(x2,y2) = xc +l2*(sin(q2)+1.1),   l2*(1.2-cos(q2))
#(x1,y1) = pos_wrap(xc) +l1*(sin(q1)-1.1),   l1*(1.2-cos(q1))
#(x2,y2) = pos_wrap(xc) +l2*(sin(q2)+1.1),   l2*(1.2-cos(q2))
#Animation_Data = array([x1, x2, y1, y2, pos_wrap(xc), [0]*len(xc)])
Animation_Data = array([x1, x2, y1, y2, xc, [0]*len(xc)])
Animation_Data = Animation_Data[:,start_i::jump_anim]
Text_Template = 'time = %.1f s,   E = %.6f (a.u.)'
#Text_Template = 'time = {:.2g} s,   E = {:.7g} (a.u.)'

#fig_anim, ax_anim = subplots()
fig_anim = figure(figsize=(10.67,6))
ax_anim = fig_anim.add_subplot(autoscale_on=False)
ax_anim.set_ylim(-0.1, 2.5*max(l1,l2))
#ax_anim.set_xlim(-3.1, 3.1)
ax_anim.set_xlim(-W_stage/2, W_stage/2)
ax_anim.set_aspect('equal')
ax_anim.grid()
ax_anim.set_xlabel('$x_i$ ($l_i$)')
ax_anim.set_ylabel('$y_i$ ($l_i$)')
line1, = ax_anim.plot([0], [0],  '-', color = "red")
line2, = ax_anim.plot([0], [0],  '-', color = "blue")
line3, = ax_anim.plot([0], [0],  '-', color = "green")
P1, = ax_anim.plot([0], [0],  'o', color = "red")
P2, = ax_anim.plot([0], [0],  'o', color = "blue")
P3, = ax_anim.plot([0], [0],  'o', color = "green")
B1, = ax_anim.plot([0], [0],  '-', color = "black")
B2, = ax_anim.plot([0], [0],  '-', color = "black")
time_text = ax_anim.text(0.05, 0.95, '', transform = ax_anim.transAxes)


def animate(i):
    if i<index_window:
        x1_i, x2_i, y1_i, y2_i, xc_i, yc_i = Animation_Data[:,:i]
    else:
        x1_i, x2_i, y1_i, y2_i, xc_i, yc_i = Animation_Data[:,i-index_window:i]
    line1.set_data(x1_i,y1_i)
    line2.set_data(x2_i,y2_i)
    line3.set_data(xc_i,yc_i)
    time_text.set_text(Text_Template % (i*dt*jump_anim,E[int(i*jump_anim)]))
    #time_text.set_text(Text_Template % (i*dt*jump_anim,E[int(i*jump_anim)]))
    #time_text.set_text(Text_Template.format(i*jump_anim,E[int(i*jump_anim)]))

    if i>0:
        P1.set_data(x1_i[-1],y1_i[-1])
        P2.set_data(x2_i[-1],y2_i[-1])
        P3.set_data(xc_i[-1],yc_i[-1])
        B1.set_data([x1_i[-1], xc_i[-1]-1.1*l1],[y1_i[-1], 1.2*l1])
        B2.set_data([x2_i[-1], xc_i[-1]+1.1*l2],[y2_i[-1], 1.2*l2])
        stage_center = (W_stage)*int(xc_i[-1]/(W_stage))
        ax_anim.set_xlim(stage_center-W_stage/2, stage_center+W_stage/2)

    return line1, line2, line3, P1, P2, P3, B1, B2, time_text


print("Starting animation.")
ani = animation.FuncAnimation(fig_anim, animate, len(Animation_Data[0]), interval = 0.6,  blit=False)

#plt.show()
show()

print("Bye!")