# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:18:44 2024

@author: hugo


integrators.py
Various Integrators for differential equations
sdeint: Stochastic Differential Equations.
ddeint: Delay Differential Equations.
my_odeint: Ordinary Differential Equations (fixed order and step).



"""

import numpy as np
from numpy.random import normal
 
def sdeint(f, y0, t, D, **args):
    """
    sdeint(f, y0, t, D, **args) 
    Integrator for stochastic differential equations (SDE).
    Input arguments:
        f: callable(y,t0,...)
            Deterministic part of the differential equation (righthand side).
        y0: scalar of vector
            Initial condition.
        t: array 
            Times at which the solution is desired.
            D: scalar or array
                Noise amplitude for each component of the equation. Should have the 
                shape as y0.      
    @author: Hugo L. D. de S. Cavalcante
    """
#    n = len(t)
#    m = len(np.array([y0]))
#    y = np.zeros((n,m))
#    y[0,:] = y0
#    psi = normal(0,1,(n,m))
#    for i in xrange(1,n):
#        h = t[i]-t[i-1]
#        F1 = f(y[i-1],t[i-1], *args)
#        F2 = f(y[i-1]+h*F1+np.sqrt(2*D*h)*psi[i,:], t[i], *args)
#        y[i,:] = y[i-1,:]+h*(F1+F2)/2+np.sqrt(2*D*h)*psi[i,:]
#    return y
    n = len(t)
    if () == np.shape(y0):
        m = len(np.array([y0]))
    else:
        m = len(y0)
    y = np.zeros((n,m))
    y[0,:] = y0
    h = t[1:]-t[0:-1]
    psi = normal(0,1,(n-1,m))
    for i in range(1,n):
        F1 = f(y[i-1],t[i-1], *args)
        F2 = f(y[i-1]+h[i-1]*F1+np.sqrt(2*D*h[i-1])*psi[i-1,:], t[i], *args)
        y[i,:] = y[i-1,:]+h[i-1]*(F1+F2)/2+np.sqrt(2*D*h[i-1])*psi[i-1,:]
    return y


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
    Returns
    -------
    Solution y of the ODE dy(t)/dt = f(y,t), y(0) = y0.
    y: array with shape (n,m), where n = len(t) and m = len(y0).
    
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

def Euler_method(f, y0, t, *args):
    """Integrator using the standard (forward, explicit) Euler Method.
        For demonstrations and comparison with other methods.

    Parameters
    ----------
    f: callable(y,t0,...)
        Right-hand side the differential equation.
    y0: scalar of vector
        Initial condition.
    t: array 
        Times at which the solution is desired. Must be finely spaced.
    *args: additional (optional) arguments that will be passed to f.

    Returns
    -------
    Solution y of the ODE dy(t)/dt = f(y,t), y(0) = y0.
    y: array with shape (n,m), where n = len(t) and m = len(y0).
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
        dy = h[i]*f(y[i,:], t[i], *args)
        y[i+1,:] = y[i,:] + dy
    return y
    
def backward_Euler(f, y0, t, IIE = 3, *args):
    """Integrator using the backard (inplicit) Euler Method.
        For demonstrations and comparison with other methods.

    Parameters
    ----------
    f: callable(y,t0,...)
        Right-hand side the differential equation.
    y0: scalar of vector
        Initial condition.
    t: array 
        Times at which the solution is desired. Must be finely spaced.
    IIE: int
        Number of iterations to solve the algebraic implicit equation
    *args: additional (optional) arguments that will be passed to f.

    Returns
    -------
    Solution y of the ODE dy(t)/dt = f(y,t), y(0) = y0.
    y: array with shape (n,m), where n = len(t) and m = len(y0).
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
        dy = h[i]*f(y[i,:], t[i], *args)
        y[i+1,:] = y[i,:] + dy
        for j in range(IIE):
            dy = h[i]*f(y[i+1,:], t[i+1], *args)
            y[i+1,:] = y[i,:] + dy
    return y

def Euler_Cromer_method(f, g, x0, v0, t):
    """Integrator using the Euler-Cromer (symplectic) Method.
        For demonstrations and comparison with other methods.

    Parameters
    ----------
    f: callable(x, v, t)
        "velocities": differential equation for the configuration coordinates 
        (positions).
    g: callable(x, v, t)
        "accelerations": differential equation for the velocities.
    x0: scalar of vector
        Initial positions.
    v0: scalar of vector
        Initial velocities.
    t: array 
        Times at which the solution is desired. Must be finely spaced.

    Returns
    -------
    Solution X = (x,v) of the ODE: 
        dx(t)/dt = f(x, v, t), x(0) = x0.
        dv(t)/dt = g(x, v, t), v(0) = v0.
    X: array with shape (n,2m), where n = len(t) and m = len(x0).
    """
    #### n: number of points desired in the solution    
    n = len(t)
    #### m: number of dimensions in the phase space (solution). Equals the size 
    #### of the initial condition
    if () == np.shape(x0):
        m = len(np.array([x0]))
    else:
        m = len(x0)
        mv = len(v0)
        if m!= mv:
            print(f"The number of positions ({m}) and velocities ({mv}) is different!")
            print("Don't know how to proceed! Aborting.")
            exit(1)
            
    #### initialize the solution array with the initial condition y0
    x = np.zeros((n,m))
    v = np.zeros((n,m))
    x[0,:] = x0
    v[0,:] = v0
    h = t[1:]-t[0:-1]
    for i in range(0,n-1):
        v[i+1,:] = v[i,:] +  h[i]*g(x[i,:],v[i,:], t[i])
        x[i+1,:] = x[i,:] +  h[i]*f(x[i,:],v[i+1,:], t[i])
    X = np.zeros((n, 2*m))
    X[:,0:m] = x
    X[:,m:] = v
    return X

def Ruth4_symplectic(f, g, x0, v0, t):
    """Integrator using the Ruth (symplectic) Method of 4th order.

    Parameters
    ----------
    f: callable(x, v, t)
        "velocities": differential equation for the configuration coordinates 
        (positions).
    g: callable(x, v, t)
        "accelerations": differential equation for the velocities.
    x0: scalar of vector
        Initial positions.
    v0: scalar of vector
        Initial velocities.
    t: array 
        Times at which the solution is desired. Must be finely spaced.

    Returns
    -------
    Solution X = (x,v) of the ODE: 
        dx(t)/dt = f(x, v, t), x(0) = x0.
        dv(t)/dt = g(x, v, t), v(0) = v0.
    X: array with shape (n,2m), where n = len(t) and m = len(x0).
    """
    #### n: number of points desired in the solution    
    n = len(t)
    #### m: number of dimensions in the phase space (solution). Equals the size 
    #### of the initial condition
    if () == np.shape(x0):
        m = len(np.array([x0]))
    else:
        m = len(x0)
        mv = len(v0)
        if m!= mv:
            print(f"The number of positions ({m}) and velocities ({mv}) is different!")
            print("Don't know how to proceed! Aborting.")
            exit(1)
    #### special parameters for the method
    c1 = 1/(2*(2-2**(1/3))) 
    c2 = (1-2**(1/3))/(2*(2-2**(1/3)))
    c3 = c2
    c4 = c1
    d1 = 1/(2-2**(1/3))
    d2 = -(2**(1/3))/(2-2**(1/3))
    d3 = d1
    d4 = 0
    #### initialize the solution array with the initial condition y0
    x = np.zeros((n,m))
    v = np.zeros((n,m))
    x[0,:] = x0
    v[0,:] = v0
    h = t[1:]-t[0:-1]
    for i in range(0,n-1):
        # v[i+1,:] = v[i,:] +  d1*h[i]*g(x[i,:],v[i,:], t[i])
        # x[i+1,:] = x[i,:] +  c1*h[i]*f(x[i,:],v[i+1,:], t[i])
        # v[i+1,:] = v[i+1,:] +  d2*h[i]*g(x[i+1,:],v[i+1,:], t[i])
        # x[i+1,:] = x[i+1,:] +  c2*h[i]*f(x[i+1,:],v[i+1,:], t[i])
        # v[i+1,:] = v[i+1,:] +  d3*h[i]*g(x[i+1,:],v[i+1,:], t[i])
        # x[i+1,:] = x[i+1,:] +  c3*h[i]*f(x[i+1,:],v[i+1,:], t[i])
        # v[i+1,:] = v[i+1,:] +  d4*h[i]*g(x[i+1,:],v[i+1,:], t[i])
        # x[i+1,:] = x[i+1,:] +  c4*h[i]*f(x[i+1,:],v[i+1,:], t[i])
        v[i+1,:] = v[i,:] +  d4*h[i]*g(x[i,:],v[i,:], t[i])
        x[i+1,:] = x[i,:] +  c4*h[i]*f(x[i,:],v[i+1,:], t[i])
        v[i+1,:] = v[i+1,:] +  d3*h[i]*g(x[i+1,:],v[i+1,:], t[i])
        x[i+1,:] = x[i+1,:] +  c3*h[i]*f(x[i+1,:],v[i+1,:], t[i])
        v[i+1,:] = v[i+1,:] +  d2*h[i]*g(x[i+1,:],v[i+1,:], t[i])
        x[i+1,:] = x[i+1,:] +  c2*h[i]*f(x[i+1,:],v[i+1,:], t[i])
        v[i+1,:] = v[i+1,:] +  d1*h[i]*g(x[i+1,:],v[i+1,:], t[i])
        x[i+1,:] = x[i+1,:] +  c1*h[i]*f(x[i+1,:],v[i+1,:], t[i])
    X = np.zeros((n, 2*m))
    X[:,0:m] = x
    X[:,m:] = v
    return X




""" Execute tests and demonstrations of the integrators, if this script is 
called at top level (python integrators.py) """
if __name__ == "__main__":
    from matplotlib.pyplot import figure, plot, xlabel, ylabel, title, show, legend


    print("Testing Euler method:")
    t = np.linspace(0, 10.0, num = 101)
    h = t[1]-t[0]
    def f(X,t):
        x, v = X[0], X[1]
        return np.array([v, -x])
    X0 = np.array([1.0, 0.0])
    X = Euler_method(f, X0, t)
    x, v = X[:,0], X[:,1]
    figure("Euler", figsize=(6,6))
    title(f"Euler method with h = {h}")
    plot(x, v, 'o-', markersize = 1.2)
    theta = np.linspace(-np.pi, np.pi, num = 10001)
    x_exact = np.sin(theta)
    y_exact = np.cos(theta)
    plot(x_exact, y_exact, color = 'gray')
    xlabel("x")
    ylabel("v")
    show()
    
    print("Testing implicit (backward) Euler method:")
    t = np.linspace(0, 10.0, num = 101)
    h = t[1]-t[0]
    def f(X,t):
        x, v = X[0], X[1]
        return np.array([v, -x])
    X0 = np.array([1.0, 0.0])
    X = backward_Euler(f, X0, t)
    x, v = X[:,0], X[:,1]
    figure("Implicit Euler", figsize=(6,6))
    title(f"Implicit Euler method with h = {h}")
    plot(x, v, 'o-', markersize = 1.2)
    theta = np.linspace(-np.pi, np.pi, num = 10001)
    x_exact = np.sin(theta)
    y_exact = np.cos(theta)
    plot(x_exact, y_exact, color = 'gray')
    xlabel("x")
    ylabel("v")
    show()
    
    print("Testing Euler-Cromer (semi-implicit, symplectic) method:")
    t = np.linspace(0, 10.0, num = 101)
    h = t[1]-t[0]
    def f(x, v, t):
        return v
    def g(x, v,t):
        return -x


    X0 = np.array([1.0, 0.0])
    X = Euler_Cromer_method(f, g, X0[0], X0[1], t)
    x, v = X[:,0], X[:,1]
    figure("Euler-Cromer", figsize=(6,6))
    title(f"Euler-Cromer (symplectic) method with h = {h}")
    plot(x, v, 'o-', markersize = 1.2)
    theta = np.linspace(-np.pi, np.pi, num = 10001)
    x_exact = np.sin(theta)
    y_exact = np.cos(theta)
    plot(x_exact, y_exact, color = 'gray')
    xlabel("x")
    ylabel("v")
    show()
    
    ### All methods for ODEs in a single figure
    def f(X,t):
        x, v = X[0], X[1]
        return np.array([v, -x])
    X0 = np.array([1.0, 0.0])
    X = Euler_method(f, X0, t)
    x_Euler, v_Euler = X[:,0], X[:,1]
    X = backward_Euler(f, X0, t)
    x_implicit, v_implicit = X[:,0], X[:,1]
    def f(x, v, t):
        return v
    def g(x, v,t):
        return -x
    X0 = np.array([1.0, 0.0])
    X = Euler_Cromer_method(f, g, X0[0], X0[1], t)
    x_Euler_Cromer, v_Euler_Cromer = X[:,0], X[:,1]
    figure("Energy", figsize=(6,6))
    plot(x_Euler, v_Euler, 'o-', markersize = 1.2, label = "Euler")
    plot(x_implicit, v_implicit, 'o-', markersize = 1.2, label = "backward Euler")
    plot(x_Euler_Cromer, v_Euler_Cromer, 'o-', markersize = 1.2, label = "Euler-Cromer")
    theta = np.linspace(-np.pi, np.pi, num = 10001)
    x_exact = np.sin(theta)
    y_exact = np.cos(theta)
    plot(x_exact, y_exact, color = 'gray', label = "Exact")
    xlabel("x")
    ylabel("v")
    legend()
    show()

    print("Testing Ruth4 (symplectic 4th-order) method:")
    t = np.linspace(0, 100.0, num = 1001)
    h = t[1]-t[0]
    def f(x, v, t):
        return v
    def g(x, v, t):
        return -x
    def F(X, t):
        x, v = X[0], X[1]
        return np.array([f(x,v,t), g(x,v,t)])
    X0 = np.array([1.0, 0.0])
    X = Ruth4_symplectic(f, g, X0[0], X0[1], t)
    x_Ruth4, v_Ruth4 = X[:,0], X[:,1]
    E_Ruth4 = (x_Ruth4**2 + v_Ruth4**2)/2
    X = my_odeint(F, X0, t)
    x_RK4, v_RK4 = X[:,0], X[:,1]
    E_RK4 = (x_RK4**2 + v_RK4**2)/2
    figure("Ruth4", figsize=(6,6))
    title(f"Ruth4 (symplectic) method with h = {h}")
    plot(x_Ruth4, v_Ruth4, 'o-', markersize = 1.2)
    theta = np.linspace(-np.pi, np.pi, num = 10001)
    x_exact = np.sin(theta)
    y_exact = np.cos(theta)
    plot(x_exact, y_exact, color = 'gray')
    xlabel("x")
    ylabel("v")
    show()
    figure("Energy_4th_order")
    plot(t, E_RK4-0.5, label = "RK4")
    plot(t, E_Ruth4-0.5, label = "Ruth4")
    xlabel("time")
    ylabel("E-1/2")
    legend()
    show()
    
    