# -*- coding: utf-8 -*-
"""
pendula_3.py

Deduzindo as equações de movimento de pêndulos acoplados através de uma 
plataforma móvel, usando sympy para cálculos simbólico. Depois, podemos também
resolver as equações numericamente e fazer uma animação com os resultados.



Numerical solutions for the equations of motion of two pendula coupled by a 
sliding cart.

After the work of Gustavo G. C. de A. Dias, J. R. Rios Leite, Josué S. da Fonseca, 
(DF-UFPE). 

Created on Fri Oct 28 15:49:18 2022

@author: Hugo L. D. de S. Cavalcante
"""

# %% Imports and initialization
from sympy import symbols, Function, diff, simplify, solve, lambdify, sqrt
from sympy import init_printing, init_session, pprint, Eq, sympify
import sympy as sym
from IPython.display import display, Math
import numpy as np
from numpy import array, zeros, linspace, arange, sin, cos, pi

from matplotlib.pyplot import plot, legend, figure, xlabel, ylabel, subplots, show
from scipy.integrate import odeint, Radau, solve_ivp


### Para usar "pretty printing" em objetos do sympy
init_printing(use_latex=True)
#init_session()
print("Motion of two pendulae in a cart.")

# %% Analytic development of the equations of motion
# %%% Definitions of parameters and variables for the analytic development
### parâmetros físicos (definição simbólica)
m1, m2, mc, l1, l2, g, t = symbols('m1 m2 m_c l1 l2 g t')
gamma1, gamma2, gammac = symbols('gamma1 gamma2 gamma_c')
mu = symbols('mu')  ## rolling friction coefficient = 0.0065 (https://www.usna.edu/Users/physics/mungan/_files/documents/Publications/PhysEd4.pdf)
### variáveis dinâmicas (funções do tempo)
theta1, theta2, xc = symbols('theta1 theta2 x_c', cls=Function)
theta1, theta2, xc = theta1(t), theta2(t), xc(t)
theta1_d, theta2_d, vc = diff(theta1, t),  diff(theta2, t), diff(xc, t)
### acelerações 
theta1_dd, theta2_dd, ac = diff(theta1_d, t),  diff(theta2_d, t), diff(vc, t)
### Posições no espaço de configurações
x1 = xc + l1*sym.sin(theta1)
x2 = xc + l2*sym.sin(theta2)
y1 = -l1*sym.cos(theta1)
y2 = -l2*sym.cos(theta2)
#eval("x1"), eval("y1")
#eval("x2"), eval("y2")
pprint(Eq(sympify("x_1"), x1, evaluate=False))
#print("\n")
pprint(Eq(sympify("x_2"), x2, evaluate=False))
#print("\n")
pprint(Eq(sympify("y_1"), y1, evaluate=False))
#print("\n")
pprint(Eq(sympify("y_2"), y2, evaluate=False))
print("\n")
#print(x2, y2)
#display(Math('x_1 = '+latex(x1)))

# %%% Lagrangian and energies
###### Equações de movimento
print("Deriving equations of motion.")
### Energia cinética
T1 = 1/2*m1*(diff(x1,t)**2 + diff(y1,t)**2)
T2 = 1/2*m2*(diff(x2,t)**2 + diff(y2,t)**2)
Tc = 1/2*mc*vc**2
T = T1+T2+Tc
### Energia potencial
U1, U2 = m1*g*y1, m2*g*y2
U = U1+U2
### Lagrangeano
L = T-U
L = simplify(L)
pprint(Eq(sympify("L"), L, evaluate=False))
print("\n\n\n")
##### Termos dissipativos
### Vamos fazer a força de arrasto proporcional à velocidade total (viscosidade)
### Poderia ter também um atrito de rolamento, proporcional à velocidade angular
F1 = 1/2*gamma1*(diff(x1,t)**2+diff(y1,t)**2)
F2 = 1/2*gamma2*(diff(x2,t)**2+diff(y2,t)**2)
### A força de atrito no carrinho é uma força de rolamento (independente de vc)
### Mas vamos aproximar como uma viscosidade também
F3 = 1/2*gammac*(vc**2) 
#F3 = mu*g*(m1+m2+mc)*vc  ### Aqui era preciso subtrair as forças verticais dos pêndulos para calcular a força normal
#F3 = gammac*g*(m1+m2+mc)*sqrt(vc**2)  ### Aqui era preciso subtrair as forças verticais dos pêndulos para calcular a força normal
F = F1+F2+F3
# %%% Deriving the Euler-Lagrange equations
#### Equações de Euler-Lagrange (com termos de dissipação)
### dL/dq[i] = d/dt(dL/q_d[i]) + dF/dq_d[i]
EqL1 = (diff(diff(L,theta1_d),t)+diff(F, theta1_d)-diff(L,theta1)).simplify()
EqL2 = (diff(diff(L,theta2_d),t)+diff(F, theta2_d)-diff(L,theta2)).simplify()
EqLc = (diff(diff(L,vc),t)+diff(F, vc)-diff(L,xc)).simplify()

### Expressão analítica das equaçoes de movimento
Solutions = solve([EqL1, EqL2, EqLc], (theta1_dd, theta2_dd, ac))
Solutions = simplify(Solutions)
###imprimir as respostas
#print(Solutions[theta1_dd])
#print(Solutions[theta2_dd])
#print(Solutions[ac])
pprint(Eq(theta1_dd, Solutions[theta1_dd], evaluate=False), num_columns=130)
print("\n\n\n")
#pprint(Eq(theta1_dd, Solutions[theta1_dd], evaluate=False))
pprint(Eq(theta2_dd, Solutions[theta2_dd], evaluate=False), num_columns=130)
print("\n\n\n")
pprint(Eq(ac, Solutions[ac], evaluate=False), num_columns=130)
print("\n\n\n")

### Se necessário, podemos expressar a energia total como
E = T+U
#print(E)
pprint(Eq(sympify("Energy"), E, evaluate=False), num_columns=130)
print("\n\n\n")
print("Equations ready.")

# %% Numerical solution
####### Solução numérica
### Vamos obter as funções numéricas a partir das funções simbólicas
### x_dot[i] = f[i](X_dot)
# %%% Creating the numerical functions for the integration
f1 = lambdify(theta1_d, theta1_d)
f2 = lambdify(theta2_d, theta2_d)
f3 = lambdify(vc, vc)
df1 = lambdify((t, g, m1, m2, mc, l1, l2, gamma1, gamma2, gammac, 
                theta1, theta2, xc, theta1_d, theta2_d, vc), 
               Solutions[theta1_dd])
df2 = lambdify((t, g, m1, m2, mc, l1, l2, gamma1, gamma2, gammac, 
                theta1, theta2, xc, theta1_d, theta2_d, vc), 
               Solutions[theta2_dd])
df3 = lambdify((t, g, m1, m2, mc, l1, l2, gamma1, gamma2, gammac, 
                theta1, theta2, xc, theta1_d, theta2_d, vc), 
               Solutions[ac])

### X é o vetor no espaço de fase, a EDO é 
### dX/dt = f(X)
def f(t,X, g, m1, m2, mc, l1, l2, gamma1, gamma2, gammac):
    q1, q2, x3, v1, v2, v3 = X
    return [
        f1(v1),
        f2(v2),
        f3(v3),
        df1(t,g, m1, m2, mc, l1, l2, gamma1, gamma2, gammac, q1, q2, x3, v1, v2, v3),
        df2(t,g, m1, m2, mc, l1, l2, gamma1, gamma2, gammac, q1, q2, x3, v1, v2, v3),
        df3(t,g, m1, m2, mc, l1, l2, gamma1, gamma2, gammac, q1, q2, x3, v1, v2, v3)
        ]


# %%% Parameters for the integration
# ### Valores numéricos dos parâmetros físicos
mc = 3.0
m1 = 1.0
m2 = 1.0
l1 = 0.20
l2 = 0.19
g = 9.782
M = m1+m2+mc
gamma1 = 1e-6 #0.0 #5e-5 #0.001 #5e-5 #0.001
gamma2 = 1e-6 #0.0 #5e-5 #0.001 #5e-5 #0.001
gammac = 4e-6 #1e-3 #0.0 #0.005 #5e-3 #0.01
gammac = 0.0065

#Method = "my_odeint"
Method = "DOP853" # other choices: "RK45" (default), "RK23", "DOP853", "Radau", "BDF", "LSODA"
###time step
dt = 0.01
#tf = (N-1)*dt
tf = 100
N = int(tf/dt)+1


# %%% Numerical integration
### Condição inicial
#X0 = array([1.9, -1.95, 0.0, 0.0, 0.0, 0.0])
X0 = array([0.2, -0.21, 0.0, 0.0, 0.0, 0.0])
time = linspace(0.0, tf, N)


print("Starting numerical integration.")
Solution = solve_ivp(f, (0.0, tf), X0, method = Method, t_eval = time, 
                     rtol = 5e-14, atol = 1e-16, args=(g, m1, m2, mc, l1, l2, 
                      gamma1, gamma2, gammac))
#Solution = solve_ivp(f, (0.0, tf), X0, method = Method, t_eval = t)
message = Solution["message"]
print(message)
print("Finished numerical integration.")
X = Solution["y"]
## When t is in the columns (shape(X) = (n,N), n = 6, N ~ thousands)
q1, q2, q3 = X[0,:], X[1,:], X[2,:]
qp1, qp2, qp3 = X[3,:], X[4,:], X[5,:]

### Energia total E = T+V
E = 0.5*((m1*(l1*qp1)**2)+(m2*(l2*qp2)**2)+M*qp3**2)+qp3*(m1*l1*qp1*cos(q1)+m2*l2*qp2*cos(q2))-(m1*l1*g*cos(q1)+m2*l2*g*cos(q2)) +m1*l1*g+m2*l2*g
qa = (q1-q2)/2
qs = (q1+q2)/2
E1 = 0.5*(m1*(l1*qp1)**2)+qp3*m1*l1*qp1*cos(q1)-m1*l1*g*cos(q1)+m1*l1*g
E2 = 0.5*(m2*(l2*qp2)**2)+qp3*m2*l2*qp2*cos(q2)-m2*l2*g*cos(q2)+m2*l2*g
E3 = 0.5*M*qp3**2
##plot(t, qa, label = '$q_a$')
##plot(t, qs, label = '$q_s$')
#plot(t, q1, label = '$q_1$')
#plot(t, q2, label = '$q_2$')
#plot(t, qc, label = '$q_c$')


# %%% Plotting results
print("Plotting results.")
fig1, (ax1, ax2) = subplots(nrows = 2, ncols = 1, sharex = True)
#ax1.plot(time, q1, label = '$q_1$')
#ax1.plot(time, q2, label = '$q_2$')
#ax1.plot(time, q3, label = '$q_3$')
ax1.plot(time, qp1, label = '$\\dot{q}_1$')
ax1.plot(time, qp2, label = '$\\dot{q}_2$')
ax1.plot(time, qp3, label = '$\\dot{q}_3$')

#ax1.plot(t[start_i::jump], phase_wrap(q1[start_i::jump]), label = '$q_1$')
#ax1.plot(t[start_i::jump], phase_wrap(q2[start_i::jump]), label = '$q_2$')
#ax1.plot(t[start_i::jump], qp1[start_i::jump], label = '$\\dot{q}_1$')
#ax1.plot(t[start_i::jump], qp2[start_i::jump], label = '$\\dot{q}_2$')
# ax1.plot(t[start_i::jump], phase_wrap(qs[start_i::jump]), label = '$q_s$')
# ax1.plot(t[start_i::jump], phase_wrap(qa[start_i::jump]), label = '$q_a$')
# ax1.plot(t[start_i::jump], vc[start_i::jump], label = '$v_c$')
ax1.set_ylabel('$q_i$, $\\dot{q}_i$')
ax1.legend(loc=1)

ax2.plot(time, E, label = '$E$')
ax2.plot(time, E1, label = '$E_1$')
ax2.plot(time, E2, label = '$E_2$')
ax2.plot(time, E3, label = '$E_3$')
# ax2.plot(t[start_i::jump], E[start_i::jump], label = '$E$')
#plot(t[::10], E1[::10], label = '$E_A$')
#plot(t[::10], E2[::10], label = '$E_B$')
ax2.set_xlabel('time')
ax2.set_ylabel('$E$')
ax2.legend(loc=1)

print("Have a nice day, and don't forget to smile.")

