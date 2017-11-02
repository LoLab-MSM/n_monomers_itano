from n_monomers_model import *
from n_monomers_class import NMonomersSol
from pysb.simulator import ScipyOdeSimulator
import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import sympy

model = generate_n_monomers_model(3)
mon3 = NMonomersSol(model)

update_initial_conditions(model, {0: 1e-6, 1: 1.57e-7, 2: 1e-6})
update_kinetic_parameters(model, {0: 1e-2, 1: 2e6, 2: 2.74e6, 3: 2e-4, 4: 2.1e7})

tspan = np.linspace(0, 5, 50)
y = ScipyOdeSimulator(model, tspan).run().all
t = sympy.Symbol('t')

plt.figure(1)
exp3 = mon3.get_lu_m_sol(3)[0][0]
f3 = sympy.lambdify(t, exp3, 'numpy')
plt.plot(tspan, f3(tspan), color='r', linestyle=':', linewidth=6, label='LU_3')
plt.plot(tspan, y['__s2'], color='b')

exp2 = mon3.get_lu_m_sol(2)[0][0]
f2 = sympy.lambdify(t, exp2, 'numpy')
plt.plot(tspan, f2(tspan), color='k', linestyle=':', linewidth=6, label='LU_2')
plt.plot(tspan, y['__s1'] + y['__s5'], color='b')

exp1 = mon3.get_lu_m_sol(1)[0][0]
f1 = sympy.lambdify(t, exp1, 'numpy')
plt.plot(tspan, f1(tspan), color='g', linestyle=':', linewidth=6, label='LU_1')
plt.plot(tspan, y['__s0'] + y['__s4'] + y['__s8'], color='b', label='simulation')
plt.legend()
plt.savefig('/Users/dionisio/Desktop/monomer.png', bbox_inches='tight', dpi=400)