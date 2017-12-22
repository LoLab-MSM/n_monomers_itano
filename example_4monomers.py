from n_monomers_model import *
from n_monomers_class import NMonomersSol
from pysb.simulator import ScipyOdeSimulator
import numpy as np
import matplotlib.pyplot as plt
import sympy

model = generate_n_monomers_model(4)
mon4 = NMonomersSol(model)

update_initial_conditions(model, {0: 1e-6, 1: 1.57e-7, 2: 1e-6})
update_kinetic_parameters(model, {0: 1e-2, 1: 2e6, 2: 2.74e6, 3: 2e-4, 4: 2.1e7})

print (mon4.get_species_ss())

tspan = np.linspace(0, 5, 50)
y = ScipyOdeSimulator(model, tspan).run().all

plt.figure()
plt.plot(tspan, y['__s1'], label='s1')
plt.plot(tspan, y['__s2'], label='s2')
plt.plot(tspan, y['__s3'], label='s3')
plt.plot(tspan, y['__s6'], label='s6')
plt.plot(tspan, y['__s7'], label='s7')
plt.legend()
plt.show()