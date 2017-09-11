from pysb import *
import re
from pysb.bng import generate_equations
import pandas as pd
import numpy as np
import sympy

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text) ]


def n_monomers_model(n):
    monomers = ['b_{0}'.format(i) for i in range(1, n+1)]
    mons = [0] * n
    for i, j in enumerate(monomers):
        if i == len(monomers)-1:
            mons[i] = Monomer(j, ["s{0}".format(1)])
        else:
            mons[i] = Monomer(j, ["s{0}".format(1), "s{0}".format(2)])
    return mons


def n_rate_constants(n):
    n_constants = (n-1) * 2 + 1
    params = [0] * n_constants
    values = range(1, n_constants, 2)
    for i, j in enumerate(values):
        params[j-1] = Parameter('k{0}'.format(i+1), 1)
        params[j] = Parameter('l{0}'.format(i+1), 1)

    params[n_constants - 1] = Parameter('k{0}'.format(n), 1)
    return params


def n_rules(n):
    mon = n_monomers_model(n)
    # Initial conditions
    for i in mon:
        state_sites = {j: None for j in i.sites}
        Initial(i(state_sites), Parameter(i.name+'_0', 5))

    pars = n_rate_constants(n)
    for i in range(n):
        if i == 0:
            m1_site = mon[i].sites[1]
            Rule('rule_{0}'.format(i+1), mon[i]({m1_site: None}) + mon[i]({m1_site: None}) <> mon[i]({m1_site: 1}) %
                 mon[i]({m1_site: 1}), pars[i], pars[i+1])

        elif i == n-1:
            m1_site = mon[i].sites[0]
            m2_site = mon[i-1].sites[0]
            Rule('rule_{0}'.format(i+1), mon[i]({m1_site: None}) + mon[i-1]({m2_site: None}) >> mon[i]({m1_site: 1}) %
                 mon[i-1]({m2_site: 1}), pars[-1])

        else:
            m1_site = mon[i].sites[1]
            m2_site = mon[i-1].sites[0]
            Rule('rule_{0}'.format(i+1), mon[i]({m1_site: None}) + mon[i-1]({m2_site: None}) <> mon[i]({m1_site: 1}) %
                 mon[i-1]({m2_site: 1}), pars[2*i], pars[(2*i)+1])
    return


def generate_model(n):
    Model()
    n_rules(n)
    return model
