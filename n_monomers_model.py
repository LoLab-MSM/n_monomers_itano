from pysb import *
from pysb.core import ComplexPattern


def generate_n_monomers_model(n_mons):
    def n_monomers_model(n):
        monomers = ['b_{0}'.format(i) for i in range(1, n + 1)]
        mons = [0] * n
        for i, j in enumerate(monomers):
            if i == len(monomers) - 1:
                mons[i] = Monomer(j, ["s{0}".format(1)])
            else:
                mons[i] = Monomer(j, ["s{0}".format(1), "s{0}".format(2)])
        return mons

    def n_rate_constants(n):
        n_constants = (n - 1) * 2 + 1
        params = [0] * n_constants
        values = range(1, n_constants, 2)
        for i, j in enumerate(values):
            params[j - 1] = Parameter('k{0}'.format(i + 1), 1)
            params[j] = Parameter('l{0}'.format(i + 1), 1)

        params[n_constants - 1] = Parameter('k{0}'.format(n), 1)
        return params

    def n_rules(n):
        mon = n_monomers_model(n)
        # Initial conditions
        for i in mon:
            state_sites = {j: None for j in i.sites}
            Initial(i(state_sites), Parameter(i.name + '_0', 5))

        pars = n_rate_constants(n)
        for i in range(n):
            if i == 0:
                m1_site = mon[i].sites[1]
                Rule('rule_{0}'.format(i + 1),
                     mon[i]({m1_site: None}) + mon[i]({m1_site: None}) <> mon[i]({m1_site: 1}) %
                     mon[i]({m1_site: 1}), pars[i], pars[i + 1])

            elif i == n - 1:
                m1_site = mon[i].sites[0]
                m2_site = mon[i - 1].sites[0]
                Rule('rule_{0}'.format(i + 1),
                     mon[i]({m1_site: None}) + mon[i - 1]({m2_site: None}) >> mon[i]({m1_site: 1}) %
                     mon[i - 1]({m2_site: 1}), pars[-1])

            else:
                m1_site = mon[i].sites[1]
                m2_site = mon[i - 1].sites[0]
                Rule('rule_{0}'.format(i + 1),
                     mon[i]({m1_site: None}) + mon[i - 1]({m2_site: None}) <> mon[i]({m1_site: 1}) %
                     mon[i - 1]({m2_site: 1}), pars[2 * i], pars[(2 * i) + 1])
        return

    n_monomer = Model()
    n_rules(n_mons)
    return n_monomer


def update_initial_conditions(model, species):
    """

    :param model: model whose initial conditions are going to be updated
    :param species: species to update their initial conditions, can be a dict with keys as species' indices or
    complex patterns and values as the new initial conditions levels
    :return:
    """

    if all(isinstance(x, int) for x in species.keys()):
        for sp, val in species.items():
            model.initial_conditions[sp][1].value = val
    elif all(isinstance(x, ComplexPattern) for x in species.keys()):
        ic = [i[0] for i in model.initial_conditions]
        for cp, val in species.items():
            idx = [i for i, x in enumerate(ic) if x.is_equivalent_to(cp)]
            if idx:
                model.initial_conditions[idx[0]][1].value = val
            else:
                print('Complex pattern ' + str(cp) + 'is not in the list of the initials conditions of the model')
    else:
        raise TypeError('Mixed indices and complex pattern argument is not supported')


def update_kinetic_parameters(model, kpar):
    """

    :param model: model whose initial conditions are going to be updated
    :param kpar: species to update their initial conditions, can be a dict with keys as species' indices or
    complex patterns and values as the new initial conditions levels
    :return:
    """

    if all(isinstance(x, int) for x in kpar.keys()):
        for par, val in kpar.items():
            model.parameters_rules()[par].value = val
    elif all(isinstance(x, str) for x in kpar.keys()):
        par_names = [i.name for i in model.parameters_rules()]
        for name, val in kpar.items():
            idx = par_names.index(name)
            model.parameters_rules()[idx].value = val

    else:
        raise TypeError('Mixed indices and complex pattern argument is not supported')