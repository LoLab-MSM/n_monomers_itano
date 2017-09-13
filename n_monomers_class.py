from pysb.bng import generate_equations
import numpy as np
import pandas as pd
import re
import sympy


class NMonomersSol(object):
    def __init__(self, model):
        self.model = model
        self.reaction_groups = self.group_species()

    @staticmethod
    def species_mon_names(species):
        """

        :param species: pysb species or list of pysb species
        :return: dictionary whose keys are the monomers and the values are the ComplexPattern of a species
        """
        if type(species) != list:
            species = [species]
        sp_dict = {}
        for i in species:
            bla = tuple([j.monomer.name for j in i.monomer_patterns])
            sp_dict[bla] = i
        return sp_dict

    @staticmethod
    def get_number(string):
        """

        :param string: a string
        :return: the number that appears in string
        """
        bla = [int(s) for s in re.findall(r'\d+', string)]
        if len(bla) == 1:
            return bla[0]
        else:
            return bla

    def group_species(self):
        """

        :return: groups the species into the table in itano's paper
        """
        if not self.model.species:
            generate_equations(self.model)
        sp_mon = self.species_mon_names(self.model.species)
        # dic whose keys are the index of monomers and the values are the species that contain that monomer as the
        # monomer with highest index
        mons_polymer = {i: [] for i in range(1, len(self.model.monomers) + 1)}
        for i, j in sp_mon.items():
            mon_idx = self.get_number(i[-1])
            mons_polymer[mon_idx].append(i)

        # sorting lists by length
        for i in mons_polymer:
            mons_polymer[i].sort(key=len)
            species = [sp_mon[j] for j in mons_polymer[i]]
            mons_polymer[i] = species

        # dataframe to group species
        df = pd.DataFrame(np.nan, index=range(1, (2 * len(self.model.monomers)) + 1),
                          columns=range(1, len(self.model.monomers) + 1))
        for i, j in mons_polymer.items():
            df.loc[:len(j), i] = j

        return df

    def get_groups(self, group, group_idx):
        """

        :param group: str, can be 'lu' or 'ld'
        :param group_idx: group index to get
        :return: a list of complex patterns that belong to a specific group
        """
        if group_idx > len(self.reaction_groups.columns):
            raise ValueError('group_idx larger than number of monomers')

        group_fixed_idx = group_idx - 1
        if group == 'ld':
            ld_idx = np.negative(group_idx)
            diagonal = pd.Series(np.diag(self.reaction_groups, ld_idx))
            #  removing nans
            # ld_sum = np.sum(diagonal.dropna())
            return diagonal.dropna()
        elif group == 'lu':
            # lu_sum = np.sum(np.diag(df, group_fixed_idx))
            return np.diag(self.reaction_groups, group_fixed_idx)
        elif group == 'b':
            # b_sum = np.sum(df[group_idx].dropna())
            return self.reaction_groups[group_idx].dropna()
        else:
            raise ValueError('Parameter value not valid')

    def get_complex_pattern_ic(self, cp):
        """

        :param cp: PySB complex pattern
        :return: gets the initial conditions of a complex pattern
        """
        ic = [i[0] for i in self.model.initial_conditions]
        idx = [i for i, x in enumerate(ic) if x.is_equivalent_to(cp)]
        if not idx:
            ic_value = 0
        else:
            ic_value = self.model.initial_conditions[idx[0]][1].value
        return ic_value

    def get_total_monomer(self, monomer_idx):
        """

        :param monomer_idx:
        :return: calculates the initial condition of monomers in group b_m
        """
        n = self.reaction_groups.columns[-1]
        b_ld_total = np.array([np.concatenate([self.get_groups(group='b', group_idx=i),
                                               self.get_groups(group='ld', group_idx=i)]) for i in
                               range(monomer_idx, n + 1)])
        if len(b_ld_total) > 1:
            b_ld_total = np.concatenate(b_ld_total)
        elif len(b_ld_total) == 1:
            b_ld_total = b_ld_total[0]

        lu_m1_total = np.array([self.get_groups(group='lu', group_idx=i + 1) for i in range(monomer_idx, n)])
        if len(lu_m1_total) > 1:
            lu_m1_total = np.concatenate(lu_m1_total)
        elif len(lu_m1_total) == 1:
            lu_m1_total = lu_m1_total[0]

        b_ld_value = 0
        if not b_ld_total.size:
            pass
        else:
            for ld in b_ld_total:
                b_ld_value += self.get_complex_pattern_ic(ld)

        lu_value = 0
        if not lu_m1_total.size:
            pass
        else:
            for lu in lu_m1_total:
                lu_value += self.get_complex_pattern_ic(lu)

        # total_monomer = [b_ld_total, lu_m1_total]
        total_value = b_ld_value - lu_value
        return total_value

    def get_lu_m_sol(self, m):
        """

        :param m: m goes from 1 to N-1
        :return:
        """
        n = len(self.model.monomers)
        if m <= 0 or m > n:
            raise ValueError('only values between 1 and n are valid')
        k_m = self.model.rules[m - 1].rate_forward.value
        rr = self.model.rules[m - 1].rate_reverse
        if rr:
            l_m = rr.value
        else:
            l_m = 0
        b_m = self.get_total_monomer(m)
        t = sympy.Symbol('t')
        if m == 1:
            disc_1 = l_m ** 2 + 8 * k_m * l_m * b_m
            if disc_1 > 0:
                xi_ss_1_pos = (-l_m + np.sqrt(disc_1)) / (4 * k_m)
                xi_ss_1_neg = (-l_m - np.sqrt(disc_1)) / (4 * k_m)

                c_1 = (b_m - xi_ss_1_pos) / (b_m - xi_ss_1_neg)
                beta1 = k_m * (xi_ss_1_pos - xi_ss_1_neg)
                xi_1_t = (xi_ss_1_pos - c_1 * xi_ss_1_neg * sympy.exp(-beta1 * t)) / \
                         (1 - c_1 * sympy.exp(-beta1 * t))
                return xi_1_t, [xi_ss_1_pos, xi_ss_1_neg]

            elif disc_1 == 0:
                xi_ss_1 = 0
                xi_1_t = b_m / (k_m * b_m * t + 1)
                return xi_ss_1, xi_1_t
            else:
                raise Exception('there must be something wrong with your model, check parameter values')

        elif 1 < m < n:
            b_minus1 = self.get_total_monomer(m - 1)

            disc_mplus1 = (k_m * (b_minus1 - b_m) + l_m) ** 2 + 4 * k_m * l_m * b_m
            if disc_mplus1 > 0:
                xi_ss_mplus1_pos = (-(k_m * (b_minus1 - b_m) + l_m) + np.sqrt(disc_mplus1)) / (2 * k_m)
                xi_ss_mplus1_neg = (-(k_m * (b_minus1 - b_m) + l_m) - np.sqrt(disc_mplus1)) / (2 * k_m)

                c_mplus1 = (b_m - xi_ss_mplus1_pos) / (b_m - xi_ss_mplus1_neg)
                beta_mplus1 = k_m * (xi_ss_mplus1_pos - xi_ss_mplus1_neg)
                xi_mplus1_t = (xi_ss_mplus1_pos - c_mplus1 * xi_ss_mplus1_neg * sympy.exp(-beta_mplus1 * t)) / \
                              (1 - c_mplus1 * sympy.exp(-beta_mplus1 * t))
                return xi_mplus1_t, [xi_ss_mplus1_pos, xi_ss_mplus1_neg]

            elif disc_mplus1 == 0:
                xi_ss_mplus1 = 0
                xi_mplus1_t = b_m / (k_m * b_m * t + 1)
                return xi_mplus1_t, xi_ss_mplus1
            else:
                raise Exception('there must be something wrong with your model, check parameter values')
        elif m == n:
            b_minus1 = self.get_total_monomer(m - 1)

            disc_n = ((k_m * (b_minus1 - b_m) + l_m) ** 2) + (4 * k_m * l_m * b_m)
            if disc_n > 0:
                xi_ss_n_pos = (-(k_m * (b_minus1 - b_m) + l_m) + np.sqrt(disc_n)) / (2 * k_m)
                xi_ss_n_neg = (-(k_m * (b_minus1 - b_m) + l_m) - np.sqrt(disc_n)) / (2 * k_m)

                c_n = (b_m - xi_ss_n_pos) / (b_m - xi_ss_n_neg)
                beta_n = k_m * (xi_ss_n_pos - xi_ss_n_neg)
                xi_n_t = (xi_ss_n_pos - c_n * xi_ss_n_neg * sympy.exp(-beta_n * t)) / \
                         (1 - c_n * sympy.exp(-beta_n * t))
                return xi_n_t, [xi_ss_n_pos, xi_ss_n_neg]

            elif disc_n == 0:
                xi_ss_n = 0
                xi_n_t = b_m / (k_m * b_m * t + 1)
                return xi_n_t, xi_ss_n

    def get_eta_m(self, m):
        b_m = self.get_total_monomer(m)
        n = len(self.model.monomers)
        if 1 <= m < n-1:
            b_mplus1 = self.get_total_monomer(m + 1)
            xi_ss_mplus1_t, xi_ss_mplus1_pos, xi_ss_mplus1_neg = self.get_lu_m_sol(m)
            eta_m = b_m - b_mplus1 + xi_ss_mplus1_t
            eta_m_ss_pos = b_m - b_mplus1 + xi_ss_mplus1_pos
            eta_m_ss_neg = b_m - b_mplus1 + xi_ss_mplus1_neg
            return eta_m, [eta_m_ss_pos, eta_m_ss_neg]
        elif m == n:
            return b_m

