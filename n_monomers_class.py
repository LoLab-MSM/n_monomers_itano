from pysb.bng import generate_equations
from pysb import ComplexPattern, Initial, Parameter
import numpy as np
import pandas as pd
import re
import sympy


def range1(start, end):
    return range(start, end + 1)


class NMonomersSol(object):
    """
    Parameters
    ----------
    model : pysb.Model
        Model to be analyzed
    """
    def __init__(self, model):
        self.model = model
        if not self.model.species:
            generate_equations(self.model)
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

    def get_sp_initial(self, sp):
        """

        :param sp: list of spcies indices or ComplexPatterns
        :return:
        """
        if isinstance(sp, int):
            sp_ready = [self.model.species[sp]]
        elif isinstance(sp, ComplexPattern):
            sp_ready = [sp]
        elif isinstance(sp, list):
            if all(isinstance(x, int) for x in sp):
                sp_ready = [self.model.species[s] for s in sp]
            elif all(isinstance(x, ComplexPattern) for x in sp):
                sp_ready = sp
            else:
                raise TypeError('Mixed indices and complex pattern argument is not supported')
        else:
            raise TypeError('format not supported')

        ic = [i[0] for i in self.model.initial_conditions]
        sp_ic = [0] * len(sp_ready)
        for index, s in enumerate(sp_ready):
            idx = [i for i, x in enumerate(ic) if x.is_equivalent_to(s)]
            if idx:
                ic_val = self.model.initial_conditions[idx[0]][1].value
                sp_ic[index] = ic_val

        return sp_ic

    def group_species(self):
        """

        :return: groups the species into the table in Itano's paper
        """

        sp_mon = self.species_mon_names(self.model.species)
        # dic whose keys are the index of monomers and the values are the species that contain that monomer as the
        # monomer with highest index
        mons_polymer = {i: [] for i in range1(1, len(self.model.monomers))}
        x_lm_to_cp = {}  # dict, monomer of length l and highest index m to ComplexPattern
        for i, j in sp_mon.items():
            mon_idx = self.get_number(i[-1])
            mon_length = len(i)
            mons_polymer[mon_idx].append(i)
            x_lm_to_cp[tuple([mon_length, mon_idx])] = j
        self.x_lm_to_cp = x_lm_to_cp
        # sorting lists by length
        for i in mons_polymer:
            mons_polymer[i].sort(key=len)
            species = [sp_mon[j] for j in mons_polymer[i]]
            mons_polymer[i] = species

        # dataframe to group species
        df = pd.DataFrame(np.nan, index=range1(1, (2 * len(self.model.monomers))),
                          columns=range1(1, len(self.model.monomers)))
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
        if cp not in self.model.species:
            raise ValueError('Complex Pattern does not exist')

        ic = [i[0] for i in self.model.initial_conditions]
        idx = [i for i, x in enumerate(ic) if x.is_equivalent_to(cp)]
        if not idx:
            cp_idx = self.model.get_species_index(cp)
            ic_value = Parameter('b_{0}_0'.format(cp_idx+1), 0)
            Initial(cp, ic_value)

        else:
            ic_value = self.model.initial_conditions[idx[0]][1]
        return ic_value

    def get_total_monomer(self, monomer_idx):
        """

        :param monomer_idx:
        :return: calculates the initial condition of monomers in group b_m
        """
        n = self.reaction_groups.columns[-1]
        b_ld_total = np.array([np.concatenate([self.get_groups(group='b', group_idx=i),
                                               self.get_groups(group='ld', group_idx=i)]) for i in
                               range1(monomer_idx, n)])
        if len(b_ld_total) > 1:
            b_ld_total = np.concatenate(b_ld_total)
        elif len(b_ld_total) == 1:
            b_ld_total = b_ld_total[0]

        lu_m1_total = np.array([self.get_groups(group='lu', group_idx=i + 1) for i in range1(monomer_idx, n - 1)])
        if len(lu_m1_total) > 1:
            lu_m1_total = np.concatenate(lu_m1_total)
        elif len(lu_m1_total) == 1:
            lu_m1_total = lu_m1_total[0]

        b_ld_value = 0
        b_ld_name = sympy.S.Zero
        if not b_ld_total.size:
            pass
        else:
            for ld in b_ld_total:
                b_ld_value += self.get_complex_pattern_ic(ld).value
                b_ld_name += sympy.Symbol(self.get_complex_pattern_ic(ld).name)

        lu_value = 0
        lu_name = sympy.S.Zero
        if not lu_m1_total.size:
            pass
        else:
            for lu in lu_m1_total:
                lu_value += self.get_complex_pattern_ic(lu).value
                lu_name += sympy.Symbol(self.get_complex_pattern_ic(lu).name)

        # total_monomer = [b_ld_total, lu_m1_total]
        total_value = b_ld_value - lu_value
        total_name = b_ld_name - lu_name
        return total_value, total_name

    def get_lu_m_sol(self, m):
        """

        :param m: m goes from 1 to N
        :return: the symbolic equation of the solution of the differential equation defined in the group LU_m
        """
        n = len(self.model.monomers)
        if m <= 0 or m > n:
            raise ValueError('only values between 1 and n are valid')
        k_m = self.model.rules[m - 1].rate_forward.value
        k_m_symb = self.model.rules[m - 1].rate_forward.name
        k_m_symb = sympy.Symbol(k_m_symb)
        rr = self.model.rules[m - 1].rate_reverse
        if rr:
            l_m = rr.value
            l_m_symb = sympy.Symbol(rr.name)
        else:
            l_m = 0
            l_m_symb = 0
        b_m = self.get_total_monomer(m)[0]
        b_m_symb = self.get_total_monomer(m)[1]
        t = sympy.Symbol('t')
        if m == 1:
            disc_1 = (l_m ** 2) + (8 * k_m * l_m * b_m)
            disc_1_symb = (l_m_symb ** 2) + (8 * k_m_symb * l_m_symb * b_m_symb)
            if disc_1 > 0:
                xi_ss_1_pos = (-l_m + np.sqrt(disc_1)) / (4 * k_m)
                xi_ss_1_neg = (-l_m - np.sqrt(disc_1)) / (4 * k_m)
                xi_ss_1_pos_symb = (-l_m_symb + sympy.sqrt(disc_1_symb)) / (4 * k_m_symb)
                xi_ss_1_neg_symb = (-l_m_symb - sympy.sqrt(disc_1_symb)) / (4 * k_m_symb)

                c_1 = (b_m - xi_ss_1_pos) / (b_m - xi_ss_1_neg)
                c_1_symb = (b_m_symb - xi_ss_1_pos_symb) / (b_m_symb - xi_ss_1_neg_symb)
                beta1 = k_m * (xi_ss_1_pos - xi_ss_1_neg)
                beta1_symb = k_m_symb * (xi_ss_1_pos_symb - xi_ss_1_neg_symb)
                xi_1_t = (xi_ss_1_pos - c_1 * xi_ss_1_neg * sympy.exp(-beta1 * t)) / \
                         (1 - c_1 * sympy.exp(-beta1 * t))
                xi_1_t_symb = (xi_ss_1_pos_symb - c_1_symb * xi_ss_1_neg_symb * sympy.exp(-beta1_symb * t)) / \
                         (1 - c_1_symb * sympy.exp(-beta1_symb * t))
                return [xi_1_t, xi_1_t_symb], np.array([xi_ss_1_pos, xi_ss_1_neg, xi_ss_1_pos_symb, xi_ss_1_neg_symb])

            elif disc_1 == 0:
                xi_ss_1 = 0
                xi_1_t = b_m / (k_m * b_m * t + 1)
                xi_1_t_symb = b_m_symb / (k_m_symb * b_m_symb * t + 1)
                return [xi_1_t, xi_1_t_symb], np.array([xi_ss_1])
            else:
                raise Exception('there must be something wrong with your model, check parameter values')

        elif 1 < m < n:
            b_minus1 = self.get_total_monomer(m - 1)[0]
            b_minus1_symb = self.get_total_monomer(m - 1)[1]

            disc_mplus1 = (k_m * (b_minus1 - b_m) + l_m) ** 2 + 4 * k_m * l_m * b_m
            disc_mplus1_symb = (k_m_symb * (b_minus1_symb - b_m_symb) + l_m_symb) ** 2 + 4 * k_m_symb * l_m_symb * b_m_symb
            if disc_mplus1 > 0:
                xi_ss_mplus1_pos = (-(k_m * (b_minus1 - b_m) + l_m) + np.sqrt(disc_mplus1)) / (2 * k_m)
                xi_ss_mplus1_neg = (-(k_m * (b_minus1 - b_m) + l_m) - np.sqrt(disc_mplus1)) / (2 * k_m)
                xi_ss_mplus1_pos_symb = (-(k_m_symb * (b_minus1_symb - b_m_symb) + l_m_symb) +
                                         sympy.sqrt(disc_mplus1_symb)) / (2 * k_m_symb)
                xi_ss_mplus1_neg_symb = (-(k_m_symb * (b_minus1_symb - b_m_symb) + l_m_symb) -
                                         sympy.sqrt(disc_mplus1_symb)) / (2 * k_m_symb)

                c_mplus1 = (b_m - xi_ss_mplus1_pos) / (b_m - xi_ss_mplus1_neg)
                c_mplus1_symb = (b_m_symb - xi_ss_mplus1_pos_symb) / (b_m_symb - xi_ss_mplus1_neg_symb)
                beta_mplus1 = k_m * (xi_ss_mplus1_pos - xi_ss_mplus1_neg)
                beta_mplus1_symb = k_m_symb * (xi_ss_mplus1_pos_symb - xi_ss_mplus1_neg_symb)
                xi_mplus1_t = (xi_ss_mplus1_pos - c_mplus1 * xi_ss_mplus1_neg * sympy.exp(-beta_mplus1 * t)) / \
                              (1 - c_mplus1 * sympy.exp(-beta_mplus1 * t))
                xi_mplus1_t_symb = (xi_ss_mplus1_pos_symb - c_mplus1_symb * xi_ss_mplus1_neg_symb * sympy.exp(-beta_mplus1_symb * t)) / \
                              (1 - c_mplus1_symb * sympy.exp(-beta_mplus1_symb * t))
                return [xi_mplus1_t, xi_mplus1_t_symb], np.array([xi_ss_mplus1_pos, xi_ss_mplus1_neg, xi_ss_mplus1_pos_symb, xi_ss_mplus1_neg_symb])

            elif disc_mplus1 == 0:
                xi_ss_mplus1 = 0
                xi_mplus1_t = b_m / (k_m * b_m * t + 1)
                xi_mplus1_t_symb = b_m_symb / (k_m_symb * b_m_symb * t + 1)
                return [xi_mplus1_t, xi_mplus1_t_symb], np.array([xi_ss_mplus1])
            else:
                raise Exception('there must be something wrong with your model, check parameter values')
        elif m == n:
            b_minus1 = self.get_total_monomer(m - 1)[0]
            b_minus1_symb = self.get_total_monomer(m - 1)[1]

            disc_n = ((k_m * (b_minus1 - b_m) + l_m) ** 2) + (4 * k_m * l_m * b_m)
            disc_n_symb = ((k_m_symb * (b_minus1_symb - b_m_symb) + l_m_symb) ** 2) + (4 * k_m_symb * l_m_symb * b_m_symb)
            if disc_n > 0:
                xi_ss_n_pos = (-(k_m * (b_minus1 - b_m) + l_m) + np.sqrt(disc_n)) / (2 * k_m)
                xi_ss_n_neg = (-(k_m * (b_minus1 - b_m) + l_m) - np.sqrt(disc_n)) / (2 * k_m)
                xi_ss_n_pos_symb = (-(k_m_symb * (b_minus1_symb - b_m_symb) + l_m_symb) + sympy.sqrt(disc_n_symb)) / (2 * k_m_symb)
                xi_ss_n_neg_symb = (-(k_m_symb * (b_minus1_symb - b_m_symb) + l_m_symb) - sympy.sqrt(disc_n_symb)) / (2 * k_m_symb)

                c_n = (b_m - xi_ss_n_pos) / (b_m - xi_ss_n_neg)
                c_n_symb = (b_m_symb - xi_ss_n_pos_symb) / (b_m_symb - xi_ss_n_neg_symb)
                beta_n = k_m * (xi_ss_n_pos - xi_ss_n_neg)
                beta_n_symb = k_m_symb * (xi_ss_n_pos_symb - xi_ss_n_neg_symb)
                xi_n_t = (xi_ss_n_pos - c_n * xi_ss_n_neg * sympy.exp(-beta_n * t)) / \
                         (1 - c_n * sympy.exp(-beta_n * t))
                xi_n_t_symb = (xi_ss_n_pos_symb - c_n_symb * xi_ss_n_neg_symb * sympy.exp(-beta_n_symb * t)) / \
                         (1 - c_n_symb * sympy.exp(-beta_n_symb * t))
                return [xi_n_t, xi_n_t_symb], np.array([xi_ss_n_pos, xi_ss_n_neg, xi_ss_n_pos_symb, xi_ss_n_neg_symb])

            elif disc_n == 0:
                xi_ss_n = 0
                xi_n_t = b_m / (k_m * b_m * t + 1)
                xi_n_t_symb = b_m_symb / (k_m_symb * b_m_symb * t + 1)
                return xi_n_t, np.array([xi_ss_n])

    def get_eta_m(self, m):
        """

        :param m: m goes from 1 to N
        :return: the solution of the differential equation of the eta_m group
        """
        b_m = self.get_total_monomer(m)[0]
        n = len(self.model.monomers)
        if 1 <= m <= n - 1:
            b_mplus1 = self.get_total_monomer(m + 1)[0]
            xi_mplus1_t = self.get_lu_m_sol(m + 1)[0][0]
            xi_ss_mplus1_pos, xi_ss_mplus1_neg = self.get_lu_m_sol(m + 1)[1][[0, 1]]
            eta_m = b_m - b_mplus1 + xi_mplus1_t
            eta_m_ss_pos = b_m - b_mplus1 + xi_ss_mplus1_pos
            eta_m_ss_neg = b_m - b_mplus1 + xi_ss_mplus1_neg
            return eta_m, np.array([eta_m_ss_pos, eta_m_ss_neg])
        elif m == n:
            return b_m

            # def get_species_ss(self):
            #     species_ss = {}
            #     N = len(self.model.monomers)
            #
            #     # getting solutions of the species in the LU_m group
            #     # LU_N is a special case
            #     species_ss[self.x_lm_to_cp[(1, N)]] = self.get_lu_m_sol(N)[1]
            #
            #     # LU_N-1 is a special case as well
            #     # for X(1, N-1) i = 1
            #     k_N = self.model.rules[N-1].rate_forward.value
            #     k_N_1 = self.model.rules[N-2].rate_forward.value
            #     l_N = self.model.rules[N-1].rate_reverse.value
            #     l_N_1 = self.model.rules[N-2].rate_reverse.value
            #     xi_N = self.get_lu_m_sol(N)[1]
            #     xi_N_1 = self.get_lu_m_sol(N-1)[1]
            #     eta_N = self.get_eta_m(N)[0]
            #     eta_N_1 = self.get_eta_m(N-1)[1]
            #     eta_N_2 = self.get_eta_m(N-2)[1]
            #
            #     A1 = k_N * xi_N + k_N_1 * eta_N_2 + l_N + l_N_1
            #     f1 = l_N * xi_N_1 + l_N_1 * eta_N_1
            #     species_ss[self.x_lm_to_cp[(1, N - 1)]] = f1 / A1
            #
            #     A2 = k_N_1 * eta_N_2 + l_N
            #     f2 = k_N * species_ss[self.x_lm_to_cp[(1, N)]] * species_ss[self.x_lm_to_cp[(1, N - 1)]] + \
            #          l_N * (eta_N - species_ss[self.x_lm_to_cp[(1, N)]])
            #     species_ss[self.x_lm_to_cp(2, N)] = f2 / A2
            #
            #     m_range = range1(2, N - 1)
            #     for m in m_range:
            #         i_range = range1(1, N - m + 1)
            #         for i in i_range:
            #             pol_l = i
            #             pol_m = m + i - 1
            #
            #             if i == 1:
            #                 # Getting A(1, m+1-1)
            #                 k_m = self.model.rules[m-1].rate_forward.value
            #                 k_mplus1 = self.model.rules[(m - 1) + i].rate_forward.value
            #                 ls = [self.model.rules[(m-1) + j].rate_reverse.value for j in range1(0, i)]
            #                 xi_mplus1_ss = self.get_lu_m_sol(m + i)[1]
            #                 eta_mminus1_ss = self.get_eta_m(m - 1)[1]
            #                 A_l_m = k_mplus1*xi_mplus1_ss + k_m*eta_mminus1_ss + sum(ls)
            #
            #                 # Getting f(1, m+1-1)
            #                 l_m = self.model.rules[m-1].rate_reverse.value
            #                 l_mplus1 = self.model.rules[m].rate_reverse.value
            #                 xi_m_ss = self.get_lu_m_sol(m)
            #                 eta_m_ss = self.get_eta_m(m)
            #                 f_l_m = l_mplus1 * xi_m_ss + l_m * eta_m_ss
            #
            #             if 2 <= i < N - m + 1:
            #                 # Getting A(i, m+i-1)
            #                 k_m = self.model.rules[m-1].rate_forward.value
            #                 k_mplusi = self.model.rules[(m - 1) + i].rate_forward.value
            #                 ls = [self.model.rules[m + j].rate_reverse.value for j in range1(0, i)]
            #                 xi_mplusi_ss = self.get_lu_m_sol(m + i)[1]
            #                 eta_mminus1_ss = self.get_eta_m(m - 1)[1]
            #                 A_l_m = k_mplusi*xi_mplusi_ss + k_m*eta_mminus1_ss + sum(ls)
            #
            #                 # Getting f(i, m+i-1)
            #                 k_m_j = [self.model.rules[(m-1) + j] for j in range1(1, i-1)]
            #                 x_l_mminusj = [self.x_lm_to_cp[tuple(pol_l, pol_m - j)] for j in range1(1, i-1)]
            #                 x_mjminus1_j = [self.x_lm_to_cp[tuple(m+j-1, j)] for j in range1(1, i-1)]
            #                 x_l_
            #
            #             else:
            #                 ls = [self.model.rules[m + j].rate_reverse.value for j in range1(1, N - m)]
            #                 A_l_m = k_m*eta_mminus1_ss + sum(ls)
            #
