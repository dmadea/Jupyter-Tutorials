

# import warnings
# warnings.filterwarnings("ignore")


from sympy import Function, symbols, solve, Eq
from IPython.display import display, Math
from typing import Iterable, List, Union


import numpy as np
import matplotlib.pyplot as plt
# import json
# from scipy.integrate import odeint
import re


# inspiration from https://stackoverflow.com/questions/4998629/split-string-with-multiple-delimiters-in-python
def split_delimiters(text: str, delimiters: Union[list, tuple]) -> List[tuple]:
    """
    Splits the text with denoted delimiters and returns the list of tuples in which
    first entry is the splitted text and the 2nd is the delimiter.
    """

    if not text:
        return [('', delimiters[0])]

    pattern = '|'.join(map(re.escape, delimiters))

    # it returns the list in which the every second entry is the original delimiter used
    sp = re.split(f'({pattern})', text)

    entries = sp[::2]
    delimiters = sp[1::2]

    if len(delimiters) < len(entries):
        delimiters.append('')  # append '' as a delimiter if text did not end with delimiter

    return list(zip(entries, delimiters))


class PhotoKineticSymbolicModel:

    delimiters = {
        'absorption': '-hv->',
        'reaction': '-->',
    }

    def __init__(self):
        # self.initial_conditions = {}
        self.elem_reactions = []  # list of dictionaries of elementary reactions
        self.scheme = ""
        # self.func = None  # function that returns dc_dt for current c and t for build model

    @classmethod
    def from_text(cls, scheme: str):
        """
        Expected format is single or multiline, reactions are denoted with equal sign,
        which allows for forward and backward rates. Names of species are case sensitive and can contain numbers.
        Eg. decay of triplet benzophenone with mixed 1st and 2nd order (self TT annihilation),
        here zero represents the ground state BP:
            BP3 = BP_GS  # decay of triplet BP to GS
            2BP3 = BP3 + BP_GS  # selfquenching of triplet BP

        Eg. pandemic SIR model:
            Susceptible + Infected = 2Infected
            Infected = Recovered
            Infected = Dead

        :param scheme:
            input text-based model
        :return:
            Model representing input reaction scheme.
        """

        if scheme.strip() == '':
            raise ValueError("Parameter scheme is empty!")

        _model = cls()
        _model.scheme = scheme

        # find any number of digits that are at the beginning of any characters
        pattern = re.compile(r'^(\d+).+')

        inv_delimiters = dict(zip(cls.delimiters.values(), cls.delimiters.keys()))

        for line in filter(None, scheme.split('\n')):  # filter removes empty entries from split
            line = line.strip()
            if line == '':
                continue

            line = list(filter(None, line.split('#')))[0]  # remove comments, take only the characters before possible #

            line, *rates = list(filter(None, line.split('//')))  # split to extract rate constant names

            sp_rates = None
            if len(rates) > 0:
                # delimiter for separating rate constants is semicolon ;
                sp_rates = list(map(lambda r: r.strip(), filter(None, rates[0].split(';'))))

            tokens = []

            delimiters = list(cls.delimiters.values())
            for side, delimiter in filter(lambda e: e[0] != '',  split_delimiters(line, delimiters)):  # remove empty entries
                entries = []

                # process possible number in front of species, species cannot have numbers in their text
                for entry in filter(None, side.split('+')):
                    entry = ''.join(filter(lambda d: not d.isspace(), list(entry)))  # remove white space chars

                    if entry == '':
                        continue

                    match = re.match(pattern, entry)

                    number = 1
                    if match is not None:  # number in front of a token
                        str_num = match.group(1)  #  get the number in front of the entry name
                        entry = entry[len(str_num):]
                        number = int(str_num)

                    if number < 1:
                        number = 1

                    entries += number * [entry]  # list arithmetics

                tokens.append([entries, delimiter])

            num_of_reactions = len(list(filter(lambda token: token[1] == cls.delimiters['reaction'], tokens)))

            if sp_rates and len(sp_rates) > 0:
                # if the numbers of reactions and rate constants does not match, lets use just the first entry
                if len(sp_rates) != num_of_reactions:
                    sp_rates = num_of_reactions * [sp_rates[0]]
            else:
                sp_rates = num_of_reactions * [None]

            rate_iterator = iter(sp_rates)
            for i in range(len(tokens) - 1):
                rs, r_del = tokens[i]
                ps, p_del = tokens[i + 1]

                r_type = inv_delimiters[r_del]
                r_name = next(rate_iterator) if r_type == 'reaction' else 'h\\nu'

                if r_name is None:
                    r_name = f"k_{{{'+'.join(rs)} {'+'.join(ps)} }}"

                _model.add_elementary_reaction(rs, ps, type=r_type, rate_constant_name=r_name)

                if i == len(tokens) - 2 and p_del != '':
                    r_type = inv_delimiters[p_del]
                    if r_type == 'reaction':
                        r_name = next(rate_iterator)
                    _model.add_elementary_reaction(ps, [], type=r_type, rate_constant_name=r_name)

        # comps = _model.get_compartments()
        # init = [1 if i == 0 else 0 for i in range(len(comps))]
        # _model.initial_conditions = dict(zip(comps, init))

        return _model

    def add_elementary_reaction(self, from_comp=('A', 'A'), to_comp=('B', 'C'), type='reaction', rate_constant_name=None):
        """
        type: 'reaction' or 'absorption'
        """
        from_comp = from_comp if isinstance(from_comp, (list, tuple)) else [from_comp]
        to_comp = to_comp if isinstance(to_comp, (list, tuple)) else [to_comp]

        el = dict(from_comp=from_comp, to_comp=to_comp, type=type, rate_constant_name=rate_constant_name)

        if el in self.elem_reactions:
            return

        self.elem_reactions.append(el)

    def pprint_model(self):
        """Pretty prints model."""

        latex_eq = ''

        for el in self.elem_reactions:
            reactants = ' + '.join([f'\\mathrm{{{comp}}}' for comp in el['from_comp']])
            products = ' + '.join([f'\\mathrm{{{comp}}}' for comp in el['to_comp']])

            r_name = el['rate_constant_name']
            # if r_name is None:
            #     r_name = f'k_{{{reactants}\\rightarrow{products}}}'

            latex_eq += f'{reactants} &\\xrightarrow{{{r_name}}} {products} \\\\'

        latex_eq = r'\begin{align}' + latex_eq + r'\end{align}'
        display(Math(latex_eq))


    def get_compartments(self):
        """
        Return the compartment names, the names are case sensitive.
        """
        names = []
        for el in self.elem_reactions:
            for c in el['from_comp']:
                if c not in names:
                    names.append(c)

            for c in el['to_comp']:
                if c not in names:
                    names.append(c)
        return names

    def build_equation(self):
        comps = self.get_compartments()
        n = len(comps)

        s_functions = []
        # s_equations = []
        s_rhss = n * [0]

        s_t, s_J = symbols('t J')  # time and concetrations of absorbed photons J

        for c in comps:
            f = Function(f'[{{{c}}}]')(s_t)
            # s_equations.append(Eq(f(s_t).diff(s_t), 0))  # intialize differential equations
            s_functions.append(f)


        idx_dict = dict(enumerate(comps))
        inv_idx = dict(zip(idx_dict.values(), idx_dict.keys()))

        # symbolic compartments dictionary
        # s_comps_dict = dict(zip(inv_idx.keys(), s_functions))

        r_names = list(map(lambda el: el['rate_constant_name'], filter(lambda el: el['type'] == 'reaction', self.elem_reactions)))
        s_rates = symbols(' '.join(r_names))

        # symbolic rate constants dictionary
        s_rates_dict = dict(zip(r_names, s_rates))   # 'k_S': sympy k_S


        # symbols of rate constants

        # idx_from = []  # arrays of arrays of indexes for each elementary reaction
        # idx_to = []  # arrays of arrays of indexes for each elementary reaction

        for el in self.elem_reactions:
            i_from = map(lambda com: inv_idx[com], el['from_comp'])  # list of indexes of starting materials
            i_to = map(lambda com: inv_idx[com], el['to_comp'])  # list of indexes of reaction products

            if el['type'] == 'absorption':
                for k in i_from:
                    s_rhss[k] -= s_J

                for k in i_to:
                    s_rhss[k] += s_J

                continue

            forward_prod = s_rates_dict[el['rate_constant_name']]

            for k in i_from:
                forward_prod = forward_prod * s_functions[k]  # forward rate products, eg. k_AB * [A] * [B]

            for k in i_from:
                s_rhss[k] += -forward_prod  # reactants

            for k in i_to:
                s_rhss[k] += forward_prod   # products

        # return s_rhss

        for rhs in s_rhss:
            display(rhs)

    def build_func(self):
        """
        Builds model and returns the function that takes c, t and rates as an argument
        and can be directly used for odeint numerical integration method.
        """

        comps = self.get_compartments()

        idx_dict = dict(enumerate(comps))
        inv_idx = dict(zip(idx_dict.values(), idx_dict.keys()))  # comp: index

        r = len(self.elem_reactions)
        idx_from = []  # arrays of arrays of indexes for each elementary reaction
        idx_to = []  # arrays of arrays of indexes for each elementary reaction
        _rates = np.empty((r, 2), dtype=np.float64)

        # build the lists of indexes and so on...
        for i, el in enumerate(self.elem_reactions):
            i_from = list(map(lambda com: inv_idx[com], el['from_comp']))  # list of indexes of starting materials
            i_to = list(map(lambda com: inv_idx[com], el['to_comp']))  # list of indexes of reaction products

            idx_from.append(i_from)
            idx_to.append(i_to)

            _rates[i, 0] = el['forward_rate']
            _rates[i, 1] = el['backward_rate']

        # TODO: possible space for optimization if found too slow for odeint, probably some Cython or C code would be
        # TODO: needed
        def func(c, t, rates=None):
            """Rates if provided must be (r x 2) matrix, where in first column are forward rates and second column
            backward rates. r is number of elementary reactions."""

            rates = _rates if rates is None else rates

            dc_dt = np.zeros_like(c)

            for i in range(r):
                forward_prod, backward_prod = rates[i]

                # eg. for elementary step A + B = C + D
                for k in idx_from[i]:
                    forward_prod *= c[k]  # forward rate products, eg. k_AB * [A] * [B]

                for k in idx_to[i]:
                    backward_prod *= c[k]  # backward rate products, eg. k_CD * [C] * [D]

                for k in idx_from[i]:
                    dc_dt[k] += backward_prod - forward_prod  # reactants

                for k in idx_to[i]:
                    dc_dt[k] += forward_prod - backward_prod  # products

            return dc_dt

        self.func = func
        return func

    def print_text_model(self):
        print(f'Scheme: {self.scheme}')
        # print(f'Initial conditions: {self.initial_conditions}')

        for el in self.elem_reactions:
            print(f"Elementary reaction: {' + '.join(el['from_comp'])} \u2192 {' + '.join(el['to_comp'])}, "
                  f"forward_rate: {el['forward_rate']}, backward_rate: {el['backward_rate']}")




if __name__ == '__main__':

    model = """

    BR -hv-> ^1BR --> BR // k_S  # population of singlet state and decay to GS with rate k_S

    ^1BR --> ^3BR --> BR  // k_{isc}; k_T

    ^3BR + ^3O_2 --> ^1O_2 + BR  // k_{TT}

    ^1O_2 --> ^3O_2  // k_d

    BR + ^1O_2 --> // k_r

    """

    reaction = """
    
    
    2A --> B -hv-> D --> 5C // k_{r} ; k_obs  # koasddd
    
    """
    # reaction = 'BP = zero, 2 BP = zero + BP'
    # SIR = 'Susceptible + Infected = 2 Infected, Infected = Recovered, Infected = Dead'

    model = PhotoKineticSymbolicModel.from_text(model)
    print(model.get_compartments())
    # print(model.elem_reactions)
    print(model.pprint_model())

    model.build_equation()
    #
    # model.elem_reactions[1]['forward_rate'] = 0.1
    # model.elem_reactions[2]['forward_rate'] = 0.01
    #
    # times = np.linspace(0, 50, 1000, dtype=np.float64)
    #
    # model.simulate_model(times, [1, 0.01, 0, 0])
    #
    # print(model.get_compartments())
    # print(model.get_rate_names())
    #
    #
    #







