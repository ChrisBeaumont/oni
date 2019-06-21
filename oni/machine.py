from copy import deepcopy
from collections import OrderedDict

import pandas as pd
import numpy as np
import yaml
import os

from .resource import _get_or_create_resource, LIQUIDS, GASES
from .util import as_list

path = os.path.join(os.path.dirname(__file__), 'machine.yaml')
with open(path) as infile:
    manifest = yaml.safe_load(infile)

__all__ = ['create_machine', 'all_machines', 'Source', 'Sink', 'System', 'MACHINES', 'MultiMachine']

MACHINES = []


def all_machines():
    return sorted(MACHINES, key=str)


def create_machine(name, needs=None, gives=None):
    needs = needs or {}
    gives = gives or {}

    def _needs(self):
        return needs

    def _gives(self):
        return gives

    result = type(name, (Machine,), dict(_needs=_needs, _gives=_gives))
    MACHINES.append(result)
    __all__.append(name)
    return result


def _parse_machine(name, contents):
    needs = {
        _get_or_create_resource(r): q
        for r, q in
        contents.get('needs', {}).items()
    }

    gives = {
        _get_or_create_resource(r): q
        for r, q in
        contents.get('gives', {}).items()
    }

    return create_machine(name, needs=needs, gives=gives)


class Machine:
    _greedy = True  # Prefer to run machine as much as possible

    def __init__(self):
        self.uptime = 1

    @property
    def label(self):
        return type(self).__name__

    def throttle(self, uptime):
        self.uptime = uptime
        return self

    def needs(self):
        return {k: v * self.uptime for k, v in self._needs().items()}

    def gives(self):
        return {k: v * self.uptime for k, v in self._gives().items()}

    def net_output(self):
        """
        Return the net output (gives - needs) from the system.

        If drop=True, resources with near-zero net output will be excluded.
        """
        result = self.gives()
        for k, v in self.needs().items():
            result[k] = result.get(k, 0) - v

        return result

    def __repr__(self):
        return self.label

    def __str__(self):
        return "{} [{:0.1f}% Uptime]".format(self.label, (self.uptime * 100))

    def __mul__(self, count):
        if not isinstance(count, int):
            raise ValueError("Not an ingeger: {}".format(count))

        return System(MultiMachine(self, count))

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        if isinstance(other, Machine):
            return System(self, other)
        elif isinstance(other, System):
            return System(*((self,) + other.subsystems))
        raise TypeError("Unexpected Type: %s" % type(other))


class Source(Machine):
    _greedy = False

    def __init__(self, resource, qty):
        super().__init__()
        self.resource = resource
        self.qty = qty

    @property
    def label(self):
        return 'Source({}, {})'.format(self.resource.__name__, self.qty)

    def _needs(self):
        return {}

    def _gives(self):
        return {self.resource: self.qty}


class MultiMachine(Machine):

    def __init__(self, machine, qty):
        super().__init__()
        self.machine = machine
        self.qty = qty

    @property
    def _greedy(self):
        return self.machine._greedy

    @property
    def label(self):
        return '{}x {}'.format(self.qty, self.machine.label)

    def _needs(self):
        return {k: v * self.qty for k, v in self.machine._needs().items()}

    def _gives(self):
        return {k: v * self.qty for k, v in self.machine._gives().items()}


class Sink(Machine):

    def __init__(self, resource, qty):
        super().__init__()
        self.resource = resource
        self.qty = qty

    @property
    def label(self):
        return 'Sink({}, {})'.format(self.resource.__name__, self.qty)

    def _gives(self):
        return {}

    def _needs(self):
        return {self.resource: self.qty}


def _expand_wildcards(name, contents):

    for wc, replacements in [('GAS', GASES), ('LIQUID', LIQUIDS)]:
        if wc in name:
            for r in replacements:
                n = name.replace(wc, r.__name__)
                c = deepcopy(contents)
                for entry in ['needs', 'gives']:
                    for k in c[entry]:
                        c[entry][k.replace(wc, r.__name__)] = c[entry].pop(k)

                yield n, c
            break
    else:
        yield name, contents


for rec in manifest.items():
    for name, contents in _expand_wildcards(*rec):
        #print(name, contents)
        globals()[name] = _parse_machine(name, contents)


class System:

    def __init__(self, *subsystems):
        self.subsystems = subsystems

    def __str__(self):
        items = ["[{}] {}".format(idx, s) for idx, s in enumerate(self)]
        return "\n".join(items)

    def __repr__(self):
        return "<System({})>".format(", ".join(m.label for m in self))

    def needs(self):
        result = {}
        for s in self.subsystems:
            for k, v in s.needs().items():
                result[k] = result.get(k, 0) + v
        return result

    def gives(self):
        result = {}
        for s in self.subsystems:
            for k, v in s.gives().items():
                result[k] = result.get(k, 0) + v
        return result

    def net_output(self, drop=True):
        """
        Return the net output (gives - needs) from the system.

        If drop=True, resources with near-zero net output will be excluded.
        """
        result = self.gives()
        for k, v in self.needs().items():
            result[k] = result.get(k, 0) - v
            if drop and abs(result[k]) < 1e-5:
                result.pop(k)

        return OrderedDict(sorted(result.items(), key=lambda x: -x[1]))

    def __add__(self, other):
        if isinstance(other, Machine):
            return System(*(self.subsystems + (other,)))
        elif isinstance(other, System):
            return System(*(self.subsystems + other.subsystems))


    def reset(self):
        for s in self.subsystems:
            s.uptime = 1

    def __getitem__(self, idx):
        return self.subsystems[idx]

    def __len__(self):
        return len(self.subsystems)

    def _usage_matrix(self, normalize=True):
        result = pd.DataFrame([
            {str(k): v for k, v in s.net_output().items()} for s in self
        ]).fillna(0)

        # Rescale each resource to unity-max
        for c in result.columns:
            result[c] /= result[c].abs().max()

        return result

    def _build_balance_problem(self, neutralize):
        usage = self._usage_matrix()
        A_ub = -1 * usage.T.values
        b_ub = np.zeros(len(A_ub))

        A_eq = usage.T.values
        b_eq = np.zeros(len(A_eq))

        if neutralize != ['all']:
            neutralize = {str(n) for n in neutralize}
            to_neutralize = np.array([c in neutralize for c in usage.columns])
            if to_neutralize.any():
                A_eq = A_eq[to_neutralize]
                b_eq = b_eq[to_neutralize]
            else:
                A_eq = None
                b_eq = None

        return dict(
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=[[0, 1] for _ in range(len(usage))]
        )


    def _balance_cvxpy(self, data):
        import cvxpy as cp

        uptime = cp.Variable(shape=(len(self),), nonneg=True)
        constraints = [
            uptime * data['A_ub'].T <= data['b_ub'],
            uptime >= 0,
            uptime <= 1
        ]

        if data['A_eq'] is not None:
            constraints.append(uptime * data['A_eq'].T == data['b_eq'])

        target = cp.Maximize(cp.sum(uptime))

        problem = cp.Problem(target, constraints)
        kwarg_set = [
            dict(solver='ECOS_BB', mi_abs_eps=1e-7, mi_rel_eps=1e-7, mi_max_iters=100000),
            dict(solver='SCS', eps=1e-7, max_iters=100000),
            dict(solver='ECOS', abstol=1e-7, reltol=1e-7, feastol=1e-7, max_iters=100000),
            dict(solver='OSQP', eps_abs=1e-7, eps_rel=1e-7, max_iter=100000),
        ]

        for kwargs in kwarg_set:
            try:
                problem.solve(**kwargs)
                break
            except cp.SolverError:
                pass
        else:
            raise cp.SolverError("Could not solve balance equation")

        return uptime.value


    def _balance_scipy(self, data):
        from scipy.optimize import linprog

        c = -1 * np.ones(data['A_ub'].shape[1])
        sol = linprog(c, **data)
        assert sol.success
        return sol.x

    def balance(self, neutralize=tuple(), **kwargs):
        self.reset()
        if neutralize == 'all' or neutralize == True:
            neutralize = ['all']

        data = self._build_balance_problem(as_list(neutralize))
        uptime = self._balance_scipy(data)
        #uptime = self._balance_cvxpy(data)

        for m, u in zip(self, uptime):
            m.uptime = u
        return self

    def impute(self):
        self.reset()
        n = self.needs()
        g = self.gives()
        to_impute = []
        for k, v in n.items():
            if k not in g:
                to_impute.append(Source(k, v))
        for k, v in g.items():
            if k not in n:
                to_impute.append(Sink(k, v))

        return System(*(self.subsystems + tuple(to_impute)))


MACHINES.extend([Source, Sink])
