from random import random

from hypothesis import given
from hypothesis.strategies import (
    lists, one_of, builds, floats, sampled_from, tuples, just, integers
)

from numpy.testing import assert_almost_equal

from oni import (
    PipedWater, Power, Hydrogen, Electrolyzer,
    Oxygen, Heat, Source, Sink, CeilingLight, System,
    MultiMachine,
    RESOURCES,
    MACHINES
)


class TestMachine:

    def test_basic(self):

        e = Electrolyzer()
        assert e.needs()[PipedWater] == 1
        assert e.needs()[Power] == 120
        assert e.gives()[Hydrogen] == 0.112
        assert e.gives()[Oxygen] == 0.888
        assert e.gives()[Heat] == 45288

    def test_throttle(self):
        e = Electrolyzer().throttle(0.5)
        assert e.uptime == 0.5
        assert e.needs() == {PipedWater: 0.5, Power: 60}
        assert e.gives() == {Oxygen: .444, Hydrogen: .056, Heat: 22644}

    def test_name(self):
        e = Electrolyzer()
        assert str(e) == "Electrolyzer [100.0% Uptime]"

    def test_source_name(self):
        e = Source(Power, 10)
        assert str(e) == "Source(Power, 10) [100.0% Uptime]"

    def test_sink_name(self):
        e = Sink(Power, 10)
        assert str(e) == "Sink(Power, 10) [100.0% Uptime]"

    def test_liquid_expansion(self):
        from oni import WaterPump, PipedLiquidChlorine


class TestSystem:

    def test_addition(self):
        machines = Electrolyzer(), Source(PipedWater, 1)

        s = machines[0] + machines[1]
        assert s.subsystems == machines

    def test_machine_system_addition(self):
        machines = Electrolyzer(), Source(PipedWater, 1), Source(Power, 120)

        s = machines[0] + (machines[1] + machines[2])
        assert s.subsystems == machines

    def test_system_machine_addition(self):
        machines = Electrolyzer(), Source(PipedWater, 1), Source(Power, 120)

        s = (machines[0] + machines[1]) + machines[2]
        assert s.subsystems == machines

    def test_mul(self):

        machine = Source(PipedWater, 1)
        s = machine * 3

        assert len(s.subsystems) == 1
        assert s.gives()[PipedWater] == 3

    def test_rmul(self):

        machine = Source(PipedWater, 1)
        s = 3 * machine

        assert len(s.subsystems) == 1
        assert s.gives()[PipedWater] == 3

    def test_index(self):
        s = Electrolyzer() + Source(PipedWater, 1)
        assert isinstance(s[0], Electrolyzer)
        assert isinstance(s[1], Source)

    def test_impute_sink(self):
        s = System(Source(PipedWater, 5)).impute()
        assert len(s) == 2
        assert isinstance(s[1], Sink)
        assert s[1].resource == PipedWater
        assert s[1].qty == 5

    def test_impute_source(self):
        s = System(Sink(PipedWater, 5)).impute()
        assert len(s) == 2
        assert isinstance(s[1], Source)
        assert s[1].resource == PipedWater
        assert s[1].qty == 5


class TestBalance:

    def test_overlapping_gives(self):
        """Multiple subsystems with overlapping gives combine"""
        s = Source(PipedWater, 1) * 2
        assert s.gives()[PipedWater] == 2

    def test_overlapping_needs(self):
        """Multiple subsystems with overlapping needs combine"""
        s = Sink(PipedWater, 1) * 2
        assert s.needs()[PipedWater] == 2

    def test_full_throughput_balance(self):

        s = Source(PipedWater, 1) + Sink(PipedWater, 1)
        s.balance()
        a, b = s.subsystems
        assert_almost_equal(a.uptime, 1)
        assert_almost_equal(b.uptime, 1)

    def test_throttled_consumer(self):

        s = Source(PipedWater, 0.5) + Sink(PipedWater, 1.0)
        s.balance()
        a, b = s.subsystems
        assert_almost_equal(a.uptime, 1)
        assert_almost_equal(b.uptime, 0.5)

    def test_neutralize_all(self):
        s = Source(PipedWater, 2) + Source(Oxygen, 2)
        s.balance(neutralize='all')
        assert_almost_equal(s[0].uptime, 0)
        assert_almost_equal(s[1].uptime, 0)

    def test_missing_consumer(self):

        # Gives but doesn't need heat
        s = Source(Power, 10) + CeilingLight()
        s.balance()

        a, b = s.subsystems
        assert_almost_equal(a.uptime, 1)
        assert_almost_equal(b.uptime, 1)

    def test_balance_idempotence(self):

        s = Source(PipedWater, 0.5) + Sink(PipedWater, 1.0)
        s.balance()
        s.balance()

        a, b = s.subsystems
        assert_almost_equal(a.uptime, 1)
        assert_almost_equal(b.uptime, 0.5)

    def test_neutralize(self):

        s = System(Source(PipedWater, 0.5))
        s.balance(neutralize=PipedWater)
        assert_almost_equal(s[0].uptime, 0)

    def test_heavy_producer(self):
        s = Source(Power, 1e3) + Sink(Power, 1)
        s.balance()
        assert_almost_equal(s[0].uptime, 1, 6)
        assert_almost_equal(s[1].uptime, 1, 6)

    def test_heavy_consumer(self):
        s = Source(Power, 1) + Sink(Power, 1e3)
        s.balance()
        print(s)
        assert_almost_equal(s[0].uptime, 1, 6)
        assert_almost_equal(s[1].uptime, 1e-3, 6)


def make_system(machines):
    return System(*[m[0](*m[1:]) for m in machines])


def instantiate(args):
    return args[0](*args[1:])

basic_machines = tuples(sampled_from([m for m in MACHINES if m not in (Source, Sink)]))
source_sink = tuples(sampled_from([Source, Sink]), sampled_from(RESOURCES), floats(min_value=1e-5, max_value=10, allow_nan=False, allow_infinity=False))
multi_machines = tuples(just(MultiMachine), builds(instantiate, basic_machines), integers(min_value=1, max_value=100))
systems = builds(make_system, lists(one_of(basic_machines, multi_machines, source_sink), min_size=2, max_size=50))


class TestFuzzBalance:

    def validate_balance(self, system):
        print(system)

        for m in system:
            assert m.uptime >= -1e-5
            assert m.uptime <= 1 + 1e-5

        for resource, qty in system.net_output().items():
            assert qty >= -1e-5

    @given(systems)
    def test_balance_fully_neutralize(self, system):
        system.balance(neutralize='all')
        self.validate_balance(system)

    @given(systems)
    def test_balance_partially_neutralize(self, system):
        system = system.impute()
        subset = {r for r in system.needs().keys() if random() > 0.5}
        system.balance(neutralize=subset)
        self.validate_balance(system)

    @given(systems)
    def test_fuzz_system(self, system):
        system.balance()
        self.validate_balance(system)

    @given(systems)
    def test_balance_impute(self, system):
        system = system.impute()
        system.balance()
        assert any(m.uptime > 1e-3 for m in system)
        self.validate_balance(system)
