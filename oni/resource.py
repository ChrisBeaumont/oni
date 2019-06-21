__all__ = ['all_resources', 'RESOURCES']

RESOURCES = []


class ResourceMeta(type):

    def __repr__(cls):
        return cls.__name__


def all_resources():
    return sorted(RESOURCES, key=str)


def _get_or_create_resource(name):
    if name not in globals():

        result = ResourceMeta(name, tuple(), {})
        globals()[name] = result
        RESOURCES.append(result)
        __all__.append(name)

    return globals()[name]


LIQUIDS = [
    _get_or_create_resource("LiquidHydrogen"),
    _get_or_create_resource("LiquidChlorine"),
    _get_or_create_resource("LiquidOxygen"),
    _get_or_create_resource("MoltenCarbon"),
    _get_or_create_resource("MoltenIron"),
    _get_or_create_resource("MoltenCopper"),
    _get_or_create_resource("MoltenGold"),
    _get_or_create_resource("LiquidPropane"),
    _get_or_create_resource("Mercury"),
    _get_or_create_resource("Water"),
    _get_or_create_resource("DirtyWater"),
    _get_or_create_resource("Magma"),
    _get_or_create_resource("LiquidCarbonDioxide"),
    _get_or_create_resource("MoltenSteel"),
    _get_or_create_resource("LiquidPhosphorus"),
    _get_or_create_resource("LiquidHelium"),
    _get_or_create_resource("CrudeOil"),
    _get_or_create_resource("MoltenTungsten"),
    _get_or_create_resource("MoltenGlass"),
    _get_or_create_resource("LiquidMethane"),
    _get_or_create_resource("Naphtha"),
    _get_or_create_resource("Petroleum"),
]


GASES = [
    _get_or_create_resource("Hydrogen"),
    _get_or_create_resource("Chlorine"),
    _get_or_create_resource("Oxygen"),
    _get_or_create_resource("ContaminatedOxygen"),
    _get_or_create_resource("CarbonGas"),
    _get_or_create_resource("IronGas"),
    _get_or_create_resource("CopperGas"),
    _get_or_create_resource("GoldGas"),
    _get_or_create_resource("Propane"),
    _get_or_create_resource("MercuryGas"),
    _get_or_create_resource("NaturalGas"),
    _get_or_create_resource("PhosphorusGas"),
    _get_or_create_resource("Helium"),
    _get_or_create_resource("TungstenGas"),
    _get_or_create_resource("Methane"),
    _get_or_create_resource("Steam"),
    _get_or_create_resource("RockGas"),
    _get_or_create_resource("SteelGas"),
    _get_or_create_resource("CarbonDioxide"),
]