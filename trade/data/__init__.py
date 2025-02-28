import os
from functools import partial
from .ala2 import ALA2BYTEMPERATURE
from .double_well import DoubleWellDataset
from .multi_well import MultiWellDataset
from .gmm import GMMDataset
from .two_moons import TwoMoonsDataset
from .base import split_data

def get_loader(name):
    if name.lower() == "ala2":
        return get_ala2
    elif name.lower().startswith("double_well"):
        dim = name.split("_")[-1]
        if not dim.isnumeric():
            dim = dim[:-1]
        assert dim.isnumeric(), "Specify the dimension of the double_well dataset \
        by passing either double_well_{dim} or double_well_{dim}d as dataset name"
        return partial(get_double_well, dim=int(dim))
    elif name.lower().startswith("multi_well"):
        dim = name.split("_")[-1]
        if not dim.isnumeric():
            dim = dim[:-1]
        assert dim.isnumeric(), "Specify the dimension of the multi_well dataset \
        by passing either multi_well_{dim} or multi_well_{dim}d as dataset name"
        return partial(get_multi_well, dim=int(dim))
    elif name.lower().startswith("gmm"):
        dim = name.split("_")[-1]
        if not dim.isnumeric():
            dim = dim[:-1]
        assert dim.isnumeric(), "Specify the dimension of the GMM dataset \
        by passing either gmm_{dim} or gmm_{dim}d as dataset name"
        return partial(get_GMM, dim=int(dim))
    elif name.lower().startswith("two_moons"):
        return partial(get_two_moons, dim=2)
    else:
        raise ValueError(f"Unknown dataset {name}")

def get_ala2(temperature=600, **kwargs):
    temperature = float(temperature)
    if temperature not in ALA2BYTEMPERATURE:
        raise ValueError(f"Unknown temperature {temperature}")
    dataset_class, filename = ALA2BYTEMPERATURE[temperature]
    is_data_here = os.path.isfile(filename)
    dataset = dataset_class(download=(not is_data_here), read=True if "read" not in kwargs else kwargs["read"])
    return dataset

def get_double_well(dim=2, **kwargs):
    return DoubleWellDataset(dim=dim, **kwargs)

def get_multi_well(dim=2, **kwargs):
    return MultiWellDataset(dim=dim, **kwargs)

def get_GMM(dim=2, **kwargs):
    return GMMDataset(dim=dim, **kwargs)

def get_two_moons(dim=2, **kwargs):
    return TwoMoonsDataset(dim=dim, **kwargs)