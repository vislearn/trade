import numpy as np
import torch
from torch.utils.data import TensorDataset
import bgflow as bg
from bgmol.datasets import DataSet as bgDataSet
from typing import List

class DataSet(bgDataSet):
    
    @property
    def has_energy(self):
        try:
            self.get_energy_model().energy(torch.zeros(1, self.dim))
            return True
        except NotImplementedError:
            return False
            


class EnergyModel(bg.Energy):
    def energy(self, x, c=None, **kwargs):
        return super().energy(x, **kwargs)

    def derivative(self, x, parameter, wrt="temperature", computed_energy=None, **kwargs):
        if computed_energy is None:
            computed_energy = self._energy(x)

        if wrt == "temperature":
            # u / T -> - u / T^2
            return - computed_energy/parameter.squeeze()
        elif wrt == "inverse_temperature":
            # beta * u -> u
            return computed_energy/parameter.squeeze()
        elif wrt == "coordinates":
            return torch.autograd.grad(computed_energy.sum(), x, create_graph=True)[0]
        else:
            raise(NotImplementedError(f"Unknown parameter {wrt}"))
        

def concatenate_datasets(datasets: List[TensorDataset]) -> TensorDataset:
    tensors = [torch.cat([ds.tensors[i] for ds in datasets]) for i in range(len(datasets[0].tensors))]
    return TensorDataset(*tensors)
    

def split_data(datasets, split=[0.8, 0.1, 0.1], 
               parameter_name="temperature", parameter_reference_value=1.0,
               seed=42):
    train_datasets = []
    val_datasets = []
    test_datasets = []

    for dataset in datasets:    
        n = len(dataset)
        indices = np.arange(n)
        np.random.seed(seed)
        np.random.shuffle(indices)
        indices = torch.from_numpy(indices)
        coordinates = torch.from_numpy(dataset.coordinates.reshape(-1, dataset.dim)).float()

        train_indices = indices[:int(split[0] * n)]
        val_indices = indices[int(split[0] * n):int((split[0] + split[1]) * n)]
        test_indices = indices[int((split[0] + split[1]) * n):]

        dataset_parameter = getattr(dataset, parameter_name)/parameter_reference_value
        
        train_parameters = torch.full((len(train_indices), 1), dataset_parameter).float()
        val_parameters = torch.full((len(val_indices), 1), dataset_parameter).float()
        test_parameters = torch.full((len(test_indices), 1), dataset_parameter).float()

        if hasattr(dataset, "conditions"):
            conditions = torch.from_numpy(dataset.conditions).float()

            train_datasets.append(TensorDataset(coordinates[train_indices], train_parameters, conditions[train_indices]))
            val_datasets.append(TensorDataset(coordinates[val_indices], val_parameters, conditions[val_indices]))
            test_datasets.append(TensorDataset(coordinates[test_indices], test_parameters, conditions[test_indices]))
        else:  
            train_datasets.append(TensorDataset(coordinates[train_indices], train_parameters))
            val_datasets.append(TensorDataset(coordinates[val_indices], val_parameters))
            test_datasets.append(TensorDataset(coordinates[test_indices], test_parameters))

    return concatenate_datasets(train_datasets), concatenate_datasets(val_datasets), concatenate_datasets(test_datasets)