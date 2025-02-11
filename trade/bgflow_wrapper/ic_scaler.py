import torch
import bgflow as bg
from bgflow import TORSIONS, BONDS, ANGLES
import numpy as np


class ICScaler(bg.Flow):
    def __init__(
        self,
        initial_structure: torch.Tensor,
        ic_trafo: bg.Flow,
        stds: dict = {BONDS: 0.005, ANGLES: 0.15 / np.pi, TORSIONS: 1.0},
        prior_std: float = 0.1,
        scale_torsions: bool = False,
        constrained_bond_indices: list = None,
    ):
        super().__init__()

        assert initial_structure.shape[0] == 1

        self.ic_trafo = ic_trafo
        self.scale_torsions = scale_torsions

        means = self._compute_means(initial_structure)

        if constrained_bond_indices is not None:
            unconstrained_bond_indices = np.array(
                [
                    i
                    for i in range(len(means[BONDS]))
                    if i not in constrained_bond_indices
                ]
            )
            means[BONDS] = means[BONDS][unconstrained_bond_indices]

        for key, value in means.items():
            self.register_buffer(f"mean_{key.name}", value)
        for key, value in stds.items():
            if isinstance(value, (int, float)):
                value = torch.tensor(value)
            self.register_buffer(f"std_{key.name}", value)

        self.dim_ic = len(means[BONDS]) + len(means[ANGLES]) + len(means[TORSIONS])
        self.dim_bonds = len(means[BONDS])
        self.dim_angles = len(means[ANGLES])
        self.dim_torsions = len(means[TORSIONS])

        scale_jac = torch.log(self.stds[BONDS] / prior_std)*self.dim_bonds + torch.log(self.stds[ANGLES] / prior_std)*self.dim_angles

        if self.scale_torsions:
            scale_jac += torch.log(self.stds[TORSIONS] / prior_std)*self.dim_torsions

        self.register_buffer("scale_jac", scale_jac)
        self.register_buffer("prior_std", torch.tensor(prior_std))

    @property
    def means(self):
        return {
            BONDS: self.mean_BONDS,
            ANGLES: self.mean_ANGLES,
            TORSIONS: self.mean_TORSIONS,
        }

    @property
    def stds(self):
        return {
            BONDS: self.std_BONDS,
            ANGLES: self.std_ANGLES,
            TORSIONS: self.std_TORSIONS,
        }

    def _compute_means(
        self,
        initial_structure: torch.Tensor,
    ):
        bonds, angles, torsions = self.ic_trafo.forward(initial_structure)[0:3]
        bonds = bonds[0, :].detach()
        angles = angles[0, :].detach()
        torsions = torsions[0, :].detach()

        return {BONDS: bonds, ANGLES: angles, TORSIONS: torsions}

    def _forward(
        self, bonds, angles, torsions, *args, **kwargs
    ):  # Going from [0,1] to IC
        bonds = (bonds - 0.5) * self.stds[BONDS] / self.prior_std + self.means[BONDS]
        angles = (angles - 0.5) * self.stds[ANGLES] / self.prior_std + self.means[
            ANGLES
        ]
        if self.scale_torsions:
            torsions = (torsions - 0.5) * self.stds[
                TORSIONS
            ] / self.prior_std + self.means[TORSIONS]

        return bonds, angles, torsions, self.scale_jac.expand(bonds.shape[0], 1)
    def _inverse(
        self, bonds, angles, torsions, *args, **kwargs
    ):  # Going from IC to [0,1]
        bonds = (bonds - self.means[BONDS]) / self.stds[BONDS] * self.prior_std + 0.5
        angles = (angles - self.means[ANGLES]) / self.stds[
            ANGLES
        ] * self.prior_std + 0.5
        if self.scale_torsions:
            torsions = (torsions - self.means[TORSIONS]) / self.stds[
                TORSIONS
            ] * self.prior_std + 0.5

        return bonds, angles, torsions, -self.scale_jac.expand(bonds.shape[0], 1)
