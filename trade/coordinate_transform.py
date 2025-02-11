import torch
import FrEIA.modules as Fm
from bgflow import MixedCoordinateTransformation, SplitFlow, InverseFlow, MergeFlow, GlobalInternalCoordinateTransformation
from bgflow.nn.flow.base import Flow
import numpy as np
from .bgflow_wrapper.ic_scaler import ICScaler
from bgmol import bond_constraints
import bgmol

class ConstrainedCoordinateTransform(Fm.InvertibleModule):
    def __init__(self, dims_in, dims_c=None, system=None, coordinates=None):
        super().__init__(dims_in, dims_c)   

        z_matrix = bgmol.systems.ala2.DEFAULT_GLOBAL_Z_MATRIX

        coordinate_trafo = GlobalInternalCoordinateTransformation(
            z_matrix=z_matrix,
            enforce_boundaries=True,
            normalize_angles=True,
        )

        constraints = bond_constraints(system.system, coordinate_trafo)

        ##### IC scaling #####
        min_energy_structure = torch.load("trade/bgflow_wrapper/data_ala2/position_min_energy.pt").to(
            dtype=torch.get_default_dtype()
        )
        ic_scaler = ICScaler(
            min_energy_structure,
            ic_trafo=coordinate_trafo,
            constrained_bond_indices=constraints[0],
        )
        self.ic_scaler = ic_scaler

        dim_bonds, dim_angles, dim_torsions = ic_scaler.dim_bonds, ic_scaler.dim_angles, ic_scaler.dim_torsions

        self.split_layer = SplitFlow(dim_bonds, dim_angles, dim_torsions)
        self.coordinate_trafo = coordinate_trafo
        self.constraint_values = torch.from_numpy(constraints[1]).float()
        unconstrained_indices = np.setdiff1d(np.arange(9+12), constraints[0])
        self.merge = MergeFlow(torch.from_numpy(unconstrained_indices), 
                               torch.from_numpy(constraints[0]), dim=-1)
        
        coordinates = torch.from_numpy(coordinates.reshape(coordinates.shape[0], system.dim))
        bonds, angles, torsions, _, _, _ = self.coordinate_trafo._forward(coordinates)
        bonds, _, _ = self.merge._inverse(bonds)
        bonds, angles, torsions, _ = self.ic_scaler._inverse(*(bonds, angles, torsions))
        self.whiten = WhitenAfterIC(bonds, angles, torsions)


    def forward(self, x_or_z, c = None, rev=False, jac=False):
        if rev:
            R = torch.ones(x_or_z[0].shape[0], 3, device=x_or_z[0].device)*0.5
            origin = torch.zeros(x_or_z[0].shape[0], 3, device=x_or_z[0].device)
            bonds, angles, torsions, dlogp_split = self.split_layer._forward(x_or_z[0])
            bonds, angles, torsions, dlogp_whiten = self.whiten._forward(*(bonds, angles, torsions))
            bonds, angles, torsions, dlogp = self.ic_scaler._forward(*(bonds, angles, torsions))
            bonds, dlogp_merge = self.merge._forward(bonds, self.constraint_values.to(bonds.device).unsqueeze(0).repeat(bonds.shape[0], 1))
            out, dlogp_ct = self.coordinate_trafo._inverse(bonds, angles, torsions, origin, R)
            return (out,), (dlogp + dlogp_split + dlogp_ct + dlogp_whiten + dlogp_merge).squeeze()
        else:
            bonds, angles, torsions, _, _, dlogp_ct = self.coordinate_trafo._forward(x_or_z[0])
            bonds, _, dlogp_merge = self.merge._inverse(bonds)
            bonds, angles, torsions, dlogp = self.ic_scaler._inverse(*(bonds, angles, torsions))
            bonds, angles, torsions, dlogp_whiten = self.whiten._inverse(*(bonds, angles, torsions))
            out, dlogp_split = self.split_layer._inverse(*(bonds, angles, torsions))
            return (out,), (dlogp + dlogp_split + dlogp_ct + dlogp_whiten + dlogp_merge).squeeze()

    def output_dims(self, input_dims):
        return [(self.ic_scaler.dim_ic,)]


class LegacyCoordinateTransform(Fm.InvertibleModule):
    def __init__(self, dims_in, dims_c=None, system=None, coordinates=None):
        dim_cartesian, dim_bonds, dim_angles, dim_torsions, dim_ics = self.get_dimensions(system)
        super().__init__(dims_in, dims_c)

        assert self.dims_in[0][0] == system.dim, "Received input dimension does not match system dimension"
        assert len(self.dims_in[0]) == 1, "Cannot perform coordinate transformation on images"
        self.dim_ics = dim_ics
        coordinates = torch.from_numpy(coordinates.reshape(coordinates.shape[0], system.dim))

        self.transform = MixedCoordinateTransformation(
            data=coordinates,
            z_matrix=system.z_matrix,
            fixed_atoms=system.rigid_block,
            keepdims=dim_cartesian,
            normalize_angles=False,
        )
        self.split_layer = SplitFlow(dim_bonds, dim_angles, dim_torsions, dim_cartesian)

    @staticmethod
    def get_dimensions(system):
        # throw away 6 degrees of freedom (rotation and translation)
        dim_cartesian = len(system.rigid_block) * 3 - 6
        dim_bonds = len(system.z_matrix)
        dim_angles = dim_bonds
        dim_torsions = dim_bonds
        dim_ics = dim_bonds + dim_angles + dim_torsions + dim_cartesian
        return dim_cartesian, dim_bonds, dim_angles, dim_torsions, dim_ics


    def forward(self, x_or_z, c = None, rev=False, jac=False):
        if not rev:
            bonds, angles, torsions, z_fixed, dlogp = self.transform._forward(x_or_z[0])
            out, dlogp_split = self.split_layer._inverse(*(bonds, angles, torsions, z_fixed), jac=jac)
            return (out,), (dlogp + dlogp_split).squeeze()
        else:
            bonds, angles, torsions, z_fixed, dlogp_split = self.split_layer._forward(x_or_z[0])
            out, dlogp = self.transform._inverse(*(bonds, angles, torsions, z_fixed))
            return (out,), (dlogp + dlogp_split).squeeze()

    def output_dims(self, input_dims):
        return [(self.dim_ics,)]


class CoordinateTransform(Fm.InvertibleModule):
    # TODO: Credit this code (original author: Leon Klein, source: bgtorch), ask for permission to publish it
    def __init__(self, dims_in, dims_c=None, system=None, coordinates=None, keepdims_rigid=9, keepdims_bonds=6):
        dim_cartesian, dim_bonds, dim_angles, dim_torsions, dim_ics = self.get_dimensions(system, keepdims_rigid, keepdims_bonds)
        super().__init__(dims_in, dims_c)

        assert self.dims_in[0][0] == system.dim, "Received input dimension does not match system dimension"
        assert len(self.dims_in[0]) == 1, "Cannot perform coordinate transformation on images"
        self.dim_ics = dim_ics
        coordinates = torch.from_numpy(coordinates.reshape(coordinates.shape[0], system.dim))


        self.rel_ic = InverseFlow(RelativeInternalCoordinatesTransformation(
            z_matrix=system.z_matrix,
            fixed_atoms=system.rigid_block,
            normalize_angles=False,
        ))
        ct = ConstrainedTransformation(-np.pi, np.pi)
        self.whiten_ic = WhitenIC(self.rel_ic, ct, coordinates, keepdims_rigid=keepdims_rigid, keepdims_bonds=keepdims_bonds)
        self.split_layer = SplitFlow(dim_bonds, dim_angles, dim_torsions, dim_cartesian)

    @staticmethod
    def get_dimensions(system, keepdims_rigid, keepdims_bonds):
        # throw away 6 degrees of freedom (rotation and translation)
        dim_cartesian = min(len(system.rigid_block) * 3 - 6, keepdims_rigid)
        dim_bonds = min(len(system.z_matrix), keepdims_bonds)
        dim_angles = dim_torsions = len(system.z_matrix)
        dim_ics = dim_bonds + dim_angles + dim_torsions + dim_cartesian
        print(dim_cartesian, dim_bonds, dim_angles, dim_torsions, dim_ics)
        return dim_cartesian, dim_bonds, dim_angles, dim_torsions, dim_ics


    def forward(self, x_or_z, c = None, rev=False, jac=False):
        if not rev:
            bonds, angles, torsions, z_fixed, dlogp1 = self.rel_ic._inverse(x_or_z[0])
            bonds, angles, torsions, z_fixed, dlogp2 = self.whiten_ic._inverse(bonds, angles, torsions, z_fixed)
            out, dlogp3 = self.split_layer._inverse(*(bonds, angles, torsions, z_fixed))
            return (out,), (dlogp1 + dlogp2 + dlogp3).squeeze()
        else:
            bonds, angles, torsions, z_fixed, dlogp1 = self.split_layer._forward(x_or_z[0])
            bonds, angles, torsions, z_fixed, dlogp2 = self.whiten_ic._forward(*(bonds, angles, torsions, z_fixed))
            out, dlogp3 = self.rel_ic._forward(bonds, angles, torsions, z_fixed)
            return (out,), (dlogp1 + dlogp2 + dlogp3).squeeze()

    def output_dims(self, input_dims):
        return [(self.dim_ics,)]

class ConstrainedTransformation(Flow):
    def __init__(self, a=0.0, b=1.0):
        super().__init__()
        self.a = a
        self.b = b

    def _inverse(self, x):
        z = torch.log(x - self.a) - torch.log(self.b - x)
        dlogp = (
            np.log(self.b - self.a) * x.shape[1]
            - torch.log(self.b - x).sum(dim=1, keepdim=True)
            - torch.log(x - self.a).sum(dim=1, keepdim=True)
        )
        return z, dlogp

    def _forward(self, x):
        z = (self.b * torch.exp(x) + self.a) / (torch.exp(x) + 1)
        dlogp = (
            np.log(self.b - self.a) * x.shape[1]
            + x.sum(dim=1, keepdim=True)
            - 2 * torch.log(torch.exp(x) + 1).sum(dim=1, keepdim=True)
        )
        return z, dlogp

def circular_mean(alpha):
    return torch.atan2(alpha.sin().mean(dim=0), alpha.cos().mean(dim=0)).view(1, -1)


def shift_angles(alpha, shift):
    # shifts and wraps angles to interval [-pi, pi)
    alpha_shifted = alpha - shift + np.pi
    return alpha_shifted % (2 * np.pi) - np.pi


def ct_shift_angle(angles, ct):
    shifts = []
    shift_proposals = np.linspace(0, 2 * np.pi, 100)
    for i in range(angles.shape[1]):
        dist = []
        for shift in shift_proposals:
            t = shift_angles(angles[:, i], shift).view(-1, 1)
            transf, _ = ct.forward(t, inverse=True)
            dist.append((transf ** 2).mean())

        shifts.append(shift_proposals[np.argmin(dist)])
    return torch.tensor(shifts).view(1, -1).float()

def _pca(X0, keepdims=None):
    """Implements PCA in Numpy.

    This is not written for training in torch because derivatives of eig are not implemented

    """
    if keepdims is None:
        keepdims = X0.shape[1]

    # pca
    X0mean = X0.mean(axis=0)
    X0meanfree = X0 - X0mean
    C = np.matmul(X0meanfree.T, X0meanfree) / (X0meanfree.shape[0] - 1.0)
    eigval, eigvec = np.linalg.eigh(C)

    # sort in descending order and keep only the wanted eigenpairs
    I = np.argsort(eigval)[::-1]
    I = I[:keepdims]
    eigval = eigval[I]
    std = np.sqrt(eigval)
    eigvec = eigvec[:, I]

    # whiten and unwhiten matrices
    Twhiten = np.matmul(eigvec, np.diag(1.0 / std))
    Tblacken = np.matmul(np.diag(std), eigvec.T)
    return X0mean, Twhiten, Tblacken, std


class WhitenFlow(Flow):
    def __init__(self, X0, keepdims=None, whiten_inverse=True):
        """Performs static whitening of the data given PCA of X0

        Parameters:
        -----------
        X0 : array
            Initial Data on which PCA will be computed.
        keepdims : int or None
            Number of dimensions to keep. By default, all dimensions will be kept
        whiten_inverse : bool
            Whitens when calling inverse (default). Otherwise when calling forward

        """
        super().__init__()
        if keepdims is None:
            keepdims = X0.shape[1]
        self.dim = X0.shape[1]
        self.keepdims = keepdims
        self.whiten_inverse = whiten_inverse

        dtype = X0.dtype

        X0_np = X0.detach().numpy().astype(np.float64)
        X0mean, Twhiten, Tblacken, std = _pca(X0_np, keepdims=keepdims)
        self.register_buffer("X0mean", torch.tensor(X0mean, dtype=dtype))
        self.register_buffer("Twhiten", torch.tensor(Twhiten, dtype=dtype))
        self.register_buffer("Tblacken", torch.tensor(Tblacken, dtype=dtype))
        self.register_buffer("std", torch.tensor(std, dtype=dtype))
        if torch.any(self.std <= 0):
            raise ValueError(
                "Cannot construct whiten layer because trying to keep nonpositive eigenvalues."
            )
        self.jacobian_xz = -torch.sum(torch.log(self.std))

    def _whiten(self, x):
        output_z = torch.matmul(x - self.X0mean, self.Twhiten)
        dlogp = self.jacobian_xz * torch.ones((x.shape[0], 1), device=x.device)
        return output_z, dlogp

    def _blacken(self, x):
        output_x = torch.matmul(x, self.Tblacken) + self.X0mean
        dlogp = -self.jacobian_xz * torch.ones((x.shape[0], 1), device=x.device)
        return output_x, dlogp

    def _forward(self, x, *args, **kwargs):
        if self.whiten_inverse:
            y, dlogp = self._blacken(x)
        else:
            y, dlogp = self._whiten(x)
        return y, dlogp

    def _inverse(self, x, *args, **kwargs):
        if self.whiten_inverse:
            y, dlogp = self._whiten(x)
        else:
            y, dlogp = self._blacken(x)
        return y, dlogp

class WhitenIC(Flow):
    def __init__(
        self, rel_ic, ct, X0, keepdims_rigid, keepdims_bonds, angle_loss_strength=5.0
    ):
        super().__init__()
        bonds_0, angles_0, torsions_0, z_0, _ = rel_ic.forward(X0, inverse=True)

        self._whitenflow_bonds = WhitenFlow(bonds_0, keepdims=keepdims_bonds)

        self._angle_means = circular_mean(angles_0)
        angles_0 = shift_angles(angles_0, self._angle_means)
        self._whitenflow_angles = WhitenFlow(angles_0)

        self._torsion_shifts = ct_shift_angle(torsions_0, ct)
        torsions_0 = shift_angles(torsions_0, self._torsion_shifts)
        self._whitenflow_torsions = WhitenFlow(torsions_0)

        self._whitenflow_rigid = WhitenFlow(z_0, keepdims=keepdims_rigid)
        self._angle_loss_strength = angle_loss_strength

    def _shift_torsions(self, torsions, inverse=False):
        if inverse:
            return torch.where(
                torsions < self._torsioncuts, torsions + 2.0 * np.pi, torsions
            )
        return torch.where(torsions > 2.0 * np.pi, torsions - 2.0 * np.pi, torsions,)

    def _forward(self, bonds, angles, torsions, z_fixed, **kwargs):
        bonds, dlogp_bonds = self._whitenflow_bonds.forward(bonds, inverse=False)

        angles, dlogp_angles = self._whitenflow_angles.forward(angles, inverse=False)
        angles = shift_angles(angles, -self._angle_means.to(angles.device))

        torsions, dlogp_torsions = self._whitenflow_torsions(torsions, inverse=False)

        torsions = shift_angles(torsions, -self._torsion_shifts.to(torsions.device))

        z_fixed, dlogp_rigid = self._whitenflow_rigid.forward(z_fixed, inverse=False)

        return (
            bonds,
            angles,
            torsions,
            z_fixed,
            dlogp_bonds + dlogp_rigid
            + dlogp_angles
            + dlogp_torsions,
        )

    def _inverse(self, bonds, angles, torsions, z_fixed, **kwargs):
        bonds, dlogp_bonds = self._whitenflow_bonds.forward(bonds, inverse=True)

        angles = shift_angles(angles, self._angle_means.to(angles.device))
        angles, dlogp_angles = self._whitenflow_angles.forward(angles, inverse=True)

        torsions = shift_angles(torsions, self._torsion_shifts.to(torsions.device))
        torsions, dlogp_torsions = self._whitenflow_torsions(torsions, inverse=True)

        z_fixed, dlogp_rigid = self._whitenflow_rigid.forward(z_fixed, inverse=True)
        return (
            bonds,
            angles,
            torsions,
            z_fixed,
            dlogp_bonds + dlogp_rigid
            + dlogp_angles + dlogp_torsions
        )

class WhitenAfterIC(Flow):
    def __init__(
        self, bonds_0, angles_0, torsions_0
    ):
        super().__init__()
        self._whitenflow_bonds = WhitenFlow(bonds_0.detach())
        self._angle_means = circular_mean(angles_0).detach()
        angles_0 = shift_angles(angles_0, self._angle_means)
        self._whitenflow_angles = WhitenFlow(angles_0.detach())
        self._torsion_shifts = circular_mean(torsions_0).detach()
        torsions_0 = shift_angles(torsions_0, self._torsion_shifts)
        self._whitenflow_torsions = WhitenFlow(torsions_0.detach())

    def _shift_torsions(self, torsions, inverse=False):
        if inverse:
            return torch.where(
                torsions < self._torsioncuts, torsions + 2.0 * np.pi, torsions
            )
        return torch.where(torsions > 2.0 * np.pi, torsions - 2.0 * np.pi, torsions,)
    
    def _forward(self, bonds, angles, torsions, **kwargs):
        bonds, dlogp_bonds = self._whitenflow_bonds.forward(bonds, inverse=False)
        angles, dlogp_angles = self._whitenflow_angles.forward(angles, inverse=False)
        angles = shift_angles(angles, -self._angle_means.to(angles.device))
        torsions, dlogp_torsions = self._whitenflow_torsions(torsions, inverse=False)
        torsions = shift_angles(torsions, -self._torsion_shifts.to(torsions.device))
        return bonds, angles, torsions, dlogp_bonds + dlogp_angles + dlogp_torsions
    
    def _inverse(self, bonds, angles, torsions, **kwargs):
        bonds, dlogp_bonds = self._whitenflow_bonds.forward(bonds, inverse=True)
        angles = shift_angles(angles, self._angle_means.to(angles.device))
        angles, dlogp_angles = self._whitenflow_angles.forward(angles, inverse=True)
        torsions = shift_angles(torsions, self._torsion_shifts.to(torsions.device))
        torsions, dlogp_torsions = self._whitenflow_torsions(torsions, inverse=True)
        return bonds, angles, torsions, dlogp_bonds + dlogp_angles + dlogp_torsions
    

def ic2xyz_deriv(p1, p2, p3, d14, a124, t1234):
    """ computes the xyz coordinates from internal coordinates
        relative to points `p1`, `p2`, `p3` together with its
        jacobian with respect to `p1`.
    """

    v1 = p1 - p2
    v2 = p1 - p3

    n = torch.cross(v1, v2, dim=-1)
    nn = torch.cross(v1, n, dim=-1)

    n_normalized = n / torch.norm(n, dim=-1, keepdim=True)
    nn_normalized = nn / torch.norm(nn, dim=-1, keepdim=True)

    n_scaled = n_normalized * -torch.sin(t1234)
    nn_scaled = nn_normalized * torch.cos(t1234)

    v3 = n_scaled + nn_scaled
    v3_norm = torch.norm(v3, dim=-1, keepdim=True)
    v3_normalized = v3 / v3_norm
    v3_scaled = v3_normalized * d14 * torch.sin(a124)

    v1_norm = torch.norm(v1, dim=-1, keepdim=True)
    v1_normalized = v1 / v1_norm
    v1_scaled = v1_normalized * d14 * torch.cos(a124)

    position = p1 + v3_scaled - v1_scaled

    J_d = v3_normalized * torch.sin(a124) - v1_normalized * torch.cos(a124)
    J_a = v3_normalized * d14 * torch.cos(a124) + v1_normalized * d14 * torch.sin(a124)

    J_t1 = (d14 * torch.sin(a124))[..., None]
    J_t2 = (
        1.0
        / v3_norm[..., None]
        * (torch.eye(3, device=v3_normalized.device)[None, :] - outer(v3_normalized, v3_normalized))
    )

    J_n_scaled = n_normalized * -torch.cos(t1234)
    J_nn_scaled = nn_normalized * -torch.sin(t1234)
    J_t3 = (J_n_scaled + J_nn_scaled)[..., None]

    J_t = (J_t1 * J_t2) @ J_t3

    J = torch.stack([J_d, J_a, J_t[..., 0]], dim=-1)

    return position, J

def decompose_z_matrix(z_matrix, fixed):
    atoms = [fixed]

    blocks = []
    given = np.sort(fixed)

    # filter out conditioned variables
    non_given = ~np.isin(z_matrix[:, 0], given)
    z_matrix = z_matrix[non_given]
    z_matrix = np.concatenate([np.arange(len(z_matrix))[:, None], z_matrix], axis=1)

    order = []
    while len(z_matrix) > 0:

        is_satisfied = np.all(np.isin(z_matrix[:, 2:], given), axis=-1)
        if (not np.any(is_satisfied)) and len(z_matrix) > 0:
            raise ValueError("Not satisfiable")

        pos = z_matrix[is_satisfied, 0]
        atom = z_matrix[is_satisfied, 1]

        atoms.append(atom)
        order.append(pos)

        blocks.append(z_matrix[is_satisfied][:, 1:])
        given = np.union1d(given, atom)
        z_matrix = z_matrix[~is_satisfied]

    index2atom = np.concatenate(atoms)
    atom2index = np.argsort(index2atom)
    index2order = np.concatenate(order)
    return blocks, index2atom, atom2index, index2order

def normalize_angles(angles):
    dlogp = -np.log(2 * np.pi) * (angles.shape[-1])
    return (angles + np.pi) / (2 * np.pi), dlogp


def unnormalize_angles(angles):
    dlogp = np.log(2 * np.pi) * (angles.shape[-1])
    return angles * (2 * np.pi) - np.pi, dlogp

def outer(x, y):
    """ outer product between input tensors """
    return x[..., None] @ y[..., None, :]


def skew(x):
    """
        returns skew symmetric 3x3 form of a 3 dim vector
    """
    assert len(x.shape) > 1, "`x` requires at least 2 dimensions"
    zero = torch.zeros(*x.shape[:-1]).to(x)
    a, b, c = x[..., 0], x[..., 1], x[..., 2]
    s = torch.stack(
        [
            torch.stack([zero, c, -b], dim=-1),
            torch.stack([-c, zero, a], dim=-1),
            torch.stack([b, -a, zero], dim=-1),
        ],
        dim=-1,
    )
    return s

def det3x3(a):
    """ batch determinant of a 3x3 matrix """
    return (torch.cross(a[..., 0, :], a[..., 1, :], dim=-1) * a[..., 2, :]).sum(dim=-1)


def dist_deriv(x1, x2):
    """
        computes distance between input points together with
        the Jacobian wrt to `x1`
    """
    r = x2 - x1
    rnorm = torch.norm(r, dim=-1, keepdim=True)
    dist = rnorm[..., 0]
    J = -r / rnorm
    # J = _safe_div(-r, rnorm)
    return dist, J


def angle_deriv(x1, x2, x3):
    """
        computes angle between input points together with
        the Jacobian wrt to `x1`
    """
    r12 = x1 - x2
    r12_norm = torch.norm(r12, dim=-1, keepdim=True)
    rn12 = r12 / r12_norm
    J = (torch.eye(3).to(x1) - outer(rn12, rn12)) / r12_norm[..., None]

    r32 = x3 - x2
    r32_norm = torch.norm(r32, dim=-1, keepdim=True)
    rn32 = r32 / r32_norm

    cos_angle = torch.sum(rn12 * rn32, dim=-1)
    J = rn32[..., None, :] @ J

    a = torch.acos(cos_angle)
    J = -J / torch.sqrt(1.0 - cos_angle.pow(2)[..., None, None])

    return a, J[..., 0, :]


def torsion_deriv(x1, x2, x3, x4):
    """
        computes torsion angle between input points together with
        the Jacobian wrt to `x1`.
    """
    b0 = -1.0 * (x2 - x1)


    b1 = x3 - x2
    b2 = x4 - x3
    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1norm = torch.norm(b1, dim=-1, keepdim=True)
    b1_normalized = b1 / b1norm

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    #
    # dv_db0 = jacobian of v wrt b0
    v = b0 - torch.sum(b0 * b1_normalized, dim=-1, keepdim=True) * b1_normalized
    dv_db0 = torch.eye(3)[None, None, :, :].to(x1) - outer(b1_normalized, b1_normalized)

    w = b2 - torch.sum(b2 * b1_normalized, dim=-1, keepdim=True) * b1_normalized

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    #
    # dx_dv = jacobian of x wrt v
    x = torch.sum(v * w, dim=-1, keepdim=True)
    dx_dv = w[..., None, :]

    # b1xv = fast cross product between b1_normalized and v
    # given by multiplying v with the skew of b1_normalized
    # (see https://en.wikipedia.org/wiki/Skew-symmetric_matrix#Cross_product)
    #
    # db1xv_dv = Jacobian of b1xv wrt v
    A = skew(b1_normalized)
    b1xv = (A @ (v[..., None]))[..., 0]
    db1xv_dv = A

    # y = dot product of b1xv and w
    # dy_db1xv = Jacobian of v wrt b1xv
    y = torch.sum(b1xv * w, dim=-1, keepdim=True)
    dy_db1xv = w[..., None, :]

    x = x[..., None]
    y = y[..., None]

    # a = torsion angle spanned by unit vector (x, y)
    # xysq = squared norm of (x, y)
    # da_dx = Jacobian of a wrt xysq
    a = torch.atan2(y, x)
    xysq = x.pow(2) + y.pow(2)
    da_dx = -y / xysq
    da_dy = x / xysq

    # compute derivative with chain rule
    J = da_dx @ dx_dv @ dv_db0 + da_dy @ dy_db1xv @ db1xv_dv @ dv_db0

    return a[..., 0, 0], J[..., 0, :]


class RelativeInternalCoordinatesTransformation(Flow):
    """ global internal coordinate transformation:
        transforms a system from xyz to ic coordinates and back.
    """

    def __init__(self, z_matrix, fixed_atoms, normalize_angles=True):
        super().__init__()

        self._z_matrix = z_matrix

        self._fixed_atoms = fixed_atoms

        (
            self._z_blocks,
            self._index2atom,
            self._atom2index,
            self._index2order,
        ) = decompose_z_matrix(z_matrix, fixed_atoms)

        self._bond_indices = self._z_matrix[:, :2]
        self._angle_indices = self._z_matrix[:, :3]
        self._torsion_indices = self._z_matrix[:, :4]

        self._normalize_angles = normalize_angles

    def _forward(self, x, with_pose=True, *args, **kwargs):
        """ Computes xyz -> ic

            Returns bonds, angles, torsions and fixed coordinates.
        """

        n_batch = x.shape[0]
        x = x.view(n_batch, -1, 3)

        # compute bonds, angles, torsions
        # together with jacobians (wrt. to diagonal atom)
        bonds, jbonds = dist_deriv(
            x[:, self._z_matrix[:, 0]], x[:, self._z_matrix[:, 1]]
        )
        angles, jangles = angle_deriv(
            x[:, self._z_matrix[:, 0]],
            x[:, self._z_matrix[:, 1]],
            x[:, self._z_matrix[:, 2]],
        )
        torsions, jtorsions = torsion_deriv(
            x[:, self._z_matrix[:, 0]],
            x[:, self._z_matrix[:, 1]],
            x[:, self._z_matrix[:, 2]],
            x[:, self._z_matrix[:, 3]],
        )

        # slice fixed coordinates needed to reconstruct the system
        x_fixed = x[:, self._fixed_atoms].view(n_batch, -1)

        # aggregated induced volume change
        dlogp = 0.0

        # transforms angles from [-pi, pi] to [0, 1]
        if self._normalize_angles:
            angles, dlogp_a = normalize_angles(angles)
            torsions, dlogp_t = normalize_angles(torsions)
            dlogp += dlogp_a + dlogp_t

        # compute volume change
        j = torch.stack([jbonds, jangles, jtorsions], dim=-2)
        dlogp += det3x3(j).abs().log().sum(dim=1, keepdim=True)

        return bonds, angles, torsions, x_fixed, dlogp

    def _inverse(self, bonds, angles, torsions, x_fixed, **kwargs):

        # aggregated induced volume change
        dlogp = 0

        # transforms angles from [0, 1] to [-pi, pi]
        if self._normalize_angles:
            angles, dlogp_a = unnormalize_angles(angles)
            torsions, dlogp_t = unnormalize_angles(torsions)
            dlogp += dlogp_a + dlogp_t

        n_batch = x_fixed.shape[0]

        # initial points are the fixed points
        ps = x_fixed.view(n_batch, -1, 3).clone()

        # blockwise reconstruction of points left
        for block in self._z_blocks:

            # map atoms from z matrix
            # to indices in reconstruction order
            ref = self._atom2index[block]

            # slice three context points
            # from the already reconstructed
            # points using the indices
            context = ps[:, ref[:, 1:]]
            p0 = context[:, :, 0]
            p1 = context[:, :, 1]
            p2 = context[:, :, 2]

            # obtain index of currently placed
            # point in original z-matrix
            idx = self._index2order[ref[:, 0] - len(self._fixed_atoms)]

            # get bonds, angles, torsions
            # using this z-matrix index
            b = bonds[:, idx, None]
            a = angles[:, idx, None]
            t = torsions[:, idx, None]

            # now we have three context points
            # and correct ic values to reconstruct the current point
            p, J = ic2xyz_deriv(p0, p1, p2, b, a, t)

            # compute jacobian
            dlogp += det3x3(J).abs().log().sum(-1)[:, None]

            # update list of reconstructed points
            ps = torch.cat([ps, p], dim=1)

        # finally make sure that atoms are sorted
        # from reconstruction order to original order
        ps = ps[:, self._atom2index]

        return ps.view(n_batch, -1), dlogp
