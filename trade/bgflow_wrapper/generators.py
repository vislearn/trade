import bgflow as bg
from bgflow import BoltzmannGenerator
from trade.bgflow_wrapper.ic_scaler import ICScaler
from bgflow import TORSIONS, BONDS, ANGLES
from trade.bgflow_wrapper.parsing import parse_architecture_layer_str
from bgmol import bond_constraints
import numpy as np
import torch
import torch.nn.functional as F
import bgmol
from trade.config import BGFlowHParams

def create_generator(cfg: BGFlowHParams, energy_model, system) -> BoltzmannGenerator:

    z_matrix = bgmol.systems.ala2.DEFAULT_GLOBAL_Z_MATRIX

    coordinate_trafo = bg.GlobalInternalCoordinateTransformation(
        z_matrix=z_matrix,
        enforce_boundaries=True,
        normalize_angles=True,
    )

    print("Number of bond constraints:", system.system.getNumConstraints())
    shape_info = bg.ShapeDictionary.from_coordinate_transform(
        coordinate_trafo,
        n_constraints=system.system.getNumConstraints(),  # Pass number of hydrogen bond length constraints
    )

    ##### Prepare priors #####

    prior_type = dict()
    prior_kwargs = dict()

    # Only torsions keep the default, which is a [0, 1] uniform distribution
    prior_type[BONDS] = bg.TruncatedNormalDistribution
    prior_type[ANGLES] = bg.TruncatedNormalDistribution

    prior_kwargs[BONDS] = {
        "mu": 0.5, # torch.ones(shape_info[BONDS])*0.5,
        "sigma": 0.1, # torch.eye(*shape_info[BONDS])*0.1,
        "lower_bound": 0.0,
        "upper_bound": 1.0,
        "assert_range": False,
    }
    prior_kwargs[ANGLES] = {
        "mu": 0.5, # torch.ones(shape_info[ANGLES])*0.5,
        "sigma": 0.1, # torch.eye(*shape_info[ANGLES])*0.1,
        "lower_bound": 0.0,
        "upper_bound": 1.0,
        "assert_range": False,
    }

    builder = bg.BoltzmannGeneratorBuilder(
        prior_dims=shape_info,
        prior_type=prior_type,
        prior_kwargs=prior_kwargs,
        target=energy_model,
        device="cuda",
        dtype=torch.float32,
    )

    shift_gen = torch.Generator().manual_seed(123)
    mask_gen = np.random.default_rng(seed=321)

    if cfg.activation == "relu":
        activation = F.relu
    elif cfg.activation == "silu":
        activation = F.silu
    else:
        raise ValueError(f"Activation not implemented: {cfg.activation}")

    for i, (what_str, on_str, add_reverse) in enumerate(cfg.architecture):
        if cfg.temperature_steerable:
            if i in [0, 1]:
               transformer_type = bg.TemperatureSteerableConditionalSplineTransformer
               transformer_kwargs = dict(left=0, right=1, bottom=0, top=1)

               if cfg.spline_disable_identity_transform:
                   transformer_kwargs["spline_disable_identity_transform"] = True
                   print("Disabling BGFlow identity transform option for splines.")
               conditioner_kwargs = dict()
            else:
                transformer_type = bg.AffineTransformer
                transformer_kwargs = dict(preserve_volume=True, restrict_to_unit_interval=False)
                conditioner_kwargs = dict(use_scaling=False, init_identity=True)
        else:
            transformer_type = bg.ConditionalSplineTransformer
            transformer_kwargs = dict(left=0, right=1, bottom=0, top=1)
            if cfg.spline_disable_identity_transform:
               transformer_kwargs["spline_disable_identity_transform"] = True
               print("Disabling BGFlow identity transform option for splines.")
            conditioner_kwargs = dict()



        what, on = parse_architecture_layer_str(
            what_str, on_str, builder.current_dims
        )

        builder.add_condition(
            what,
            on=on,
            add_reverse=add_reverse,
            rng=mask_gen,
            conditioner_type=cfg.conditioner_type,
            transformer_type=transformer_type,
            transformer_kwargs=transformer_kwargs,
            context_dims=1 if cfg.parameter_aware else 0,
            activation=activation,
            **conditioner_kwargs,
        )


        if i == len(cfg.architecture) - 1:
            builder.add_layer(ClampTransform(0, 1))

        if cfg.torsion_shifts:
            builder.add_torsion_shifts(torch.rand((), generator=shift_gen))

    if cfg.constrain_chirality:
        chiral_torsions = bgmol.is_chiral_torsion(
            coordinate_trafo.torsion_indices, system.mdtraj_topology
        )
        builder.add_constrain_chirality(chiral_torsions)
        print(
            "Added constraint for chirality of torsions:",
            np.argwhere(chiral_torsions).flatten(),
        )

    constraints = bond_constraints(system.system, coordinate_trafo)

    ##### IC scaling #####
    min_energy_structure = torch.load(cfg.min_energy_structure_path).to(
        dtype=torch.get_default_dtype()
    )
    ic_scaler = ICScaler(
        min_energy_structure,
        ic_trafo=coordinate_trafo,
        constrained_bond_indices=constraints[0],
    )
    ic_scaler = ic_scaler.to("cuda")
    builder.add_layer(ic_scaler)
    ##########

    if len(constraints[0]) > 0:
        builder.add_merge_constraints(*constraints)

    builder.add_map_to_cartesian(coordinate_transform=coordinate_trafo)

    if cfg.parameter_preprocessing is None or not cfg.parameter_aware:
        context_preprocessor = None
    elif cfg.parameter_preprocessing == "log":
        context_preprocessor = torch.log
    else:
        raise ValueError(f"Unknown parameter preprocessing: {cfg.parameter_preprocessing}")

    generator = builder.build_generator(
        use_sobol=cfg.use_sobol_prior,
        context_preprocessor=context_preprocessor,
    )

    print("Total number of parameters:", sum(p.numel() for p in generator.parameters()))
    return generator, energy_model


class SigmoidTransform(bg.Flow):
    def __init__(self):
        super().__init__()

    def _forward(self, *xs, **kwargs):
        ys = [torch.sigmoid(x*3) for x in xs]
        dlogp = sum(torch.sum(torch.log(x*3 * (1 - x*3)), dim=-1, keepdims=True) for x in xs) + np.log(3)*sum([x.shape[-1] for x in xs])
        return *ys, dlogp
    
    def _inverse(self, *ys, **kwargs):
        xs = [torch.logit(y)/3 for y in ys]
        dlogp = - sum(torch.sum(torch.log(y * (1 - y)), dim=-1, keepdims=True) for y in ys) - np.log(3)*sum([y.shape[-1] for y in ys])
        return *xs, dlogp
        

class ClampTransform(bg.Flow):
    def __init__(self, lower, upper):
        super().__init__()
        self.lower = lower
        self.upper = upper

    def _forward(self, *xs, **kwargs):
        ys = [torch.clamp(x, self.lower, self.upper) for x in xs]
        return *ys, torch.zeros(1, device=xs[0].device)
    
    def _inverse(self, *ys, **kwargs):
        xs = [torch.clamp(y, self.lower, self.upper) for y in ys]
        return *xs, torch.zeros(1, device=xs[0].device)
