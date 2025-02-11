import yaml
from dataclasses import dataclass, field
from lightning_trainable import TrainableHParams
from typing import List, Tuple
import torch

def read_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

@dataclass(kw_only=True)
class CouplingBlockHParams:
    """
    Base class for coupling block hyperparameters
    
    Attributes:
    hidden: List of int if resnet is false, else List of List of int
        default=[256, 256]
        Each entry creates a hidden layer with the specified number of units
    resnet: bool
        default=False
        If true, a resnet architecture is used
    parameter_aware: bool
        default=False
        If true, the coupling block receives the parameter as an additional input
    activation: str
        default="silu"
        Activation function used in the coupling block
    dropout: float
        default=0.0
        Dropout rate applied to the hidden layers
    volume_preserving: bool
        default=False
        If true, the coupling block is volume preserving
    zero_init: bool
        default=False
        If true, the coupling block is initialized with zeros
    """

    hidden: List = field(default_factory=lambda: [256, 256])
    resnet: bool = False
    parameter_aware: bool = False
    activation: str = "silu"
    dropout: float = 0.0
    volume_preserving: bool = False
    zero_init: bool = False

@dataclass(kw_only=True)
class RQSplineHParams(CouplingBlockHParams):
    """
    Hyperparameters for Rational Quadratic Spline coupling blocks

    Attributes:
    module_params: dict
        default={"bins": 8}
        Parameters passed to the RationalQuadraticSpline
    """

    module_params: dict = field(default_factory=lambda: {
    "bins": 8,
    })

    def __post_init__(self):
        if self.volume_preserving:
            raise NotImplementedError("Volume preserving coupling blocks are not implemented for RationalQuadraticSplines")
        self.coupling_transform = "RationalQuadraticSpline"
        self.elementwise_transform = "ElementwiseRationalQuadraticSpline"

@dataclass(kw_only=True)
class AffineHParams(CouplingBlockHParams):
    """
    Hyperparameters for Affine coupling blocks

    Attributes:
    module_params: dict
        default={"clamp": 0.3, "clamp_activation": "TANH", "learnable_clamp": True}
        Parameters passed to the AffineCouplingBlock
    """

    module_params: dict = field(default_factory=lambda: {
    "clamp": 0.3,
    "clamp_activation": "TANH",
    "learnable_clamp": True,
    })

    def __post_init__(self):
        self.coupling_transform = "GLOWCouplingBlock" if not self.volume_preserving else "GINCouplingBlock"
        self.elementwise_transform = "ActNorm" if not self.volume_preserving else None # TODO: This cannot be conditional
        if "clamp" in self.module_params.keys():
            self.module_params["clamp"] = torch.nn.Parameter(torch.tensor(self.module_params["clamp"]), requires_grad=self.module_params.get("learnable_clamp", False))
        if "learnable_clamp" in self.module_params.keys():
            self.module_params.pop("learnable_clamp")

@dataclass(kw_only=True)
class FlowHParams:
    """
    Base class for flow hyperparameters
    
    Attributes:
    dim: int | Tuple[int]
        Dimensionality of the input (set automatically)
    dims_c: List
        default=[]
        Dimensions of the conditioning input (set automatically)
    n_transforms: int
        default=20
        Number of coupling blocks
    prior: str
        default="normal"
        Prior distribution for the latent variable
    coupling_hparams: dict | CouplingBlockHParams
        default=AffineHParams()
        Hyperparameters for the coupling blocks (see CouplingBlockHParams)
    augmentation_dim: int
        default=0
        Dimensionality of the augmentation
    parameter_preprocessing: str | None
        default="log"
        Preprocessing applied to the parameter (see flow.py)
    scale_latent_with_parameter: bool
        default=True
        If true, the latent variable is scaled with the parameter
    trainable_prior: bool
        default=False
        If true, the prior is trainable
    permutation_type: str | None
        default="hard"
        Type of permutation applied between coupling blocks
    use_actnorm: bool
        default=False
        Whether to use ActNorm layers between coupling blocks
    coordinate_transform: bool | str | dict
        default=True
        If true, a coordinate transformation is applied
        If a string or dict, the type of coordinate transformation is specified (see coordinate_transform.py)
    sigmoid_layer: bool
        default=False
        If true, a sigmoid layer is applied at the data output of the flow
    """

    dim: int | Tuple[int]
    dims_c: List = field(default_factory=lambda: [])

    n_transforms: int = 20
    prior: str = "normal"
    coupling_hparams: dict | CouplingBlockHParams =field(default_factory=lambda: AffineHParams())
    augmentation_dim: int = 0
    parameter_preprocessing: str | None = "log"
    scale_latent_with_parameter: bool = True
    trainable_prior: bool = False
    permutation_type: str | None = "hard"
    use_actnorm: bool = False
    coordinate_transform: bool | str | dict = True
    sigmoid_layer: bool = False

@dataclass(kw_only=True)
class BGFlowHParams:
    """
    Hyperparameters for BGFlow

    Attributes:
    dim: int | Tuple[int]
        Dimensionality of the input (set automatically)
    spline_disable_identity_transform: bool
        default=True
        If true, the identity transform initialization is disabled for the spline coupling blocks
    min_energy_structure_path: str
        default="./bgflow_wrapper/data_ala2/position_min_energy.pt"
        Path to the minimum energy structure
    torsion_shifts: bool
        default=True
        If true, torsion shifts are applied
    conditioner_type: str
        default="residual"
        Type of conditioner applied to the input
    use_sobol_prior: bool
        default=False
        If true, a Sobol prior is used
    parameter_preprocessing: str | None
        default="log"
        Preprocessing applied to the parameter (see flow.py)
    n_workers: int
        default=16
        Number of workers used for the energy model
    parameter_aware: bool   
        default=True
        If true, the flow receives the parameter as an additional input
    constrain_chirality: bool
        default=True
        If true, chirality constraints are applied
    activation: str
        default="silu"
        Activation function used in the flow
    architecture: List[List[str | bool]]
        default=None
        Architecture of the flow (see bgflow_wrapper.models)
    temperature_steerable: bool
        default=False
        If true, the flow is constructed as a temperature-steerable flow
    """

    dim: int | Tuple[int]
    spline_disable_identity_transform: bool = True
    min_energy_structure_path: str = (
        "./bgflow_wrapper/data_ala2/position_min_energy.pt"
    )
    torsion_shifts: bool = True
    conditioner_type: str = "residual"
    use_sobol_prior: bool = False
    parameter_preprocessing: str | None = "log"
    n_workers: int = 16
    parameter_aware: bool = True
    constrain_chirality: bool = True
    activation: str = "silu"
    architecture: List[List[str | bool]] = None
    temperature_steerable: bool = False

@dataclass(kw_only=True)
class LossHParams:
    """
    Base class for loss hyperparameters

    Attributes:
    pct_start: float
        default=0.0
        Percentage of steps before the loss starts to be evaluated
    start_value: float
        default=1.0
        Initial value of the loss weight
    end_value: float
        default=1.0
        Final value of the loss weight
    adaptive: bool
        default=False
        If true, the loss weight is adapted based on gradient magnitude
    adaptive_weight: float
        default=1.0
        Weight that the loss should have after being adapted according to the gradient magnitude
    alpha_adaptive_update: float
        default=0.1
        Momentum for the EMA for the adaptive update
    log_clip: float | None
        default=1e3
        Value at which the loss is clipped to log(1 + value - log_clip) + log_clip
    clip: float | None
        default=1e9
        Value at which the loss is clipped to clip
    additional_kwargs: dict
        default={}
        Additional keyword arguments passed to the loss function
    """

    pct_start: float = 0.0
    start_value: float = 1.0
    end_value: float = 1.0
    adaptive: bool = False
    adaptive_weight: float = 1.0
    alpha_adaptive_update: float = 0.1
    log_clip: float | None = 1e3
    clip: float | None =  1e9
    additional_kwargs: dict = field(default_factory=lambda: {})

@dataclass(kw_only=True)
class ParameterPriorHParams:
    """
    Hyperparameters for the parameter prior

    Attributes:
    parameters: float | Tuple[float]
        default=1.0
        If a float, the parameter is fixed to this value
        If a tuple, the parameter is sampled from a skewed log-uniform distribution between the two values
    s_min: float | None
        default=None
        The skew of the distribution to the edges at the start of training
        0 = delta distribution, 1 = uniform distribution, inf = delta distribution at the edges
    s_max: float | None
        default=None
        The skew of the distribution to the edges at the end of training
        0 = delta distribution, 1 = uniform distribution, inf = delta distribution at the edges
    sample_parameter_per_batch: bool
        default=False
        If true, only one target parameter is sampled per batch and repeated for all samples
    """

    parameters: float | Tuple[float] = 1.0
    s_min: float | None = None
    s_max: float | None = None
    sample_parameter_per_batch: bool = False

@dataclass(kw_only=True)
class DataAugmentorHParams:
    """
    Hyperparameters for the data augmentor

    Attributes:
    rotation: bool | List[float]
        default=False
        If true, random rotations are applied
        If a list, the rotations are sampled from a uniform distribution between the two values
    translation: bool | List[float]
        default=False
        If true, random translations are applied
        If a list, the translations are sampled from a uniform distribution between the two values
    flip: bool | float
        default=False
        If true, random flips are applied
        If a float, the probability of a flip is given
    add_noise: bool | float
        default=False
        If true, random noise is added
        If a float, the noise level is given
    scale: bool | List[float]
        default=False
        If true, random scalings are applied
        If a list, the scalings are sampled from a uniform distribution between the two values
    """

    rotation: bool | List[float] = False
    translation: bool | List[float] = False
    flip: bool | float = False
    add_noise: bool | float = False
    scale: bool | List[float] = False

class BoltzmannGeneratorHParams(TrainableHParams):
    """
    Hyperparameters for Boltzmann Generator
    
    Attributes:
    target_parameter_name: str
        default="temperature"
        Name of the target parameter (determines how the energy is differentiated and the TRADE loss is evaluated)
    parameter_reference_value: float
        default=1.0
        Reference value of the target parameter, which the model should treat as 1.0
    flow_type: str
        default="freia"
        Whether to construct the flow with bgflow or freia
    flow_hparams: FlowHParams | BGFlowHParams | dict
        Hyperparameters for the flow (see FlowHParams and BGFlowHParams)
    plotting: dict
        default={"interval": 5, "n_samples": 100000, "parameters": [300, 600, 1000]}
        Parameters for the plotting function (see boltzmann_generator.py)
    dataset: dict
        default={"training_data_parameters": 600, "name": "ala2", "split": [0.8, 0.1, 0.1]}
        Parameters for the dataset (see data/__init__.py)
    parameter_prior_hparams: dict | ParameterPriorHParams
        default=ParameterPriorHParams()
        Hyperparameters for the parameter prior (see ParameterPriorHParams)
    nll_loss: dict | LossHParams | None
        default=None
        Hyperparameters for the NLL loss (see LossHParams)
    kl_loss: dict | LossHParams | None
        default=None
        Hyperparameters for the KL loss (see LossHParams)
    temperature_weighted_loss: dict | LossHParams | None
        default=None
        Hyperparameters for the temperature-weighted loss (see LossHParams)
    trade_loss: dict | LossHParams | None
        default=None
        Hyperparameters for the trade loss (see LossHParams)
    var_loss: dict | LossHParams | None
        default=None
        Hyperparameters for the variance loss (see LossHParams)
    update_adaptive_weights_every: int
        default=25
        Number of steps between adaptive loss weight updates
    update_min_nll_every: int | None
        default=None
        Number of steps between minimum NLL updates (used to normalize the temperature-weighted loss)
    dequantization_noise: float | list
        default=0.0
        Dequantization noise applied to the data. If a list, the noise is sampled from a log-uniform distribution between the two values
    softflow: bool
        default=False
        If true, the noise strength is given as a condition to the flow
    """

    target_parameter_name: str = "temperature"
    parameter_reference_value: float = 1.0
    flow_type: str = "freia"
    flow_hparams: FlowHParams | BGFlowHParams | dict
    data_augmentation_hparams: DataAugmentorHParams | dict = DataAugmentorHParams()

    plotting: dict = {
        "interval": 5,
        "n_samples": 100000,
        "parameters": [300, 600, 1000],
    }

    dataset: dict = {
        "training_data_parameters": [600],
        "name": "ala2",
        "split": [0.8, 0.1, 0.1],
    }

    parameter_prior_hparams: dict | ParameterPriorHParams = ParameterPriorHParams()

    nll_loss: dict | LossHParams | None = None
    kl_loss: dict | LossHParams | None = None
    temperature_weighted_loss: dict | LossHParams | None = None
    trade_loss: dict | LossHParams | None = None
    var_loss: dict | LossHParams | None = None

    update_adaptive_weights_every: int = 25
    update_min_nll_every: int | None = None
    dequantization_noise: float | list = 0.0
    softflow: bool = False


    def __post_init__(self):
        if isinstance(self.energy_lambda, float):
            self.energy_lambda = [self.energy_lambda, self.energy_lambda]
        if len(self.energy_lambda) != 2:
            raise ValueError("energy_lambda must be a float or a list of two floats")
        if not 0 <= self.energy_start_pct <= 1:
            raise ValueError("energy_start_pct must be in [0, 1]")
        if not 0 <= self.energy_clip:
            raise ValueError("energy_clip must be non-negative")
        super().__post_init__()
