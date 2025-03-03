import torch
import torch.nn as nn
import numpy as np
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from typing import Union,Callable,List
from functools import partial
from FrEIA.utils import force_to
import torch.distributions as D 
import torch.nn.functional as F

coupling_block_dict = {
    "AllInOne":Fm.AllInOneBlock,
    "RQSpline":Fm.RationalQuadraticSpline
}

activation_dict = {
    "Tanh":torch.nn.Tanh,
    "ReLU":torch.nn.ReLU,
    "SiLU":torch.nn.SiLU,
    "ELU":torch.nn.ELU,
    "Sigmoid":torch.nn.Sigmoid,
    "Tanhshrink":torch.nn.Tanhshrink,
    "SoftSign":torch.nn.Softsign,
    "leakyReLU":torch.nn.LeakyReLU,
    "GeLU":nn.GELU
    }

condition_specs_dict = {
        "ignore_beta":[None,None],
        "log_beta":[0,(1,)]
    }
######################################################################################################################################
# Preprocessing of the condition
######################################################################################################################################

def _beta_processing_ignore_beta(beta_tensor:torch.tensor)-> None:
    """
    Ignore the input tensor. Use for unconditional INN
    """
    return None

def _beta_processing_log(beta_tensor:torch.tensor)-> torch.tensor:
    """
    Return the logarithm of the input tensor
    """
    return beta_tensor.log()

_beta_processing_dict = {
    "log_beta":_beta_processing_log,
    "ignore_beta":_beta_processing_ignore_beta
}

######################################################################################################################################
# Wrapper for INN operations
######################################################################################################################################
class INN_Model():
    def __init__(self,d:int,inn:Ff.InvertibleModule,device:str,latent_mode:str = "standard_normal",process_beta_mode:str = "log_beta",embedding_model:Callable=None)->None:
        """
        This is a wrapper class to handle all the operations on the distribution defined by the INN.

        parameters:
            d:                  Dimensionality fo the data space
            inn:                Invertible function
            device:             Device to run the code on 
            latent_mode:        Mode of the latent distribution of the INN
            process_beta_mode:  How to preprocess the inverse temperature passed to the INN
            embedding_model:    Function mapping the beta_tensor to a d'dimensional embedding
        """

        self.inn = inn
        self.device = device

        if latent_mode == "standard_normal":    
            self.p_0 = force_to(D.MultivariateNormal(torch.zeros(d).to(device), torch.eye(d)),device)
        
        self.latent_mode = latent_mode
        self.d = d
        self.process_beta_mode = process_beta_mode

        #Use learneable processing of the condition
        if self.process_beta_mode == "learnable": 
            print("Learable temperature embedding")
            self.beta_processing_function = embedding_model

        else:
            "Use standard temperature embedding"
            self.beta_processing_function = _beta_processing_dict[self.process_beta_mode]

    def load_state_dict(self,path:str)->None:
        """
        Load stored parameters.

        parameters:
            path:   Location of the stored parameters.
        """

        state_dict = torch.load(path,weights_only = False)["state_dict"]

        print("Load state dict for invertible function")

        self.inn.load_state_dict(state_dict=state_dict["INN"])

        if self.process_beta_mode == "learnable": 
            print("Load state dict for embedding model")
            self.beta_processing_function.load_state_dict(state_dict=state_dict["Embedder"])

    def eval(self):
        self.inn.eval()

        if self.process_beta_mode == "learnable":
            self.beta_processing_function.eval()

    def train(self,b:bool=True):
        self.inn.train(b)

        if self.process_beta_mode == "learnable":
            self.beta_processing_function.train(b)
    
    def log_prob_p_0(self,z_tensor:torch.tensor,beta_tensor:torch.tensor)->torch.tensor:
        """
        Compute the log-likelihood of the latent distribution.

        parameters:
            z_tensor:       Latent code
            beta_tensor:    Inverse temperature

        return:
            log_p_z:        Log-likelihood of the latent code 
        """

        if self.latent_mode == "standard_normal":
            log_p_z = self.p_0.log_prob(z_tensor)
        
        elif self.latent_mode == "temperature_scaling":
            log_p_z = self.d / 2 * torch.log(beta_tensor.reshape(-1) / (2 * np.pi)) - z_tensor.pow(2).sum(-1) * beta_tensor.reshape(-1) / 2

        return log_p_z

    def _beta_processing(self,beta_tensor:torch.tensor)->torch.tensor:
        """
        Compute the condition of the INN

        parameters:
            beta_tensor:    Inverse temperature

        return:
            beta_tensor_processed:  Processed inverse temperature

        """
        beta_tensor_processed = self.beta_processing_function(beta_tensor)

        if beta_tensor_processed is not None:
            return [beta_tensor_processed]
        else:
            return beta_tensor_processed

    def log_prob(self,x:torch.tensor,beta_tensor:Union[float,torch.tensor])->torch.tensor:
        """
        Compute the log-likelihood of data points.

        parameters:
            x:              Data points to evaluate
            beta_tensor:    Inverse temperature

        return:
            log_prob_x:     log-likelihood of x
        """

        #If only a float is givne for the inverse temperature, use it for the whole batch
        if isinstance(beta_tensor,float):
            beta_tensor = torch.ones([x.shape[0],1]).to(self.device).to(self.device) * beta_tensor

        z,jac = self.inn(x,self._beta_processing(beta_tensor),rev=False)

        p_0_z = self.log_prob_p_0(z_tensor = z,beta_tensor = beta_tensor)
        
        log_prob_x =  p_0_z + jac

        return log_prob_x

    def sample(self,n_samples:int,beta_tensor:Union[float,torch.tensor])-> torch.tensor:
        """
        Generate Samples following the distribution defined by the INN.

        parameters:
            n_samples:      Number of samples to generate
            beta_tensor:    Inverse temperature

        return:
            x:              Samples following the distribution defined by the INN
        """
        
        #If only a float is givne for the inverse temperature, use it for the whole batch
        if isinstance(beta_tensor,float):
            beta_tensor = torch.ones([n_samples,1]).to(self.device).to(self.device) * beta_tensor

        if self.latent_mode == "standard_normal":
            z = self.p_0.sample([n_samples])
        
        elif self.latent_mode == "temperature_scaling":
            z = torch.randn([n_samples,self.d]).to(self.device) * 1 / beta_tensor.sqrt()

        x,_ = self.inn(z,self._beta_processing(beta_tensor),rev=True)

        return x
    
    def parameters(self):
        """
        return the learnable parameters of the Wrapper
        """

        if self.process_beta_mode == "learnable":
            return list(self.inn.parameters()) + list(self.beta_processing_function.parameters())
        else:
            return self.inn.parameters()
        
######################################################################################################################################
# Invertible blocks for INN
######################################################################################################################################

class WrappedConditionalCouplingBlock(Fm.InvertibleModule):
    def __init__(self,dims_in,BlockType,subnet_constructor,dims_c = None,**kwargs):
        """
        Wrap a predefined coupling block and rehsape the condition to the required shape.

        parameters:
            dims_in: Tupel containing a list representing the shape of an input instance.
            BlockType: Predefined coupling block
            subnet_constructor: Sonstructor for the subnetd of the coupling blocks
            dims_c: Dimensinality of the condition as passed to this block during a call. Not the dimensinality of the condition expected by the internal block.
        """
        super().__init__(dims_in,dims_c)

        self.dims_c = dims_c
        self.dims_in = dims_in

        #Unconditional block
        if self.dims_c is None:
            self.dims_c_internal = []

        #Conditional block
        else:
            #Shape of the consition as expected by the internal block
            self.dims_c_internal = [(dims_c[0][0],*dims_in[0][1:])]
            #print(self.dims_c_internal,self.dims_c)

        #Initialize the coupling block
        self.block = BlockType(
            dims_in = dims_in,
            dims_c = self.dims_c_internal,
            subnet_constructor = subnet_constructor,
            **kwargs)
        
    def transform_condition(self,c:torch.tensor)->torch.tensor:
        """
        Reshape the condition of shape [N,dims_c] in to a condition of shape [N,dims_c,H,W]

        parameters:
            c:  Condition of shape [N,dims_C] 

        returns:
            c_recomputed:   Condition tensor of shape [N,dims_C,H,W] where the i th channel contains only the number at 
                            the i th position of the input. 
        """

        #Create conditional channels in case of convolutional subnetworks
        if len(self.dims_in[0]) > 1: #Convolutional subnet
            c_recomputed = c.reshape(c.shape[0],c.shape[1],1,1).expand(-1,-1,self.dims_in[0][1],self.dims_in[0][2])
        
        #No action needed in case of fully connected subnetworks
        else:
            c_recomputed = c

        #print(c_recomputed.shape)

        return c_recomputed

    def forward(self,x_or_z, c:Union[list[torch.tensor],None] = None, rev:bool = False, jac:bool = True)->torch.tensor:

        #Reshape the conditin passed to the block
        if self.dims_c is not None:
            reomputed_c = [self.transform_condition(c = c[0])]

        else:
            reomputed_c = None

        #reomputed_c = c[:,:,None,None]

        #Pass it through the internal coupling block
        output = self.block(
            x_or_z, 
            c = reomputed_c, 
            rev = rev, 
            jac = jac
        )
        
        return output

    def output_dims(self,input_dims):
        return input_dims

class ConditionalScalingLayer(Fm.InvertibleModule):
    def __init__(self,dims_in,dims_c = None,subnet_constructor:Callable = None):
        super().__init__(dims_in,dims_c)
        
        #convolutional layer
        if len(dims_in[0]) == 3:
            print("Initialize learnable scaling layer for convolutional input")
            self.weight_model = subnet_constructor(dims_c[0][0],dims_in[0][0])
            self.bias_model = subnet_constructor(dims_c[0][0],dims_in[0][0])
        
        else:
            print("Initialize learnable scaling layer for fully connected input")
            self.weight_model = subnet_constructor(dims_c[0][0],1)
            self.bias_model = subnet_constructor(dims_c[0][0],1)

    def forward(self,x_or_z, c:Union[List[torch.tensor],None] = None, rev:bool = False, jac:bool = True):

        log_w = 5 * F.tanh(self.weight_model(c[0]))
        b = self.bias_model(c[0])

        #2D convolution in put
        if len(x_or_z[0].shape) == 4:
            log_w = log_w.reshape(x_or_z[0].shape[0],self.dims_in[0][0],1,1)
            b = b.reshape(x_or_z[0].shape[0],self.dims_in[0][0],1,1)

            jac = torch.sum(log_w,dim = (1,2,3)) * self.dims_in[0][1] * self.dims_in[0][2]
        
        #Fully connected input
        else:
            jac = torch.sum(log_w,dim = 1) * x_or_z[0].shape[1]

        if not rev:
            x_trafo = x_or_z[0] * log_w.exp() + b

        else:
            x_trafo = (x_or_z[0] - b) / log_w.exp()
            jac = -jac

        assert(x_trafo.shape == x_or_z[0].shape)

        return (x_trafo,),jac

    def output_dims(self, input_dims):
        return input_dims
    
######################################################################################################################################
# Constructors for subnetworks
######################################################################################################################################

def constructor_subnet_fc_plain(c_in:int,c_out:int,c_hidden:int,activation_type:str)->nn.Module:
    """
    Initialize a fully connected neural network.

    parameters:
        c_in:               Number of input channels
        c_out:              Number of output channels
        c_hidden:           Dimensionality of the hidden layers
        activation_type:    String specifying the non-linear function 
    
    return:
        layers:             Fully connected neural network
    """

    activation = activation_dict[activation_type]

    layers = nn.Sequential(
        nn.Linear(c_in, c_hidden), 
        activation(),
        nn.Linear(c_hidden, c_hidden), 
        activation(),
        nn.Linear(c_hidden, c_hidden), 
        activation(),
        nn.Linear(c_hidden, c_hidden), 
        activation(),
        nn.Linear(c_hidden, c_out)
        )
    
    #Initialize the weights of the linear layers
    for layer in layers:
        if isinstance(layer,nn.Linear):
            nn.init.xavier_normal_(layer.weight)

    #Set the weights and the bias of the final convolution to zero
    layers[-1].weight.data.fill_(0.0)
    layers[-1].bias.data.fill_(0.0)

    return layers

def constructor_subnet_fc_configured(d_hidden:int,activation_type:str)->Callable:
    return partial(constructor_subnet_fc_plain,c_hidden = d_hidden,activation_type = activation_type)

def constructor_subnet_2Dconv_plain(c_in:int,c_out:int,c_hidden:int,activation_type:str)->nn.Module:
    """
    Initialize a convolutional neural network.

    parameters:
        c_in:               Number of input channels
        c_out:              Number of output channels
        c_hidden:           Dimensionality of the hidden layers
        activation_type:    String specifying the non-linear function 
    
    return:
        layers:             Fully connected neural network
    """

    activation = activation_dict[activation_type]

    #Construct the layers
    layers = nn.Sequential(
        nn.Conv2d(in_channels=c_in,out_channels=c_hidden,kernel_size=3,padding=1),
        activation(),
        nn.Conv2d(in_channels=c_hidden,out_channels=c_hidden,kernel_size=3,padding=1),
        activation(),
        nn.Conv2d(in_channels=c_hidden,out_channels=int(c_hidden / 2),kernel_size = 1),
        activation(),
        nn.Conv2d(in_channels=int(c_hidden / 2),out_channels=int(c_hidden / 2),kernel_size = 1),
        activation(),
        nn.Conv2d(in_channels=int(c_hidden / 2),out_channels=c_out,kernel_size = 1),
    )

    #Initialize the weights of the convolutional layers
    for layer in layers:
        if isinstance(layer,nn.Conv2d):
            nn.init.xavier_normal_(layer.weight)

    #Set the weights and the bias of the final convolution to zero
    layers[-1].weight.data.fill_(0.0)
    layers[-1].bias.data.fill_(0.0)

    return layers

def constructor_subnet_2D_conv_configured(c_hidden:int,activation_type:str)->Callable:
    return partial(constructor_subnet_2Dconv_plain,c_hidden = c_hidden,activation_type = activation_type)

def constructor_conditional_scaling_subnet_fc(c_in:int,c_out:int,c_hidden:int,activation_type:str)->Callable:
    """
    Initialize a fully connected neural network.

    parameters:
        c_in:               Number of input channels
        c_out:              Number of output channels
        c_hidden:           Dimensionality of the hidden layers
        activation_type:    String specifying the non-linear function 
    
    return:
        layers:             Fully connected neural network
    """

    print("Initialize fully connected subnetwork for conditional scaling")
    activation = activation_dict[activation_type]

    layers = nn.Sequential(
        nn.Linear(c_in, c_hidden), 
        activation(),
        nn.Linear(c_hidden, c_hidden), 
        activation(),
        nn.Linear(c_hidden, c_hidden), 
        activation(),
        nn.Linear(c_hidden, c_out)
        )
    
    #Initialize the weights of the linear layers
    for layer in layers:
        if isinstance(layer,nn.Linear):
            nn.init.xavier_normal_(layer.weight)

    #Set the weights and the bias of the final convolution to zero
    layers[-1].weight.data.fill_(0.0)
    layers[-1].bias.data.fill_(0.0)

    return layers

def constructor_conditional_scaling_subnet_fc_configured(d_hidden:int,activation_type:str)->Callable:
    return partial(constructor_conditional_scaling_subnet_fc,c_hidden = d_hidden,activation_type = activation_type)

######################################################################################################################################
# Constructors for invertible functions
######################################################################################################################################
class MLPEmbedder(nn.Module):
    def __init__(self,d_hidden,d_out,activation):

        super().__init__()
        layers = nn.Sequential(
            nn.Linear(d_out,d_hidden),
            activation(),
            nn.Linear(d_hidden,d_hidden),
            activation(),
            nn.Linear(d_hidden,d_hidden),
            activation(),
            nn.Linear(d_hidden,d_out)
        )

        self.layers = layers

        #Initialize the weights of the linear layers
        for layer in layers:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self,beta):

        z =  self.layers(beta.log()) + beta.log()

        return z
    
def set_up_sequence_INN_ScalarTheory(config:dict,training_set:torch.utils.data.Dataset = None)->INN_Model:
    """
    Initialisation of an INN object based on a FrEIA sequence INN.

    parameters:
        config:         Conifguration file to set up the model
        training_set:   Training set

    return:
        INN:  Initialized wrapper for INN operations
    """

    #Collect information for logging
    output_str = "*********************************************************************************************\n"
    output_str += "\nINFO:\n\nInitialize sequence INN\n\n"

    output_str += f"\tModel on device \t{config['device']}\n"

    ######################################################################################################################################
    #Initialize the invertible function
    ######################################################################################################################################

    inn = Ff.SequenceINN(1,config["config_data"]["N"],config["config_data"]["N"])

    #Get the coupling block class used internally
    coupling_block_class = coupling_block_dict[config["config_model"]["coupling_block_type"]]

    #Modify the configuration file for the condition in case of learnable embedding
    embedding_model = None
    if config["config_model"]["process_beta_parameters"]["mode"] == "learnable":

        condition_specs_dict["learnable"] = [0,(config["config_model"]["process_beta_parameters"]["learnable_temperature_embedding_dim"],)]

        #Initialize the embedding model for the inverse temperature
        embedding_model = MLPEmbedder(
            d_hidden = config["config_model"]["process_beta_parameters"]["d_hidden"],
            d_out = condition_specs_dict[config["config_model"]["process_beta_parameters"]["mode"]][1][0],
            activation = activation_dict[config["config_model"]["process_beta_parameters"]["activation_function_type"]]
            )
    
        embedding_model.to(config["device"])

    #Get the shape of the condition passed to the coupling blocks
    cond_num = condition_specs_dict[config["config_model"]["process_beta_parameters"]["mode"]][0]
    cond_shape = condition_specs_dict[config["config_model"]["process_beta_parameters"]["mode"]][1]

    output_str += "\tBeta Processing:\t\t" + config["config_model"]["process_beta_parameters"]["mode"] + "\n"
    output_str += f"\tCondition specs:\t\tidx:{cond_num}\tshape:{cond_shape}\n"

    ######################################################################################################################################
    #Add fixed global scaling to the network
    ######################################################################################################################################

    inn.append(
        ConditionalScalingLayer,
        subnet_constructor = constructor_conditional_scaling_subnet_fc_configured(d_hidden = config["config_model"]["d_hidden_fc_learnable_scaling"],activation_type = config["config_model"]["activation_function_type"]),
        cond_shape = cond_shape,
        cond = cond_num
        )
        
    ######################################################################################################################################
    #Add the coupling blocks with convolutional subnetworks
    ######################################################################################################################################
    for i in range(config["config_model"]["n_stages_conv"]):

        #Add downsampling and increase the number of channels accordingly
        inn.append(Fm.IRevNetDownsampling)

        #Get the number of data channels in this stage
        n_channels_hidden_i = config["config_model"]["n_channels_conv_hidden_list"][i]

        for j in range(config["config_model"]["coupling_block_number_per_stage_conv"]):
            
            #Add act norm
            if config["config_model"]["use_act_norm"]:
                inn.append(Fm.ActNorm)

            #constructor_conv_subnet_dict

            #Concatenete the condition to the input of the coupling block as additional channels, one per condition dimension
            if config["config_model"]["condition_inclusion_conv_layers"] == "concatenate":
                inn.append(
                    WrappedConditionalCouplingBlock,
                    BlockType = coupling_block_class,
                    subnet_constructor = constructor_subnet_2D_conv_configured(c_hidden = n_channels_hidden_i,activation_type = config["config_model"]["activation_function_type"]),
                    cond_shape = cond_shape,
                    cond = cond_num,
                    **config["config_model"]["coupling_block_params"]
                    )
            
            else:
                raise NotImplementedError()
            
            #Add permutation
            if config["config_model"]["coupling_block_type"] != "AllInOne":
                inn.append(Fm.PermuteRandom)

    ######################################################################################################################################
    #Add the coupling blocks with fully connected subnetworks
    ######################################################################################################################################
    inn.append(Fm.Flatten)

    for j in range(config["config_model"]["coupling_block_number_per_stage_fc"]):

        #Add act norm
        if config["config_model"]["use_act_norm"]:
            inn.append(Fm.ActNorm)

        #Concatenete the condition to the input of the coupling block as additional channels, one per condition dimension
        if config["config_model"]["condition_inclusion_fc_layers"] == "concatenate":

            inn.append(
                WrappedConditionalCouplingBlock,
                BlockType = coupling_block_class,
                subnet_constructor = constructor_subnet_fc_configured(d_hidden = config["config_model"]["d_hidden_fc"],activation_type = config["config_model"]["activation_function_type"]),
                cond_shape = cond_shape,
                cond = cond_num,
                **config["config_model"]["coupling_block_params"]
                )
            
        else:
            raise NotImplementedError()

        #Add permutation
        if config["config_model"]["coupling_block_type"] != "AllInOne":
            inn.append(Fm.PermuteRandom)

    #Set the model to the device
    inn.to(config["device"])

    ######################################################################################################################################
    #Wrapper for all INN operations
    ######################################################################################################################################
    INN = INN_Model(
        d = config["config_data"]["N"] * config["config_data"]["N"], 
        inn = inn, 
        device = config["device"],
        latent_mode=config["config_model"]["latent_mode"],
        process_beta_mode = config["config_model"]["process_beta_parameters"]["mode"],
        embedding_model=embedding_model
        )

    ######################################################################################################################################
    #Summary
    ######################################################################################################################################

    num_params = sum(p.numel() for p in INN.inn.parameters())
    output_str += f"\nNumber of model parameters: {num_params}"

    #Print information
    output_str += "\n*********************************************************************************************\n"
    if config["verbose"]: print(output_str)

    return INN

def set_up_sequence_INN_DoubleWell(config:dict,training_set:torch.utils.data.Dataset = None)->INN_Model:
    """
    Initialisation of an INN object based on a FrEIA sequence INN.

    parameters:
        config:         Conifguration file to set up the model
        training_set:   Training set

    return:
        INN:  Initialized wrapper for INN operations
    """

    #Collect information for logging
    output_str = "*********************************************************************************************\n"
    output_str += "\nINFO:\n\nInitialize sequence INN\n\n"

    output_str += f"\tModel on device \t{config['device']}\n"

    ######################################################################################################################################
    #Initialize the invertible function
    ######################################################################################################################################

    inn = Ff.SequenceINN(config["config_data"]["init_data_set_params"]["d"])

    #Get the coupling block class used internally
    coupling_block_class = coupling_block_dict[config["config_model"]["coupling_block_type"]]

    #Get the shape of the condition passed to the coupling blocks
    cond_num = condition_specs_dict[config["config_model"]["process_beta_parameters"]["mode"]][0]
    cond_shape = condition_specs_dict[config["config_model"]["process_beta_parameters"]["mode"]][1]

    output_str += "\tBeta Processing:\t\t" + config["config_model"]["process_beta_parameters"]["mode"] + "\n"
    output_str += f"\tCondition specs:\t\tidx:{cond_num}\tshape:{cond_shape}\n"

    ######################################################################################################################################
    #Add the coupling blocks with fully connected subnetworks
    ######################################################################################################################################

    for j in range(config["config_model"]["coupling_block_number_per_stage_fc"]):

        #Add act norm
        if config["config_model"]["use_act_norm"]:
            inn.append(Fm.ActNorm)

        #Concatenete the condition to the input of the coupling block as additional channels, one per condition dimension
        if config["config_model"]["condition_inclusion_fc_layers"] == "concatenate":

            inn.append(
                WrappedConditionalCouplingBlock,
                BlockType = coupling_block_class,
                subnet_constructor = constructor_subnet_fc_configured(d_hidden = config["config_model"]["d_hidden_fc"],activation_type = config["config_model"]["activation_function_type"]),
                cond_shape = cond_shape,
                cond = cond_num,
                **config["config_model"]["coupling_block_params"]
                )

        else:
            raise NotImplementedError()

        #Add permutation
        if config["config_model"]["coupling_block_type"] != "AllInOne":
            inn.append(Fm.PermuteRandom)

    #Set the model to the device
    inn.to(config["device"])

    ######################################################################################################################################
    #Wrapper for all INN operations
    ######################################################################################################################################
    INN = INN_Model(
        d = config["config_data"]["init_data_set_params"]["d"], 
        inn = inn, 
        device = config["device"],
        latent_mode=config["config_model"]["latent_mode"],
        process_beta_mode = config["config_model"]["process_beta_parameters"]["mode"]
        )

    ######################################################################################################################################
    #Summary
    ######################################################################################################################################

    num_params = sum(p.numel() for p in INN.inn.parameters())
    output_str += f"\nNumber of model parameters: {num_params}"

    #Print information
    output_str += "\n*********************************************************************************************\n"
    if config["verbose"]: print(output_str)

    return INN