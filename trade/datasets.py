import torch
import numpy as np
from torch.utils.data import Dataset
from FrEIA.utils import force_to
import torch.distributions as D
from typing import List
import json

###############################################################
#Transformations
###############################################################
class RandomChangeOfSign(torch.nn.Module):
    def __init__(self, p=0.5):
        super(RandomChangeOfSign,self).__init__()
        self.p = p

    def forward(self, img):
        factor = (2*(torch.rand(1) < self.p).float() - 1).item()
        img_new = factor*img

        return img_new
    
class RandomHorizentalRoll(torch.nn.Module):
    def __init__(self):
        super(RandomHorizentalRoll,self).__init__()
        
    def forward(self, img):

        N = img.shape[-1]
        
        steps = torch.randint(low = int(-np.floor(N/2)),high = int(np.floor(N/2)),size = (1,)).item()

        img_new = torch.roll(img, shifts = steps, dims = -1)

        return img_new
    
class RandomVerticalRoll(torch.nn.Module):
    def __init__(self):
        super(RandomVerticalRoll,self).__init__()
        
    def forward(self, img):

        N = img.shape[-1]
        
        steps = torch.randint(low = int(-np.floor(N/2)),high = int(np.floor(N/2)),size = (1,)).item()

        img_new = torch.roll(img, shifts = steps, dims = -2)

        return img_new

##########################################################################################
# GMM class
##########################################################################################

class GMM():
    def __init__(self,means:torch.tensor,covs:torch.tensor,weights:torch.tensor = None,device:str = "cpu")->None:
        """
        parameters:
            means: Tensor of shape [M,d] containing the locations of the gaussian modes
            covs: Tensor of shape [M,d,d] containing the covariance matrices of the gaussian modes
            weights: Tensor of shape [M] containing the weights of the gaussian modes. Uniform weights are used if not specified
        """

        #get dimensionality of the data set
        self.d = len(means[0])

        #Get the number of modes
        self.M = len(means)
        self.mode_list = []

        #Check weights
        if weights is None:
            self.weights = torch.ones(self.M) / self.M
        else:
            self.weights = weights

        if self.weights.sum() != 1.0: raise ValueError()

        #Initialize the normal modes
        for i in range(self.M):
            self.mode_list.append(force_to(D.MultivariateNormal(loc = means[i],covariance_matrix = covs[i]),device))

    def __call__(self,x:torch.tensor)->torch.tensor:
        """
        Evaluate the pdf of the model.

        parameters:
            x: Tensor of shape [N,d] containing the evaluation points

        returns:
            p: Tensor of shape [N] contaiing the pdf value for the evaluation points
        """

        p =  self.log_prob(x).exp()
        return p
    
    def log_prob(self,x)->torch.tensor:
        """
        Evaluate the log pdf of the model.

        parameters:
            x: Tensor of shape [N,d] containing the evaluation points

        returns:
            log_p: Tensor of shape [N] contaiing the log pdf value for the evaluation points
        """

        log_p_i_storage = torch.zeros([self.M,len(x)]).to(x.device)

        for i in range(self.M):
            log_p_i_storage[i] = self.mode_list[i].log_prob(x).squeeze()


        log_p = torch.logsumexp(log_p_i_storage + torch.log(self.weights).to(x.device)[:,None],dim = 0)

        return log_p
    
    def sample(self,N:int)->torch.tensor:
        """
        Generate samples following the distribution

        parameters:
            N: Number of samples

        return:
            s: Tensor of shape [N,d] containing the generated samples
        """
        weights = np.zeros(len(self.weights))
        weights[:-1] = self.weights[:-1].cpu().detach().numpy()
        weights[-1] = 1.0 - self.weights[:-1].cpu().detach().numpy().sum()
        i = np.random.choice(a = self.M,size = (N,),p = weights)
        u,c = np.unique(i,return_counts=True)

        s = torch.zeros([0,self.d])

        for i in range(self.M):
            s_i = self.mode_list[u[i]].sample([c[i]])
            s = torch.cat((s,s_i),dim = 0)

        return s

##########################################################################################
# Scalar Theory
##########################################################################################

class DataSetScalarTheory2D(Dataset):
    """
    This class is adapted from https://github.com/StefanWahl/Applying-Energy-Based-Models-on-the-Ising-model-and-a-scalar-lattice-field-theory-in-two-dimensions
    """
    def __init__(self,N:int,kappa_list:list[float],lambda_list:list[float],n_correlation_times:int = 2,max_samples:int = 6250,mode:str = "training",augment:bool = False,sigma_noise:float = 0.0,base_path:str = None):
        '''
        parameters:
            N:                          Width and lenght of the lattice 
            kappa_list:                 List of parameters kappa for which data is loaded
            lambda_list:                List of parameters lambda for which data is loaded
            n_correlation_times:        Number of correlation times between the staes used as training date
            max_samples:                Number od samples taken from the recorded data set to build the data set
            mode:                       Training or validation mode
            augment:                    Apply fixed transformation to the data
            sigma_noise:                Standard deviation of the dequantization noise added to each lattice site
            base_path:                  Location of the data
        '''
        super().__init__()

        # Set the base path
        if base_path is None:
            base_path = f"./data/ScalarTheory/{mode}_data/"

        # Internal data storage
        self.data = torch.zeros([0,1,N,N])
        self.kappa_action = torch.zeros([0,1])
        self.lambda_action = torch.zeros([0,1])

        self.sigma_noise = sigma_noise

        print("Initialize dataset...")

        # Load the stored states
        for l in lambda_list:
            for k in kappa_list:

                print(f"kappa = {k}, lambda = {l}:")

                folder_i = base_path+f"N_{N}_LANGEVIN_SPECIFIC_Data_Set/kappa_{k}_lambda_{l}/"
                data_i = torch.load(folder_i + "states_0.pt")
                print(f"\t{len(data_i)} instances loaded")

                # Get the inforamtion about the simulation
                with open(folder_i + "info_0.json","r") as file:
                    info_i = json.load(file)
                file.close()

                # Check 
                assert(info_i["kappa_action"] == k)
                assert(info_i["lambda_action"] == l)

                #Get the part of the stored states that is in equillibrium
                lower_lim_i = int(info_i["t_eq"] / info_i["freq_save_samples"])+1
                data_i = data_i[lower_lim_i:]
                print(f"\t{len(data_i)} instances in equilibrium")

                #Select states that are at least two correlation times away from each other
                step_size_i = int(n_correlation_times * abs(info_i["tau_action"]) / info_i["freq_save_samples"])+1
                data_i = data_i[::step_size_i].view(-1,1,N,N)
                print(f"\t{len(data_i)} independen instances")

                # Select the desired number of states
                indices_i = np.random.permutation(len(data_i))[:min([len(data_i),max_samples])]
                data_i = data_i[indices_i]

                # Use the symmetry of the problem to increase the number of training samples 
                if augment:
                    # Horizontal flip
                    data_horizontal_flip_i = torch.flip(data_i,[2])

                    # Vertical flip
                    data_vertical_flip_i = torch.flip(data_i,[3])

                    # Horizontal and vertical flip
                    data_horizontal_vertical_flip_i = torch.flip(data_i,[2,3])

                    data_i = torch.cat((data_i,data_horizontal_flip_i,data_vertical_flip_i,data_horizontal_vertical_flip_i),dim = 0)

                    # Use the negative data set
                    data_neg_i = -1 * data_i

                    data_i = torch.cat((data_i,data_neg_i),dim = 0)

                # Inverse temperatures
                kappa_tensor_i = torch.ones([len(data_i),1]) * k
                lambda_tensor_i = torch.ones([len(data_i),1]) * l

                self.data = torch.cat((self.data,data_i),0)#[:20]
                self.kappa_action = torch.cat((self.kappa_action,kappa_tensor_i),0)#[:20]
                self.lambda_action = torch.cat((self.lambda_action,lambda_tensor_i),0)#[:20]

                print(f"\t{len(data_i)} instances added to data set")
                print(f"\tsigma = {self.sigma_noise}\n")

        print("Data set succesfully initialized\n\n")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index:int):
        image = self.data[index] + torch.randn_like(self.data[index]) * self.sigma_noise
        return self.kappa_action[index],self.lambda_action[index],image

def ActionScalarTheory(mus,kappas,lambdas):

    '''
    This function is adapted from https://github.com/StefanWahl/Applying-Energy-Based-Models-on-the-Ising-model-and-a-scalar-lattice-field-theory-in-two-dimensions

    Action function for 2D lattice.
    
    parameters:
        mus:        Initial states
        kappas:     Tensor containing the hopping-parameters
        lambdas:    Tensor containing the quadric-couplings

    returns:
        actions:    Containing the action of the different states
    '''

    if isinstance(kappas,float):
        kappas = torch.ones(len(mus)).to(mus.device) * kappas
    if isinstance(lambdas,float):
        lambdas = torch.ones(len(mus)).to(mus.device) * lambdas

    lambdas = lambdas.squeeze()
    kappas = kappas.squeeze()

    #Get the quadric coupling
    actions = (1 - 2 * lambdas[:,None,None,None]) * mus.pow(2) +lambdas[:,None,None,None] * mus.pow(4)

    #Get the term depending on the hopping parameter
    actions += - 2 * kappas[:,None,None,None] * torch.roll(input=mus,shifts=1,dims=2) * mus
    actions += - 2 * kappas[:,None,None,None] * torch.roll(input=mus,shifts=1,dims=3) * mus

    actions = torch.sum(input=actions,dim = [1,2,3])

    return actions

def dActionScalarTheory_dkappa(mus,parameter_list:List[torch.Tensor|float] = None,device:str = None):

    actions = - 2  * torch.roll(input=mus,shifts=1,dims=2) * mus
    actions += - 2 * torch.roll(input=mus,shifts=1,dims=3) * mus

    actions = torch.sum(input=actions,dim = [1,2,3])

    return actions

def log_p_scalar_theory(x,beta_tensor,lambdas,device):
    """
    Logarithm of the unnormalized target distribution for the scalar theory.
    """
    return - ActionScalarTheory(x,beta_tensor,lambdas)

##########################################################################################
# 2D GMM
##########################################################################################

class DataSet2DGMM(Dataset):
    def __init__(self,d:int,temperature_list:list[float],mode:str = "training",base_path:str = None,n_samples:int = 50000):

        print("base_path: ",base_path)
        if base_path is None:
            base_path = f"./data/{d}D_DoubleWell/{mode}_data/"
        else:
            base_path = base_path + f"{mode}_data/"

        self.data = torch.zeros([0,d])
        self.beta = torch.zeros([0,1])

        for T in temperature_list:

            print(f"T = {T}:")

            folder_i = base_path+f"T_{T}_dim_{d}.pt"
            data_i = torch.load(folder_i)
            r=torch.randperm(len(data_i))
            data_i = data_i[r][:min([len(data_i),n_samples])]
            print(f"\t{len(data_i)} instances loaded")

            beta_tensor_i = torch.ones([len(data_i),1]) / T

            self.data = torch.cat((self.data,data_i),0)
            self.beta = torch.cat((self.beta,beta_tensor_i),0)

        print("Data set succesfully initialized\n\n")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index:int):
        return self.beta[index],self.data[index]
    
def log_p_2D_GMM(x:torch.tensor,beta_tensor:float,device:str,gmm:GMM):

    if isinstance(beta_tensor,float):
        beta_tensor = torch.ones(len(x),1).to(device) * beta_tensor

    log_prob = gmm.log_prob(x)*beta_tensor.squeeze()

    assert (log_prob.shape == torch.Size([len(x)]))

    return log_prob

def S_2D_GMM(x,beta,gmm,device):
    return - log_p_2D_GMM(x = x,beta_tensor = beta,device = device,gmm = gmm)

def dS_dbeta_2D_GMM(x,gmm,device,beta = None):
    return - log_p_2D_GMM(x = x,beta_tensor = 1.0,device = device,gmm = gmm)

dS_dparam_dict = {
    "ScalarTheory":dActionScalarTheory_dkappa,
    "2D_GMM":dS_dbeta_2D_GMM,
}

S_dict = {
    "ScalarTheory":ActionScalarTheory,
    "2D_GMM":S_2D_GMM,
}

log_p_target_dict = {
    "ScalarTheory":log_p_scalar_theory,
    "2D_GMM":log_p_2D_GMM,
}