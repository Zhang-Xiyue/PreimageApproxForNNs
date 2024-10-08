import json
import math
import numpy as np
import os
import torch
from torch import autograd
from torch.distributions import Uniform
from .utils import logger, eyeC
from .patches import Patches, patches_to_matrix
from .linear_bound import LinearBound
# import preimage_arguments
import arguments
import pickle
def load_act_vecs(dataset_tp):
    # if arguments.Config["model"]["onnx_path"] is None:
    act_file = os.path.join(arguments.Config['preimage']["sample_dir"], 'act_vec_{}_{}.pkl'.format(dataset_tp, arguments.Config["preimage"]["atk_tp"]))
    with open(act_file, 'rb') as f:
        activation = pickle.load(f)
    if "MNIST" in dataset_tp:
        pre_relu_layer = ['2', '4', '6', '8', '10']
    elif "auto_park" in dataset_tp:
        pre_relu_layer = ['2']
    elif "vcas" in dataset_tp:
        pre_relu_layer = ["H0"]
    elif dataset_tp == 'cartpole' or dataset_tp == "lunarlander":
        pre_relu_layer = ['8', '10']
    elif "dubinsrejoin" in dataset_tp:
        pre_relu_layer = ["StatefulPartitionedCall/sequential/dense/BiasAdd:0",  "StatefulPartitionedCall/sequential/dense_1/BiasAdd:0"]
    acti_vecs = []
    for i, layer in enumerate(pre_relu_layer):
        act_vec = activation[layer].cpu().detach().numpy()
        acti_vecs.append(act_vec)
    return acti_vecs

def calc_history_idxs(acti_vecs, part_history):
    sample_part = None
    for i, layer_info in enumerate(part_history): # enumerate over relu layers
        neuron_idxs = layer_info[0]
        if len(neuron_idxs) == 0:
            continue
        neuron_signs = layer_info[1]
        for j, neuron_id in enumerate(neuron_idxs):
            neuron_sign = neuron_signs[j]
            if neuron_sign == +1:
                temp_idx = np.where(acti_vecs[i][:, neuron_id]>=0)[0]
            elif neuron_sign == -1:
                temp_idx = np.where(acti_vecs[i][:, neuron_id]<0)[0]
            else:
                print("neuron sign assignment error")
            if sample_part is None:
                sample_part = set(temp_idx)
            else:
                sample_part = sample_part.intersection(set(temp_idx))
    return sample_part
class Perturbation:
    r"""
    Base class for a perturbation specification. Please see examples
    at `auto_LiRPA/perturbations.py`.

    Examples:

    * `PerturbationLpNorm`: Lp-norm (p>=1) perturbation.

    * `PerturbationL0Norm`: L0-norm perturbation.

    * `PerturbationSynonym`: Synonym substitution perturbation for NLP.
    """

    def __init__(self):
        pass

    def set_eps(self, eps):
        self.eps = eps

    def concretize(self, x, A, sign=-1, aux=None):
        r"""
        Concretize bounds according to the perturbation specification.

        Args:
            x (Tensor): Input before perturbation.

            A (Tensor) : A matrix from LiRPA computation.

            sign (-1 or +1): If -1, concretize for lower bound; if +1, concretize for upper bound.

            aux (object, optional): Auxilary information for concretization.

        Returns:
            bound (Tensor): concretized bound with the shape equal to the clean output.
        """
        raise NotImplementedError

    def init(self, x, aux=None, forward=False):
        r"""
        Initialize bounds before LiRPA computation.

        Args:
            x (Tensor): Input before perturbation.

            aux (object, optional): Auxilary information.

            forward (bool): It indicates whether forward mode LiRPA is involved.

        Returns:
            bound (LinearBound): Initialized bounds.

            center (Tensor): Center of perturbation. It can simply be `x`, or some other value.

            aux (object, optional): Auxilary information. Bound initialization may modify or add auxilary information.
        """

        raise NotImplementedError


"""Perturbation constrained by the L_0 norm (assuming input data is in the range of 0-1)."""
class PerturbationL0Norm(Perturbation):
    def __init__(self, eps, x_L=None, x_U=None, ratio=1.0):
        self.eps = eps
        self.x_U = x_U
        self.x_L = x_L
        self.ratio = ratio

    def concretize(self, x, A, sign=-1, aux=None):
        if A is None:
            return None

        eps = math.ceil(self.eps)
        x = x.reshape(x.shape[0], -1, 1)
        center = A.matmul(x)

        x = x.reshape(x.shape[0], 1, -1)

        original = A * x.expand(x.shape[0], A.shape[-2], x.shape[2])
        neg_mask = A < 0
        pos_mask = A >= 0

        if sign == 1:
            A_diff = torch.zeros_like(A)
            A_diff[pos_mask] = A[pos_mask] - original[pos_mask]# changes that one weight can contribute to the value
            A_diff[neg_mask] = - original[neg_mask]
        else:
            A_diff = torch.zeros_like(A)
            A_diff[pos_mask] = original[pos_mask]
            A_diff[neg_mask] = original[neg_mask] - A[neg_mask]

        # FIXME: this assumes the input pixel range is between 0 and 1!
        A_diff, _= torch.sort(A_diff, dim = 2, descending=True)

        bound = center + sign * A_diff[:, :, :eps].sum(dim = 2).unsqueeze(2) * self.ratio

        return bound.squeeze(2)

    def init(self, x, aux=None, forward=False):
        # For other norms, we pass in the BoundedTensor objects directly.
        x_L = x
        x_U = x
        if not forward:
            return LinearBound(None, None, None, None, x_L, x_U), x, None
        batch_size = x.shape[0]
        dim = x.reshape(batch_size, -1).shape[-1]
        eye = torch.eye(dim).to(x.device).unsqueeze(0).repeat(batch_size, 1, 1)
        lw = eye.reshape(batch_size, dim, *x.shape[1:])
        lb = torch.zeros_like(x).to(x.device)
        uw, ub = lw.clone(), lb.clone()
        return LinearBound(lw, lb, uw, ub, x_L, x_U), x, None

    def __repr__(self):
        return 'PerturbationLpNorm(norm=0, eps={})'.format(self.eps)


"""Perturbation constrained by the L_p norm."""
class PerturbationLpNorm(Perturbation):
    def __init__(self, eps=0, norm=np.inf, x_L=None, x_U=None):
        self.eps = eps
        self.norm = norm
        self.dual_norm = 1 if (norm == np.inf) else (np.float64(1.0) / (1 - 1.0 / self.norm))
        self.x_L = x_L
        self.x_U = x_U
        self.sparse = False

    def get_input_bounds(self, x, A):
        if self.sparse:
            if self.x_L_sparse.shape[-1] == A.shape[-1]:
                x_L, x_U = self.x_L_sparse, self.x_U_sparse
            else:
                # In backward mode, A is not sparse
                x_L, x_U = self.x_L, self.x_U
        else:
            x_L = x - self.eps if self.x_L is None else self.x_L
            x_U = x + self.eps if self.x_U is None else self.x_U
        return x_L, x_U

    # If A is an identity matrix, we will handle specially.
    def concretize_matrix(self, x, A, sign, extra_constr):
        if not isinstance(A, eyeC):
            # A has (Batch, spec, *input_size). For intermediate neurons, spec is *neuron_size.
            A = A.reshape(A.shape[0], A.shape[1], -1)

            if extra_constr is not None:
                # For each neuron, we have a beta, so beta size is (Batch, *neuron_size, n_beta) (in A, spec is *neuron_size).
                # For intermediate layer neurons, A has *neuron_size specifications.
                beta = extra_constr['beta']
                beta = beta.view(beta.size(0), -1, beta.size(-1))
                # coeffs are linear relationships between split neurons and x. They have size (batch, n_beta, *input_size), and unreated to neuron_size.
                beta_coeffs = extra_constr['coeffs']
                beta_coeffs = beta_coeffs.view(beta_coeffs.size(0), beta_coeffs.size(1), -1)
                # biases are added for each batch each spec, size is (batch, n_beta), and unrelated to neuron_size.
                beta_bias = extra_constr['bias']
                # Merge beta into extra A and bias. Extra A has size (batch, spec, *input_size). For intermediate neurons, spec is *neuron_size.
                extra_A = torch.einsum('ijk,ikl->ijl', beta, beta_coeffs)
                # Merge beta into the bias term. Output has size (batch, spec).
                extra_bias = torch.einsum('ijk,ik->ij', beta, beta_bias)

        if self.norm == np.inf:
            # For Linfinity distortion, when an upper and lower bound is given, we use them instead of eps.
            x_L, x_U = self.get_input_bounds(x, A)
            x_ub = x_U.reshape(x_U.shape[0], -1, 1)
            x_lb = x_L.reshape(x_L.shape[0], -1, 1)
            # Find the uppwer and lower bound similarly to IBP.
            center = (x_ub + x_lb) / 2.0
            diff = (x_ub - x_lb) / 2.0
            if not isinstance(A, eyeC):
                if extra_constr is not None:
                    # Extra linear and bias terms from constraints.
                    print(
                        f'A extra: {(sign * extra_A).abs().sum().item()}, '
                        f'b extra: {(sign * extra_bias).abs().sum().item()}')
                    A = A - sign * extra_A
                    bound = A.matmul(center) - sign * extra_bias.unsqueeze(-1) + sign * A.abs().matmul(diff)
                else:
                    bound = A.matmul(center) + sign * A.abs().matmul(diff)
            else:
                assert extra_constr is None
                # A is an identity matrix. No need to do this matmul.
                bound = center + sign * diff
        else:
            assert extra_constr is None
            x = x.reshape(x.shape[0], -1, 1)
            if not isinstance(A, eyeC):
                # Find the upper and lower bounds via dual norm.
                deviation = A.norm(self.dual_norm, -1) * self.eps
                bound = A.matmul(x) + sign * deviation.unsqueeze(-1)
            else:
                # A is an identity matrix. Its norm is all 1.
                bound = x + sign * self.eps
        bound = bound.squeeze(-1)
        return bound

    # If A is an identity matrix, we will handle specially.
    def concretize_matrix_poly(self, x, A, bias, sign, extra_constr):
        if not isinstance(A, eyeC):
            # A has (Batch, spec, *input_size). For intermediate neurons, spec is *neuron_size.
            A = A.reshape(A.shape[0], A.shape[1], -1)
            # print('check bias shape', bias.shape)
            if extra_constr is not None:
                # For each neuron, we have a beta, so beta size is (Batch, *neuron_size, n_beta) (in A, spec is *neuron_size).
                # For intermediate layer neurons, A has *neuron_size specifications.
                beta = extra_constr['beta']
                beta = beta.view(beta.size(0), -1, beta.size(-1))
                # coeffs are linear relationships between split neurons and x. They have size (batch, n_beta, *input_size), and unreated to neuron_size.
                beta_coeffs = extra_constr['coeffs']
                beta_coeffs = beta_coeffs.view(beta_coeffs.size(0), beta_coeffs.size(1), -1)
                # biases are added for each batch each spec, size is (batch, n_beta), and unrelated to neuron_size.
                beta_bias = extra_constr['bias']
                # Merge beta into extra A and bias. Extra A has size (batch, spec, *input_size). For intermediate neurons, spec is *neuron_size.
                extra_A = torch.einsum('ijk,ikl->ijl', beta, beta_coeffs)
                # Merge beta into the bias term. Output has size (batch, spec).
                extra_bias = torch.einsum('ijk,ik->ij', beta, beta_bias)
        if self.norm == np.inf:
            # For Linfinity distortion, when an upper and lower bound is given, we use them instead of eps.
            # if arguments.Config["model"]["onnx_path"] is None:
            #     sample_file = os.path.join(arguments.Config["preimage"]["sample_dir"], 'sample_{}.pt'.format(arguments.Config["data"]["dataset"]))
            #     prop_samples = torch.load(sample_file)     
            #     prop_samples = prop_samples.reshape(prop_samples.shape[0], -1)  
            # else:
            #     if arguments.Config["data"]["dataset"] == "vcas":
            #         samples = np.load(os.path.join(arguments.Config["preimage"]["sample_dir"], "sample_{}_{}.npy".format(arguments.Config["data"]["dataset"], arguments.Config["preimage"]["upper_time_loss"])))
            #     else:
            #         samples = np.load(os.path.join(arguments.Config["preimage"]["sample_dir"], "sample_{}.npy".format(arguments.Config["data"]["dataset"])))
            #     prop_samples = np.squeeze(samples, axis=1)
            #     prop_samples = torch.tensor(prop_samples).to(arguments.Config["general"]["device"])
            x_L, x_U = self.get_input_bounds(x, A)
            # if torch.all(x_L >= x_U):
            # print('check', x_L, '\n', x_U)                
            torch.manual_seed(arguments.Config["general"]["seed"])
            sample_num = arguments.Config["preimage"]["sample_num"]
            prop_samples = Uniform(x_L, x_U).sample([sample_num])
            # prop_samples = torch.transpose(prop_samples, 0, 1) 
            # prop_samples.requires_grad = x_L.requires_grad
            bound = None
            if not isinstance(A, eyeC):
                for i in range(prop_samples.shape[1]):
                    samples_tmp = prop_samples[:,i,:]
                    samples_tmp = torch.transpose(samples_tmp,0,1)
                    if bound is None:
                        bound = A[i].matmul(samples_tmp)+bias[i]
                    else:
                        bound_tmp = A[i].matmul(samples_tmp)+bias[i]
                        bound = torch.cat((bound, bound_tmp))
                # for i in range(A.shape[0]):# spec dimension
                #     samples_tmp = torch.transpose(prop_samples,0,1)

                #     if bound is None: # NOTE here should be A[:,i,:]? as the second is the spec dim
                #         bound = A[i].squeeze(1).matmul(samples_tmp)+bias[i]
                #     else:
                #         bound_tmp = A[i].matmul(samples_tmp)+bias[i]
                #         bound = torch.cat((bound, bound_tmp))
            else:
                assert extra_constr is None
                # A is an identity matrix. No need to do this matmul.
                # bound = center + sign * diff
                bound = prop_samples.unsqueeze(0)
        else:
            assert extra_constr is None
            x = x.reshape(x.shape[0], -1, 1)
            if not isinstance(A, eyeC):
                # Find the upper and lower bounds via dual norm.
                deviation = A.norm(self.dual_norm, -1) * self.eps
                bound = A.matmul(x) + sign * deviation.unsqueeze(-1)
            else:
                # A is an identity matrix. Its norm is all 1.
                bound = x + sign * self.eps
        # FIXME xy: why change it into 0       
        # bound = bound.squeeze(-1)
        # print('check bound shape', bound.shape)
        # bound = bound.squeeze(0)
        return bound
    
    # If A is an identity matrix, we will handle specially.
    def concretize_matrix_relu_poly(self, x, A, bias, sign, extra_constr, sample_left_idx, sample_right_idx, debug=False):
        if not isinstance(A, eyeC):
            # A has (Batch, spec, *input_size). For intermediate neurons, spec is *neuron_size.
            A = A.reshape(A.shape[0], A.shape[1], -1)
            # print('check bias shape', bias.shape)
            if extra_constr is not None:
                # For each neuron, we have a beta, so beta size is (Batch, *neuron_size, n_beta) (in A, spec is *neuron_size).
                # For intermediate layer neurons, A has *neuron_size specifications.
                beta = extra_constr['beta']
                beta = beta.view(beta.size(0), -1, beta.size(-1))
                # coeffs are linear relationships between split neurons and x. They have size (batch, n_beta, *input_size), and unreated to neuron_size.
                beta_coeffs = extra_constr['coeffs']
                beta_coeffs = beta_coeffs.view(beta_coeffs.size(0), beta_coeffs.size(1), -1)
                # biases are added for each batch each spec, size is (batch, n_beta), and unrelated to neuron_size.
                beta_bias = extra_constr['bias']
                # Merge beta into extra A and bias. Extra A has size (batch, spec, *input_size). For intermediate neurons, spec is *neuron_size.
                extra_A = torch.einsum('ijk,ikl->ijl', beta, beta_coeffs)
                # Merge beta into the bias term. Output has size (batch, spec).
                extra_bias = torch.einsum('ijk,ik->ij', beta, beta_bias)
        if self.norm == np.inf:
            # For Linfinity distortion, when an upper and lower bound is given, we use them instead of eps.
            if arguments.Config["model"]["onnx_path"] is None:
                sample_file = os.path.join(arguments.Config["preimage"]["sample_dir"], 'sample_{}_{}.pt'.format(arguments.Config["data"]["dataset"], arguments.Config["preimage"]["atk_tp"]))
                prop_samples = torch.load(sample_file)
                prop_samples = prop_samples.reshape(prop_samples.shape[0], -1)
                # sample_left_idx_file = os.path.join(arguments.Config["preimage"]["sample_dir"],'sample_left_{}_{}.pt'.format(arguments.Config["data"]["dataset"], arguments.Config["preimage"]["atk_tp"]))
                # sample_right_idx_file = os.path.join(arguments.Config["preimage"]["sample_dir"],'sample_right_{}_{}.pt'.format(arguments.Config["data"]["dataset"], arguments.Config["preimage"]["atk_tp"]))
            else:
                if arguments.Config["data"]["dataset"] == "vcas":
                    samples = np.load(os.path.join(arguments.Config["preimage"]["sample_dir"], "sample_{}_{}.npy".format(arguments.Config["data"]["dataset"], arguments.Config["preimage"]["upper_time_loss"])))
                else:
                    samples = np.load(os.path.join(arguments.Config["preimage"]["sample_dir"], "sample_{}.npy".format(arguments.Config["data"]["dataset"])))
                prop_samples = np.squeeze(samples, axis=1)
                prop_samples = torch.tensor(prop_samples).to(arguments.Config["general"]["device"])
                # sample_left_idx_file = os.path.join(arguments.Config["preimage"]["sample_dir"],'sample_left_{}.pt'.format(arguments.Config["data"]["dataset"]))
                # sample_right_idx_file = os.path.join(arguments.Config["preimage"]["sample_dir"],'sample_right_{}.pt'.format(arguments.Config["data"]["dataset"]))
            # if not os.path.exists(sample_left_idx_file):                
            #     acti_vecs = load_act_vecs(arguments.Config["data"]["dataset"])
            #     sample_left_idx = calc_history_idxs(acti_vecs, left_history)
            #     sample_right_idx = calc_history_idxs(acti_vecs, right_history)
            #     with open(sample_left_idx_file, 'wb') as f:
            #         pickle.dump(sample_left_idx, f)
            #     with open(sample_right_idx_file, 'wb') as f:
            #         pickle.dump(sample_right_idx, f)
            # else:
            #     with open(sample_left_idx_file, 'rb') as f:
            #         sample_left_idx = pickle.load(f)
            #     with open(sample_right_idx_file, 'rb') as f:
            #         sample_right_idx = pickle.load(f)
            bound = None
            if not isinstance(A, eyeC):
                # for i in range(prop_samples.shape[1]):
                # samples_tmp = prop_samples[:,0,:]
                # for i in range(A.shape[0]):
                if debug:
                    if (sample_left_idx is None) and (sample_right_idx is None):
                        # Test on one single sample
                        prop_samples = prop_samples[0].reshape(-1, prop_samples.shape[1])
                        samples_tmp = torch.transpose(prop_samples,0,1)
                        bound_tmp = A[0].matmul(samples_tmp)+bias[0]
                        bound = torch.transpose(bound_tmp, 0, 1)
                        bound = torch.sigmoid(bound)
                        # bound = torch.mean(bound)
                        bound = torch.mean(bound).reshape(-1, bound.shape[1])
                    else:
                        if len(sample_left_idx)>0:
                            samples_left = prop_samples[list(sample_left_idx)]
                            # Test on one single sample
                            samples_left = samples_left[0].reshape(-1, samples_left.shape[1])
                            samples_left = torch.transpose(samples_left,0,1)
                            bound_tmp = A[0].matmul(samples_left)+bias[0]
                            bound = torch.transpose(bound_tmp, 0, 1)
                            bound = torch.sigmoid(bound)
                            
                        if len(sample_right_idx)>0:
                            samples_right = prop_samples[list(sample_right_idx)]
                            samples_right = samples_right[0].reshape(-1, samples_right.shape[1])
                            samples_right = torch.transpose(samples_right,0,1)
                            if bound is None:
                                bound_tmp = A[1].matmul(samples_right)+bias[1]
                                bound = torch.transpose(bound_tmp, 0, 1)
                                bound = torch.sigmoid(bound)
                                bound = torch.mean(bound)
                            else:
                                bound_tmp = A[1].matmul(samples_right) + bias[1]
                                bound_tmp = torch.transpose(bound_tmp, 0, 1)
                                bound_tmp = torch.sigmoid(bound_tmp)
                                # bound_tmp = torch.mean(bound_tmp)
                                bound = torch.vstack((bound, bound_tmp))
                        if len(sample_left_idx) == 0 and len(sample_right_idx) == 0:
                            print("no left and no right samples")
                else:
                    if (sample_left_idx is None) and (sample_right_idx is None):
                        samples_tmp = torch.transpose(prop_samples,0,1)
                        bound_tmp = A[0].matmul(samples_tmp)+bias[0]
                        bound = torch.transpose(bound_tmp, 0, 1)
                    else:
                        if len(sample_left_idx)>0:
                            samples_left = prop_samples[list(sample_left_idx)]
                            samples_left = torch.transpose(samples_left,0,1)
                            bound_tmp = A[0].matmul(samples_left)+bias[0]
                            bound = torch.transpose(bound_tmp, 0, 1)
                        if len(sample_right_idx)>0:
                            samples_right = prop_samples[list(sample_right_idx)]
                            samples_right = torch.transpose(samples_right,0,1)
                            if bound is None:
                                bound_tmp = A[1].matmul(samples_right)+bias[1]
                                bound = torch.transpose(bound_tmp, 0, 1)
                            else:
                                bound_tmp = A[1].matmul(samples_right)+bias[1]
                                bound_tmp = torch.transpose(bound_tmp, 0, 1)
                                bound = torch.vstack((bound, bound_tmp))
                        if len(sample_left_idx) == 0 and len(sample_right_idx) == 0:
                            print("no left and no right samples")
            else:
                assert extra_constr is None
                # A is an identity matrix. No need to do this matmul.
                # bound = center + sign * diff
                bound = prop_samples.unsqueeze(0)
        else:
            assert extra_constr is None
            x = x.reshape(x.shape[0], -1, 1)
            if not isinstance(A, eyeC):
                # Find the upper and lower bounds via dual norm.
                deviation = A.norm(self.dual_norm, -1) * self.eps
                bound = A.matmul(x) + sign * deviation.unsqueeze(-1)
            else:
                # A is an identity matrix. Its norm is all 1.
                bound = x + sign * self.eps
        # FIXME xy: why change it into 0       
        # bound = bound.squeeze(-1)
        # print('check bound shape', bound.shape)
        # bound = bound.squeeze(0)
        if (sample_left_idx is None) and (sample_right_idx is None):
            return bound
        else:
            if debug:
                return bound
            else:
                return bound
            


    # If A is an identity matrix, we will handle specially.
    def concretize_matrix_poly_LSE(self, x, A, bias, sign, extra_constr, sample_num=1000):
        spec_num = arguments.Config["bab"]["initial_max_domains"]
        if not isinstance(A, eyeC):
            # A has (Batch, spec, *input_size). For intermediate neurons, spec is *neuron_size.
            A = A.reshape(A.shape[0], A.shape[1], -1)
            # print('check bias shape', bias.shape)
            if extra_constr is not None:
                # For each neuron, we have a beta, so beta size is (Batch, *neuron_size, n_beta) (in A, spec is *neuron_size).
                # For intermediate layer neurons, A has *neuron_size specifications.
                beta = extra_constr['beta']
                beta = beta.view(beta.size(0), -1, beta.size(-1))
                # coeffs are linear relationships between split neurons and x. They have size (batch, n_beta, *input_size), and unreated to neuron_size.
                beta_coeffs = extra_constr['coeffs']
                beta_coeffs = beta_coeffs.view(beta_coeffs.size(0), beta_coeffs.size(1), -1)
                # biases are added for each batch each spec, size is (batch, n_beta), and unrelated to neuron_size.
                beta_bias = extra_constr['bias']
                # Merge beta into extra A and bias. Extra A has size (batch, spec, *input_size). For intermediate neurons, spec is *neuron_size.
                extra_A = torch.einsum('ijk,ikl->ijl', beta, beta_coeffs)
                # Merge beta into the bias term. Output has size (batch, spec).
                extra_bias = torch.einsum('ijk,ik->ij', beta, beta_bias)
        if self.norm == np.inf:
            # For Linfinity distortion, when an upper and lower bound is given, we use them instead of eps.
            x_L, x_U = self.get_input_bounds(x, A)
            # x_ub = x_U.reshape(x_U.shape[0], -1, 1)
            # x_lb = x_L.reshape(x_L.shape[0], -1, 1)
            # Perform the sampling here so that we can optimize the alpha parameter based on the sample-based volume approx
            # This gives tensor of size (1000, input_dim)
            # prop_samples = None
            # input_dim = x_lb.shape[1]
            # for i in range(input_dim):
            #     if i == 0:
            #         prop_samples = Uniform(low=x_lb[0][i][0],high=x_ub[0][i][0]).sample([1, 1000])
            #     else:
            #         prop_features = Uniform(low=x_lb[0][i][0],high=x_ub[0][i][0]).sample([1, 1000])
            #         prop_samples = torch.cat((prop_samples, prop_features))
            torch.manual_seed(seed=1)
            # NOTE xy: the sample shape is (sample_num, dom_num, input_dim)
            group_num = int(x_L.shape[0]/spec_num)
            idx_list = torch.tensor([i*spec_num for i in range(group_num)])
            prop_samples = Uniform(x_L[idx_list], x_U[idx_list]).sample([sample_num])
            # NOTE: xy: only need linear map and correponding domain

            assert group_num == prop_samples.shape[1]
            # prop_samples = torch.transpose(prop_samples, 0, 1) 
            # prop_samples = prop_samples.reshape(x_L.shape[0],-1,sample_num)
            # prop_samples = autograd.Variable(prop_samples)
            prop_samples.requires_grad = x_L.requires_grad
            final_bound = None
            if not isinstance(A, eyeC):
                # if extra_constr is not None:
                #     # Extra linear and bias terms from constraints.
                #     print(
                #         f'A extra: {(sign * extra_A).abs().sum().item()}, '
                #         f'b extra: {(sign * extra_bias).abs().sum().item()}')
                #     A = A - sign * extra_A
                #     bound = A.matmul(center) - sign * extra_bias.unsqueeze(-1) + sign * A.abs().matmul(diff)
                # else:
                # for i in range(prop_samples.shape[1]):
                for i in range(group_num):
                    bound = None
                    samples_tmp = prop_samples[:,i,:]
                    samples_tmp = torch.transpose(samples_tmp,0,1)
                    for j in range(spec_num):
                        if bound is None:
                            bound = A[i*spec_num+j].matmul(samples_tmp)+bias[i*spec_num+j]
                        else:
                            bound_tmp = A[i*spec_num+j].matmul(samples_tmp)+bias[i*spec_num+j]
                            bound = torch.cat((bound, bound_tmp))
                    bound = -torch.logsumexp(-bound, dim=0)
                    bound = torch.sum(torch.sigmoid(bound))
                    if final_bound is None:
                        # final_bound = bound.repeat(spec_num, 1)
                        final_bound = bound.unsqueeze(0)
                    else:
                        # bound = bound.repeat(spec_num, 1)
                        bound = bound.unsqueeze(0)
                        final_bound = torch.cat((final_bound, bound))
            else:
                assert extra_constr is None
                # A is an identity matrix. No need to do this matmul.
                # bound = center + sign * diff
                bound = prop_samples.unsqueeze(0)
        else:
            assert extra_constr is None
            x = x.reshape(x.shape[0], -1, 1)
            if not isinstance(A, eyeC):
                # Find the upper and lower bounds via dual norm.
                deviation = A.norm(self.dual_norm, -1) * self.eps
                bound = A.matmul(x) + sign * deviation.unsqueeze(-1)
            else:
                # A is an identity matrix. Its norm is all 1.
                bound = x + sign * self.eps
        # FIXME xy: why change it into 0       
        # bound = bound.squeeze(-1)
        # print('check bound shape', bound.shape)
        # bound = bound.squeeze(0)
        return final_bound
    def concretize_patches(self, x, A, sign, extra_constr):
        if self.norm == np.inf:
            x_L, x_U = self.get_input_bounds(x, A)

            # Here we should not reshape
            # Find the uppwer and lower bound similarly to IBP.
            center = (x_U + x_L) / 2.0
            diff = (x_U - x_L) / 2.0

            if not A.identity == 1:
                bound = A.matmul(center)
                bound_diff = A.matmul(diff, patch_abs=True)

                if sign == 1:
                    bound += bound_diff
                elif sign == -1:
                    bound -= bound_diff
                else:
                    raise ValueError("Unsupported Sign")

                # The extra bias term from beta term.
                if extra_constr is not None:
                    bound += extra_constr
            else:
                assert extra_constr is None
                # A is an identity matrix. No need to do this matmul.
                bound = center + sign * diff
            return bound
        else:  # Lp norm
            input_shape = x.shape
            if not A.identity:
                # Find the upper and lower bounds via dual norm.
                # matrix has shape (batch_size, out_c * out_h * out_w, input_c, input_h, input_w) or (batch_size, unstable_size, input_c, input_h, input_w)
                matrix = patches_to_matrix(A.patches, input_shape, A.stride, A.padding, A.output_shape, A.unstable_idx)
                # Note that we should avoid reshape the matrix. Due to padding, matrix cannot be reshaped without copying.
                deviation = matrix.norm(p=self.dual_norm, dim=(-3,-2,-1)) * self.eps
                # Bound has shape (batch, out_c * out_h * out_w) or (batch, unstable_size).
                bound = torch.einsum('bschw,bchw->bs', matrix, x) + sign * deviation
                if A.unstable_idx is None:
                    # Reshape to (batch, out_c, out_h, out_w).
                    bound = bound.view(matrix.size(0), A.patches.size(0), A.patches.size(2), A.patches.size(3))
            else:
                # A is an identity matrix. Its norm is all 1.
                bound = x + sign * self.eps
            return bound
    """Given an variable x and its bound matrix A, compute approximated volume based on samples and continuous relaxation."""
    def concretize_poly_vol(self, x, A, bias, sign=-1, aux=None, extra_constr=None):
        if A is None:
            return None
        if isinstance(A, eyeC) or isinstance(A, torch.Tensor):
            return self.concretize_matrix_poly(x, A, bias, sign, extra_constr)
        elif isinstance(A, Patches):
            return self.concretize_patches(x, A, sign, extra_constr)
        else:
            raise NotImplementedError()
        
    def concretize_relu_poly_vol(self, x, A, bias, sign=-1, aux=None, extra_constr=None, sample_left_idx=None, sample_right_idx=None):
        if A is None:
            return None
        if isinstance(A, eyeC) or isinstance(A, torch.Tensor):
            return self.concretize_matrix_relu_poly(x, A, bias, sign, extra_constr, sample_left_idx, sample_right_idx)
        elif isinstance(A, Patches):
            return self.concretize_patches(x, A, sign, extra_constr)
        else:
            raise NotImplementedError()
    """Given an variable x and its bound matrix A, compute approximated volume based on samples and continuous relaxation."""
    def concretize_poly_vol_LSE(self, x, A, bias, sign=-1, aux=None, extra_constr=None):
        if A is None:
            return None
        if isinstance(A, eyeC) or isinstance(A, torch.Tensor):
            return self.concretize_matrix_poly_LSE(x, A, bias, sign, extra_constr)
        elif isinstance(A, Patches):
            return self.concretize_patches(x, A, sign, extra_constr)
        else:
            raise NotImplementedError()
    """Given an variable x and its bound matrix A, compute worst case bound according to Lp norm."""
    def concretize(self, x, A, sign=-1, aux=None, extra_constr=None):
        if A is None:
            return None
        if isinstance(A, eyeC) or isinstance(A, torch.Tensor):
            return self.concretize_matrix(x, A, sign, extra_constr)
        elif isinstance(A, Patches):
            return self.concretize_patches(x, A, sign, extra_constr)
        else:
            raise NotImplementedError()

    """ Sparse Linf perturbation where only a few dimensions are actually perturbed"""
    def init_sparse_linf(self, x, x_L, x_U):
        self.sparse = True
        batch_size = x_L.shape[0]
        perturbed = (x_U > x_L).int()
        logger.debug(f'Perturbed: {perturbed.sum()}')
        lb = ub = x_L * (1 - perturbed) # x_L=x_U holds when perturbed=0
        perturbed = perturbed.view(batch_size, -1)
        index = torch.cumsum(perturbed, dim=-1)
        dim = max(perturbed.view(batch_size, -1).sum(dim=-1).max(), 1)
        self.x_L_sparse = torch.zeros(batch_size, dim + 1).to(x_L)
        self.x_L_sparse.scatter_(dim=-1, index=index, src=(x_L - lb).view(batch_size, -1), reduce='add')
        self.x_U_sparse = torch.zeros(batch_size, dim + 1).to(x_U)
        self.x_U_sparse.scatter_(dim=-1, index=index, src=(x_U - ub).view(batch_size, -1), reduce='add')
        self.x_L_sparse, self.x_U_sparse = self.x_L_sparse[:, 1:], self.x_U_sparse[:, 1:]
        lw = torch.zeros(batch_size, dim + 1, perturbed.shape[-1], device=x.device)
        perturbed = perturbed.to(torch.get_default_dtype())
        lw.scatter_(dim=1, index=index.unsqueeze(1), src=perturbed.unsqueeze(1))
        lw = uw = lw[:, 1:, :].view(batch_size, dim, *x.shape[1:])
        print(f'Using Linf sparse perturbation. Perturbed dimensions: {dim}.')
        print(f'Avg perturbation: {(self.x_U_sparse - self.x_L_sparse).mean()}')
        return LinearBound(
            lw, lb, uw, ub, x_L, x_U), x, None

    def init(self, x, aux=None, forward=False):
        self.sparse = False
        if self.norm == np.inf:
            x_L = x - self.eps if self.x_L is None else self.x_L
            x_U = x + self.eps if self.x_U is None else self.x_U
        else:
            # For other norms, we pass in the BoundedTensor objects directly.
            x_L = x_U = x
        if not forward:
            return LinearBound(
                None, None, None, None, x_L, x_U), x, None
        if self.norm == np.inf and x_L.numel() > 1 and (x_L == x_U).sum() > 0.5 * x_L.numel():
            return self.init_sparse_linf(x, x_L, x_U)

        batch_size = x.shape[0]
        dim = x.reshape(batch_size, -1).shape[-1]
        lb = ub = torch.zeros_like(x)
        eye = torch.eye(dim).to(x).expand(batch_size, dim, dim)
        lw = uw = eye.reshape(batch_size, dim, *x.shape[1:])
        return LinearBound(
            lw, lb, uw, ub, x_L, x_U), x, None

    def __repr__(self):
        if self.norm == np.inf:
            if self.x_L is None and self.x_U is None:
                return 'PerturbationLpNorm(norm=inf, eps={})'.format(self.eps)
            else:
                return 'PerturbationLpNorm(norm=inf, eps={}, x_L={}, x_U={})'.format(self.eps, self.x_L, self.x_U)
        else:
            return 'PerturbationLpNorm(norm={}, eps={})'.format(self.norm, self.eps)

class PerturbationSynonym(Perturbation):
    def __init__(self, budget, eps=1.0, use_simple=False):
        super(PerturbationSynonym, self).__init__()
        self._load_synonyms()
        self.budget = budget
        self.eps = eps
        self.use_simple = use_simple
        self.model = None
        self.train = False

    def __repr__(self):
        return 'perturbation(Synonym-based word substitution budget={}, eps={})'.format(
            self.budget, self.eps)

    def _load_synonyms(self, path='data/synonyms.json'):
        with open(path) as file:
            self.synonym = json.loads(file.read())
        logger.info('Synonym list loaded for {} words'.format(len(self.synonym)))

    def set_train(self, train):
        self.train = train

    def concretize(self, x, A, sign, aux):
        assert(self.model is not None)

        x_rep, mask, can_be_replaced = aux
        batch_size, length, dim_word = x.shape[0], x.shape[1], x.shape[2]
        dim_out = A.shape[1]
        max_num_cand = x_rep.shape[2]

        mask_rep = torch.tensor(can_be_replaced, dtype=torch.float32, device=A.device)

        num_pos = int(np.max(np.sum(can_be_replaced, axis=-1)))
        update_A = A.shape[-1] > num_pos * dim_word
        if update_A:
            bias = torch.bmm(A, (x * (1 - mask_rep).unsqueeze(-1)).reshape(batch_size, -1, 1)).squeeze(-1)
        else:
            bias = 0.
        A = A.reshape(batch_size, dim_out, -1, dim_word)

        A_new, x_new, x_rep_new, mask_new = [], [], [], []
        zeros_A = torch.zeros(dim_out, dim_word, device=A.device)
        zeros_w = torch.zeros(dim_word, device=A.device)
        zeros_rep = torch.zeros(max_num_cand, dim_word, device=A.device)
        zeros_mask = torch.zeros(max_num_cand, device=A.device)
        for t in range(batch_size):
            cnt = 0
            for i in range(0, length):
                if can_be_replaced[t][i]:
                    if update_A:
                        A_new.append(A[t, :, i, :])
                    x_new.append(x[t][i])
                    x_rep_new.append(x_rep[t][i])
                    mask_new.append(mask[t][i])
                    cnt += 1
            if update_A:
                A_new += [zeros_A] * (num_pos - cnt)
            x_new += [zeros_w] * (num_pos - cnt)
            x_rep_new += [zeros_rep] * (num_pos - cnt)
            mask_new += [zeros_mask] * (num_pos - cnt)
        if update_A:
            A = torch.cat(A_new).reshape(batch_size, num_pos, dim_out, dim_word).transpose(1, 2)
        x = torch.cat(x_new).reshape(batch_size, num_pos, dim_word)
        x_rep = torch.cat(x_rep_new).reshape(batch_size, num_pos, max_num_cand, dim_word)
        mask = torch.cat(mask_new).reshape(batch_size, num_pos, max_num_cand)
        length = num_pos

        A = A.reshape(batch_size, A.shape[1], length, -1).transpose(1, 2)
        x = x.reshape(batch_size, length, -1, 1)

        if sign == 1:
            cmp, init = torch.max, -1e30
        else:
            cmp, init = torch.min, 1e30

        init_tensor = torch.ones(batch_size, dim_out).to(x.device) * init
        dp = [[init_tensor] * (self.budget + 1) for i in range(0, length + 1)]
        dp[0][0] = torch.zeros(batch_size, dim_out).to(x.device)

        A = A.reshape(batch_size * length, A.shape[2], A.shape[3])
        Ax = torch.bmm(
            A,
            x.reshape(batch_size * length, x.shape[2], x.shape[3])
        ).reshape(batch_size, length, A.shape[1])

        Ax_rep = torch.bmm(
            A,
            x_rep.reshape(batch_size * length, max_num_cand, x.shape[2]).transpose(-1, -2)
        ).reshape(batch_size, length, A.shape[1], max_num_cand)
        Ax_rep = Ax_rep * mask.unsqueeze(2) + init * (1 - mask).unsqueeze(2)
        Ax_rep_bound = cmp(Ax_rep, dim=-1).values

        if self.use_simple and self.train:
            return torch.sum(cmp(Ax, Ax_rep_bound), dim=1) + bias

        for i in range(1, length + 1):
            dp[i][0] = dp[i - 1][0] + Ax[:, i - 1]
            for j in range(1, self.budget + 1):
                dp[i][j] = cmp(
                    dp[i - 1][j] + Ax[:, i - 1],
                    dp[i - 1][j - 1] + Ax_rep_bound[:, i - 1]
                )
        dp = torch.cat(dp[length], dim=0).reshape(self.budget + 1, batch_size, dim_out)

        return cmp(dp, dim=0).values + bias

    def init(self, x, aux=None, forward=False):
        tokens, batch = aux
        self.tokens = tokens # DEBUG
        assert(len(x.shape) == 3)
        batch_size, length, dim_word = x.shape[0], x.shape[1], x.shape[2]

        max_pos = 1
        can_be_replaced = np.zeros((batch_size, length), dtype=np.bool)

        self._build_substitution(batch)

        for t in range(batch_size):
            cnt = 0
            candidates = batch[t]['candidates']
            # for transformers
            if tokens[t][0] == '[CLS]':
                candidates = [[]] + candidates + [[]]
            for i in range(len(tokens[t])):
                if tokens[t][i] == '[UNK]' or \
                        len(candidates[i]) == 0 or tokens[t][i] != candidates[i][0]:
                    continue
                for w in candidates[i][1:]:
                    if w in self.model.vocab:
                        can_be_replaced[t][i] = True
                        cnt += 1
                        break
            max_pos = max(max_pos, cnt)

        dim = max_pos * dim_word
        if forward:
            eye = torch.eye(dim_word).to(x.device)
            lw = torch.zeros(batch_size, dim, length, dim_word).to(x.device)
            lb = torch.zeros_like(x).to(x.device)
        x_new = []
        word_embeddings = self.model.word_embeddings.weight
        vocab = self.model.vocab
        x_rep = [[[] for i in range(length)] for t in range(batch_size)]
        max_num_cand = 1
        for t in range(batch_size):
            candidates = batch[t]['candidates']
            # for transformers
            if tokens[t][0] == '[CLS]':
                candidates = [[]] + candidates + [[]]
            cnt = 0
            for i in range(length):
                if can_be_replaced[t][i]:
                    word_embed = word_embeddings[vocab[tokens[t][i]]]
                    # positional embedding and token type embedding
                    other_embed = x[t, i] - word_embed
                    if forward:
                        lw[t, (cnt * dim_word):((cnt + 1) * dim_word), i, :] = eye
                        lb[t, i, :] = torch.zeros_like(word_embed)
                    for w in candidates[i][1:]:
                        if w in self.model.vocab:
                            x_rep[t][i].append(
                                word_embeddings[self.model.vocab[w]] + other_embed)
                    max_num_cand = max(max_num_cand, len(x_rep[t][i]))
                    cnt += 1
                else:
                    if forward:
                        lb[t, i, :] = x[t, i, :]
        if forward:
            uw, ub = lw, lb
        else:
            lw = lb = uw = ub = None
        zeros = torch.zeros(dim_word, device=x.device)

        x_rep_, mask = [], []
        for t in range(batch_size):
            for i in range(length):
                x_rep_ += x_rep[t][i] + [zeros] * (max_num_cand - len(x_rep[t][i]))
                mask += [1] * len(x_rep[t][i]) + [0] * (max_num_cand - len(x_rep[t][i]))
        x_rep_ = torch.cat(x_rep_).reshape(batch_size, length, max_num_cand, dim_word)
        mask = torch.tensor(mask, dtype=torch.float32, device=x.device)\
            .reshape(batch_size, length, max_num_cand)
        x_rep_ = x_rep_ * self.eps + x.unsqueeze(2) * (1 - self.eps)

        inf = 1e20
        lower = torch.min(mask.unsqueeze(-1) * x_rep_ + (1 - mask).unsqueeze(-1) * inf, dim=2).values
        upper = torch.max(mask.unsqueeze(-1) * x_rep_ + (1 - mask).unsqueeze(-1) * (-inf), dim=2).values
        lower = torch.min(lower, x)
        upper = torch.max(upper, x)

        return LinearBound(lw, lb, uw, ub, lower, upper), x, (x_rep_, mask, can_be_replaced)

    def _build_substitution(self, batch):
        for t, example in enumerate(batch):
            if not 'candidates' in example or example['candidates'] is None:
                candidates = []
                tokens = example['sentence'].strip().lower().split(' ')
                for i in range(len(tokens)):
                    _cand = []
                    if tokens[i] in self.synonym:
                        for w in self.synonym[tokens[i]]:
                            if w in self.model.vocab:
                                _cand.append(w)
                    if len(_cand) > 0:
                        _cand = [tokens[i]] + _cand
                    candidates.append(_cand)
                example['candidates'] = candidates
