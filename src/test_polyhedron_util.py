import torch
from torch.distributions import Uniform
import pickle
import os
import numpy as np
from collections import defaultdict
# from test_parse_args import get_args
from preimage_model_utils import load_input_bounds_numpy, load_model_simple
from preimage_model_utils import load_input_bounds
# NOTE for adding the arguments module
import arguments
from utils import load_model, load_verification_dataset
from preimage_model_utils import build_model_activation
RES_DIR = '/home/xiyue/LinInv/test_res'

def load_A_dict(A_file):
    with open(A_file, 'rb') as f:
        linear_rep_dict = pickle.load(f)
    return linear_rep_dict


def save_A_dict(A, A_path, A_dir=RES_DIR):
    # A_file= os.path.join(RES_DIR, 'A_dict_BaB_init')
    A_file= os.path.join(A_dir, A_path)
    with open(A_file, 'wb') as f:
        pickle.dump(A, f)
        
def post_process_A(A_dict):
    linear_rep_dict = dict()
    for output_node, linear_rep in A_dict.items():
        for input_node, param_dict in linear_rep.items():
            for k, v in param_dict.items():
                if v is not None: 
                    if k == 'lA' or k == 'uA':
                        if "MNIST" in arguments.Config["data"]["dataset"]:
                            v = v.reshape(v.shape[0], v.shape[1], -1)
                            linear_rep_dict[k] = torch.squeeze(v,0).cpu().detach().numpy()
                        else:
                            linear_rep_dict[k] = torch.squeeze(v,0).cpu().detach().numpy()
                    elif k == 'lbias' or k == 'ubias':
                        # NOTE: we dont squeeze bias as it can help with the batch addition
                        linear_rep_dict[k] = v.cpu().detach().numpy()
    return linear_rep_dict

def post_process_multiple_A(A_dict):
    # linear_rep_dict_multi.setdefault()
    linear_rep_dict_multi = defaultdict(list)
    for output_node, linear_rep in A_dict.items():
        for input_node, param_dict in linear_rep.items():
            for k, v in param_dict.items():
                if v is not None:
                    if k == 'lA' or k == 'uA':
                        if "MNIST" in arguments.Config["data"]["dataset"]:
                            v = v.reshape(v.shape[0],v.shape[1],-1)
                            for i in range(v.shape[0]):
                                linear_rep_dict_multi[k].append(v[i].cpu().detach().numpy())      
                        else:              
                            for i in range(v.shape[0]):
                                linear_rep_dict_multi[k].append(v[i].cpu().detach().numpy())
                    elif k == 'lbias' or k == 'ubias':
                        for i in range(v.shape[0]):
                            bias = v[i].cpu().detach().numpy()
                            linear_rep_dict_multi[k].append(bias)
    # print(linear_rep_dict_multi)
    return linear_rep_dict_multi
def post_process_greedy_A(A_dict):
    linear_rep_dict_multi = dict()
    for output_node, linear_rep in A_dict.items():
        for input_node, param_dict in linear_rep.items():
            for k, v in param_dict.items():
                if v is not None:
                    if k == 'lA' or k == 'uA':
                        if "MNIST" in arguments.Config["data"]["dataset"]:
                            v = v.reshape(v.shape[0],v.shape[1],-1)
                            linear_rep_dict_multi[k] = v.cpu().detach().numpy()      
                        else:              
                            linear_rep_dict_multi[k] = v.cpu().detach().numpy()
                    elif k == 'lbias' or k == 'ubias':
                        linear_rep_dict_multi[k] = v.cpu().detach().numpy()
    return linear_rep_dict_multi
def post_process_A_dict_relu(A_dict_relus):
    linear_lb_ub_dict = [defaultdict(list) for _ in range(len(A_dict_relus))]
    # for relu_node, linear_rep in A_dict_relu.items():
    for layer, A_dict_relu in enumerate(A_dict_relus):
        for input_node, param_dict in A_dict_relu.items():
            for k, v in param_dict.items():
                if k == 'lA' or k == 'uA':
                    if "MNIST" in arguments.Config["data"]["dataset"]:
                        v = v.reshape(v.shape[0],v.shape[1],-1)
                        for i in range(v.shape[0]):
                            linear_lb_ub_dict[layer][k].append(v[i].cpu().detach().numpy())                  
                    else:                  
                        for i in range(v.shape[0]):
                            linear_lb_ub_dict[layer][k].append(torch.squeeze(v[i]).cpu().detach().numpy())
                elif k == 'lbias' or k == 'ubias':
                    for i in range(v.shape[0]):
                        # bias = v[i].cpu().detach().numpy()
                        linear_lb_ub_dict[layer][k].append(v[i].cpu().detach().numpy())
                else:
                    continue
    return linear_lb_ub_dict
                                         
# def save_linear_rep_dict(A, A_path, args):
#     linear_rep_dict = post_process_A(A)
#     # if args is not None:
#     #     save_path = '{}_{}_{}_label_{}'.format(args.model_name, args.dataset, args.base, args.label)
#     # save_path = os.path.join(args.result_dir, save_path)
#     with open(A_path, "wb") as f:
#         pickle.dump(linear_rep_dict, f)
#     return linear_rep_dict

# def calc_refine_under_approx_coverage(A, dm_l_all, dm_u_all, spec_dim):
#     # dm_l_all_np = dm_l_all.cpu().detach().numpy()
#     # dm_u_all_np = dm_u_all.cpu().detach().numpy()
#     # print(dm_l_all_np, dm_u_all_np)
#     A_b_dict_multi = post_process_multiple_A(A)
#     # now changed it to pairwise one, for preset branching depth, use multi
#     # cov_quota = calc_Hrep_coverage_multi(A_b_dict_multi, dm_l_all_np, dm_u_all_np)
#     # List of list (3 elements)
#     cov_input_idx_all = calc_Hrep_coverage_multi_spec_pairwise(A_b_dict_multi, dm_l_all, dm_u_all, spec_dim)
#     return cov_input_idx_all, A_b_dict_multi

# Calculate the subregion volume    
def calc_total_sub_vol(dm_l, dm_u):
    total_sub_vol = 1
    in_dim = dm_l.shape[-1]
    dm_shape = dm_l.shape
    # print("check dm_l, dm_u shape", dm_shape)
    assert len(dm_shape) == 2 or len(dm_shape) == 1
    if len(dm_shape) == 2:
        dm_diff = dm_u[0] - dm_l[0]
        if arguments.Config["data"]["dataset"] == "vcas":
            for i in range(in_dim):
                # if i != 2:
                total_sub_vol = total_sub_vol * dm_diff[i]
        else:            
            for i in range(in_dim):
                if dm_diff[i] > 1e-6:
                    total_sub_vol = total_sub_vol * dm_diff[i]
    elif len(dm_shape) == 1:
        dm_diff = dm_u - dm_l
        if arguments.Config["data"]["dataset"] == "vcas":
            for i in range(in_dim):
                # if i != 2:
                total_sub_vol = total_sub_vol * dm_diff[i]
        else:
            for i in range(in_dim):
                if dm_diff[i] > 1e-6:
                    total_sub_vol = total_sub_vol * dm_diff[i]
    return total_sub_vol

def calc_input_coverage_initial_image_under(A_b_dict, label):
    dataset_tp = arguments.Config["data"]["dataset"]    
    sample_num = arguments.Config["preimage"]["sample_num"]
    sample_dir = arguments.Config["preimage"]["sample_dir"]
    atk_tp = arguments.Config["preimage"]["atk_tp"]
    upper_time_loss = arguments.Config["preimage"]["upper_time_loss"]
    if arguments.Config["specification"]["type"] == "bound":
        if arguments.Config["model"]["onnx_path"] is None: 
            samples = torch.load(os.path.join(sample_dir, 'sample_{}_{}.pt'.format(dataset_tp, atk_tp)))
        else:
            if arguments.Config["data"]["dataset"] == "vcas":
                samples = np.load(os.path.join(sample_dir, "sample_{}_{}.npy".format(dataset_tp, upper_time_loss)))
            else:
                samples = np.load(os.path.join(sample_dir, f"sample_{dataset_tp}.npy"))
            samples = np.squeeze(samples, axis=1)
    elif arguments.Config["specification"]["type"] == 'lp':
        samples = torch.load(os.path.join(sample_dir, 'sample_{}_{}.pt'.format(dataset_tp, atk_tp)))
    if arguments.Config["model"]["onnx_path"] is None:
        if "Customized" in arguments.Config["data"]["dataset"]:  
            model = load_model(weights_loaded=False)
        else:
            model = load_model()
        device = arguments.Config["general"]["device"]
        model = model.to(device)
        predicted = model(samples).argmax(dim=1).cpu().detach().numpy()
    else:
        if dataset_tp == "vcas":
            predicted = np.load(os.path.join(sample_dir, f"pred_output_{dataset_tp}_{upper_time_loss}.npy"))
            predicted = predicted.argmax(axis=1)
        else:
            predicted = np.load(os.path.join(sample_dir, f"pred_output_{dataset_tp}.npy"))
            predicted = np.squeeze(predicted, axis=1).argmax(axis=1)
    idxs = np.where(predicted==label)[0]
    if len(idxs)>0:
        target_vol_portion = len(idxs)/sample_num
        print('Label: {}, Num: {}'.format(label, len(idxs)))
        if arguments.Config["model"]["onnx_path"] is None: 
            samples = samples.reshape(samples.shape[0], -1)
            samples_tmp = samples[idxs]
            samples_tmp = samples_tmp.cpu().detach().numpy()
        else:
            samples_tmp = samples[idxs]
        mat = A_b_dict["lA"]
        bias = A_b_dict["lbias"]
        # print('mat shape: {}, sample_tmp_T shape: {}'.format(mat.shape, samples_tmp.T.shape))    
        result = np.matmul(mat, samples_tmp.T)+bias
        spec_dim = result.shape[0]
        if spec_dim > 1:
            idxs_True = None
            for i in range(spec_dim):
                idxs_tmp = set(np.where(result[i]>=0)[0])
                if idxs_True is None:
                    idxs_True = idxs_tmp
                else:
                    idxs_True = idxs_True.intersection(idxs_tmp)
        else:
            idxs_True = np.where(result>=0)[0]
        cov_quota = len(idxs_True)/len(idxs)
        print("Coverage quota {}/{}:  {:.3f}".format(len(idxs_True), len(idxs), cov_quota))
    else:
        target_vol_portion = 0
        cov_quota = 0
    return target_vol_portion, cov_quota

    
def calc_input_coverage_initial_image_over(A_b_dict, label):
    dataset_tp = arguments.Config["data"]["dataset"]    
    sample_num = arguments.Config["preimage"]["sample_num"]
    sample_dir = arguments.Config["preimage"]["sample_dir"]
    atk_tp = arguments.Config["preimage"]["atk_tp"]
    # upper_time_loss = arguments.Config["preimage"]["upper_time_loss"]
    if arguments.Config["specification"]["type"] == "bound":
        if arguments.Config["model"]["onnx_path"] is None: 
            samples = torch.load(os.path.join(sample_dir, 'sample_{}_{}.pt'.format(dataset_tp, atk_tp)))
        else:
            samples = np.load(os.path.join(sample_dir, "{}/sample_{}.npy".format(dataset_tp, dataset_tp)))
            # if arguments.Config["data"]["dataset"] == "vcas":
            #     samples = np.load(os.path.join(sample_dir, "{}/sample_{}.npy".format(dataset_tp, dataset_tp)))
            # else:
            #     samples = np.load(os.path.join(sample_dir, f"sample_{dataset_tp}.npy"))
            samples = np.squeeze(samples, axis=1)
    elif arguments.Config["specification"]["type"] == 'lp':
        samples = torch.load(os.path.join(sample_dir, 'sample_{}_{}.pt'.format(dataset_tp, atk_tp)))
    if arguments.Config["model"]["onnx_path"] is None:
        if "Customized" in arguments.Config["data"]["dataset"]:  
            model = load_model(weights_loaded=False)
        else:
            model = load_model()
        device = arguments.Config["general"]["device"]
        model = model.to(device)
        predicted = model(samples).argmax(dim=1).cpu().detach().numpy()
    else:
        if dataset_tp == "vcas":
            predicted = np.load(os.path.join(sample_dir, f"{dataset_tp}/pred_output_{dataset_tp}.npy"))
            predicted = predicted.argmax(axis=1)
        elif dataset_tp == 'dubinsrejoin':
            predicted = np.load(os.path.join(sample_dir, f"{dataset_tp}/pred_output_{dataset_tp}.npy"))
            if label == 0:
                predicted = predicted[:, :4]
            elif label == 4:
                predicted = predicted[:, 4:]
            predicted = predicted.argmax(axis=1)
        else:
            predicted = np.load(os.path.join(sample_dir, f"{dataset_tp}/pred_output_{dataset_tp}.npy"))
            predicted = np.squeeze(predicted, axis=1).argmax(axis=1)
    idxs = np.where(predicted==label)[0]
    if len(idxs)>0:
        target_vol_portion = len(idxs)/sample_num
        print('Label: {}, Num: {}'.format(label, len(idxs)))
        # if arguments.Config["model"]["onnx_path"] is None: 
        #     samples = samples.reshape(samples.shape[0], -1)
        #     samples_tmp = samples[idxs]
        #     samples_tmp = samples_tmp.cpu().detach().numpy()
        # else:
        #     samples_tmp = samples[idxs]
        samples = samples.reshape(samples.shape[0], -1)
        if arguments.Config["model"]["onnx_path"] is None:
            samples_tmp = samples.cpu().detach().numpy()
        else:
            samples_tmp = samples
        mat = A_b_dict["uA"]
        bias = A_b_dict["ubias"]
        # print('mat shape: {}, sample_tmp_T shape: {}'.format(mat.shape, samples_tmp.T.shape))    
        result = np.matmul(mat, samples_tmp.T)+bias
        spec_dim = result.shape[0]
        if spec_dim > 1:
            idxs_True = None
            for i in range(spec_dim):
                idxs_tmp = set(np.where(result[i]>=0)[0])
                if idxs_True is None:
                    idxs_True = idxs_tmp
                else:
                    idxs_True = idxs_True.intersection(idxs_tmp)
        else:
            idxs_True = np.where(result>=0)[0]
        cov_quota = len(idxs_True)/len(idxs)
        print("Coverage quota {}/{}:  {:.3f}".format(len(idxs_True), len(idxs), cov_quota))
    else:
        target_vol_portion = 0
        cov_quota = 0
    return target_vol_portion, cov_quota

def calc_input_coverage_initial(A_b_dict, label):
    dataset_tp = arguments.Config["data"]["dataset"]    
    sample_num = arguments.Config["preimage"]["sample_num"]
    if arguments.Config["specification"]["type"] == "bound":
        samples = torch.load(os.path.join(arguments.Config['preimage']["sample_dir"], 'sample_{}.pt'.format(dataset_tp)))
    elif arguments.Config["specification"]["type"] == 'lp':
        X, labels, runnerup, data_max, data_min, perturb_epsilon, target_label = load_verification_dataset(eps_before_normalization=None)
        # img_shape = X[0].cpu().detach().numpy().shape
        img_shape = [-1] + list(X.shape[1:])
        x_lower = (X[0] - perturb_epsilon).clamp(min=data_min).flatten(1)
        x_upper = (X[0] + perturb_epsilon).clamp(max=data_max).flatten(1)
        data_ub = torch.squeeze(x_upper).cpu().detach().numpy()
        data_lb = torch.squeeze(x_lower).cpu().detach().numpy()
        label = labels.cpu().detach().numpy()[0]
    # dm_vol = calc_total_sub_vol(data_lb, data_ub)
    # samples = np.random.uniform(low=data_lb, high=data_ub, size=(sample_num, len(data_lb)))
    if "Customized" in dataset_tp:
        model = load_model(weights_loaded=False)
    else:
        model = load_model()
    device = arguments.Config["general"]["device"]
    model = model.to(device)
    predicted = model(samples).argmax(dim=1).cpu().detach().numpy()
    idxs = np.where(predicted==label)[0]
    if len(idxs)>0:
        target_vol = len(idxs)/sample_num
        print('Label: {}, Num: {}'.format(label, len(idxs)))
        samples_tmp = samples[idxs]
        samples_tmp = samples_tmp.cpu().detach().numpy()
        mat = A_b_dict["lA"]
        bias = A_b_dict["lbias"]
        # print('mat shape: {}, sample_tmp_T shape: {}'.format(mat.shape, samples_tmp.T.shape))    
        result = np.matmul(mat, samples_tmp.T)+bias
        spec_dim = result.shape[0]
        if spec_dim > 1:
            idxs_True = None
            for i in range(spec_dim):
                idxs_tmp = set(np.where(result[i]>=0)[0])
                if idxs_True is None:
                    idxs_True = idxs_tmp
                else:
                    idxs_True = idxs_True.intersection(idxs_tmp)
        else:
            idxs_True = np.where(result>=0)[0]
        cov_quota = len(idxs_True)/len(idxs)
        print("Coverage quota {}/{}:  {:.3f}".format(len(idxs_True), len(idxs), cov_quota))
    else:
        target_vol = 0
        cov_quota = 0
    return target_vol, cov_quota
def calc_pos_neg_indices(dm_history, split_relu_indices):
    layers_pos_neg = [[[],[]] for i in range(len(split_relu_indices))]
    for i, relu_idx in enumerate(split_relu_indices):
        pos_neg_list = layers_pos_neg[i]
        for j, neuron_idx in enumerate(dm_history[relu_idx][0]):
            neuron_sign = dm_history[relu_idx][1][j]
            if neuron_sign == 1:
                pos_neg_list[0].append(neuron_idx)
            elif neuron_sign == -1:
                pos_neg_list[1].append(neuron_idx)
            else:
                print("neuron sign assignment error")
    return layers_pos_neg

def calc_extra_A_b(A_dict_relus, dm_history, split_relu_indices, indicator=0):
    layers_pos_neg = calc_pos_neg_indices(dm_history, split_relu_indices)
    dm_relu_A = None
    dm_relu_b = None
    for i, post_relu_dict in enumerate(A_dict_relus):
        # post_relu_dict = post_process_A_dict_relu(A_dict)
        pos_indices = np.array(layers_pos_neg[i][0])
        neg_indices = np.array(layers_pos_neg[i][1])
        lA_mat = post_relu_dict['lA'][indicator]
        lbias_vec = post_relu_dict['lbias'][indicator]
        uA_mat = post_relu_dict['uA'][indicator]
        ubias_vec = post_relu_dict['ubias'][indicator]        
        if len(pos_indices) != 0:
            pos_mat = lA_mat[pos_indices,:]
            pos_bias = lbias_vec[pos_indices]
            pos_bias = pos_bias.reshape(pos_bias.shape[0],1)
        if len(neg_indices) != 0:
            neg_mat = uA_mat[neg_indices,:]
            neg_bias = ubias_vec[neg_indices]
            neg_bias = neg_bias.reshape(neg_bias.shape[0],1)
        if dm_relu_A is None:
            if len(pos_indices) != 0 and len(neg_indices) != 0:
                dm_relu_A = np.vstack((pos_mat, -neg_mat))
                dm_relu_b = np.vstack((pos_bias, -neg_bias))
            elif len(pos_indices) != 0 and len(neg_indices) == 0:
                dm_relu_A = pos_mat
                dm_relu_b = pos_bias
            elif len(pos_indices) == 0 and len(neg_indices) != 0:
                dm_relu_A = -neg_mat
                dm_relu_b = -neg_bias
            else:
                continue
        else:
            if len(pos_indices) != 0 and len(neg_indices) != 0:
                dm_relu_A = np.vstack((dm_relu_A, pos_mat, -neg_mat))
                dm_relu_b = np.vstack((dm_relu_b, pos_bias, -neg_bias))
            elif len(pos_indices) != 0 and len(neg_indices) == 0:
                dm_relu_A = np.vstack((dm_relu_A, pos_mat))
                dm_relu_b = np.vstack((dm_relu_b, pos_bias))
            elif len(pos_indices) == 0 and len(neg_indices) != 0:
                dm_relu_A = np.vstack((dm_relu_A, -neg_mat))
                dm_relu_b = np.vstack((dm_relu_b, -neg_bias))                
    dm_relu_b = dm_relu_b.reshape(dm_relu_b.shape[0],1)
    return dm_relu_A, dm_relu_b
def load_act_vecs(dataset_tp):
    # if arguments.Config["model"]["onnx_path"] is None:
    act_file = os.path.join(arguments.Config['preimage']["sample_dir"], 'act_vec_{}.pkl'.format(arguments.Config["data"]["dataset"]))
    with open(act_file, 'rb') as f:
        activation = pickle.load(f)
    if "MNIST" in dataset_tp:
        pre_relu_layer = ['2', '4', '6', '8', '10']
    elif "auto_park" in dataset_tp:
        pre_relu_layer = ['2']
    elif "vcas" in dataset_tp:
        pre_relu_layer = ['1']
    elif "cartpole" in dataset_tp:
        pre_relu_layer = ['8', '10']
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
def calc_relu_refine_approx_coverage_image_under(A, label, sample_left_idx,sample_right_idx,spec_dim=1):
    A_b_dict_multi = post_process_greedy_A(A)
    # poly_dict = dict([('lA', []), ('lbias', [])])
    dataset_tp = arguments.Config["data"]["dataset"]    
    sample_num = arguments.Config["preimage"]["sample_num"]   
    sample_dir = arguments.Config["preimage"]["sample_dir"]
    atk_tp = arguments.Config["preimage"]["atk_tp"]
    upper_time_loss = arguments.Config["preimage"]["upper_time_loss"]
    if arguments.Config["specification"]["type"] == "bound":
        if arguments.Config["model"]["onnx_path"] is None:
            samples = torch.load(os.path.join(sample_dir, 'sample_{}_{}.pt'.format(arguments.Config["data"]["dataset"], atk_tp)))
        else:
            if arguments.Config["data"]["dataset"] == "vcas":
                samples = np.load(os.path.join(sample_dir, "sample_{}_{}.npy".format(dataset_tp, upper_time_loss)))
            else:
                samples = np.load(os.path.join(sample_dir, f"sample_{dataset_tp}.npy"))
            samples = np.squeeze(samples, axis=1)
    elif arguments.Config["specification"]["type"] == 'lp':
        samples = torch.load(os.path.join(arguments.Config['preimage']["sample_dir"], 'sample_{}_{}.pt'.format(arguments.Config["data"]["dataset"], arguments.Config["preimage"]["atk_tp"])))
    if arguments.Config["model"]["onnx_path"] is None:   
        if "Customized" in dataset_tp:
            model = load_model(weights_loaded=False)
        else:
            model = load_model()
        device = arguments.Config["general"]["device"]
        model = model.to(device)      
        predicted = model(samples).argmax(dim=1).cpu().detach().numpy()
    else:
        if dataset_tp == "vcas":
            predicted = np.load(os.path.join(sample_dir, f"pred_output_{dataset_tp}_{upper_time_loss}.npy"))
            predicted = predicted.argmax(axis=1)
        else:
            predicted = np.load(os.path.join(sample_dir, f"pred_output_{dataset_tp}.npy"))
            predicted = np.squeeze(predicted, axis=1).argmax(axis=1)
    sample_label_idxs = set(np.where(predicted==label)[0])
    # sample_left_idx_file = os.path.join(sample_dir,'sample_left_{}.pt'.format(dataset_tp))
    # sample_right_idx_file = os.path.join(sample_dir,'sample_right_{}.pt'.format(dataset_tp))
    # if not os.path.exists(sample_left_idx_file):
    #     acti_vecs = load_act_vecs(dataset_tp)
    #     sample_left_idx = calc_history_idxs(acti_vecs,left_history)
    #     sample_right_idx = calc_history_idxs(acti_vecs, right_history)
    # else:
    #     with open(sample_left_idx_file, 'rb') as f:
    #         sample_left_idx = pickle.load(f)
    #     with open(sample_right_idx_file, 'rb') as f:
    #         sample_right_idx = pickle.load(f)
    cov_subdomain_info = []  
    sample_label_left = sample_label_idxs.intersection(sample_left_idx) 
    sample_label_right = sample_label_idxs.intersection(sample_right_idx)            
    if len(sample_label_left) > 0:
        target_vol_left = len(sample_label_left)/sample_num
        print('Label: {}, Split left, Num: {}'.format(label, len(sample_label_left)))
        sample_temp = samples[list(sample_label_left)]
        if arguments.Config["model"]["onnx_path"] is None: 
            sample_temp = sample_temp.reshape(sample_temp.shape[0], -1)
            sample_temp = sample_temp.cpu().detach().numpy()
        mat = A_b_dict_multi['lA'][0]
        bias = A_b_dict_multi['lbias'][0]
        print('mat shape: {}, samples_T shape: {}'.format(mat.shape, sample_temp.T.shape))
        result = np.matmul(mat, sample_temp.T)+bias
        # NOTE spec dim to be dealt with future
        if len(result) > 1:
            idxs_True = None
            for const_dim in range(len(result)):
                idxs_tmp = set(np.where(result[const_dim]>=0)[0])
                if idxs_True is None:
                    idxs_True = idxs_tmp
                else:
                    idxs_True = idxs_True.intersection(idxs_tmp)
        else:
            idxs_True = np.where(result>=0)[0] 
        cov_subdomain_info.append((target_vol_left, len(idxs_True)/len(sample_label_left)))         
    else:
        cov_subdomain_info.append((0,0))
        print('No samples evaluted follow this left relu split history')
    if len(sample_label_right) > 0:
        target_vol_right = len(sample_label_right)/sample_num
        print('Label: {}, Split right, Num: {}'.format(label, len(sample_label_right)))
        sample_temp = samples[list(sample_label_right)]
        if arguments.Config["model"]["onnx_path"] is None: 
            sample_temp = sample_temp.reshape(sample_temp.shape[0], -1)
            sample_temp = sample_temp.cpu().detach().numpy()
        mat = A_b_dict_multi['lA'][1]
        bias = A_b_dict_multi['lbias'][1]
        print('mat shape: {}, samples_T shape: {}'.format(mat.shape, sample_temp.T.shape))
        result = np.matmul(mat, sample_temp.T)+bias
        # NOTE spec dim to be dealt with future
        if len(result) > 1:
            idxs_True = None
            for const_dim in range(len(result)):
                idxs_tmp = set(np.where(result[const_dim]>=0)[0])
                if idxs_True is None:
                    idxs_True = idxs_tmp
                else:
                    idxs_True = idxs_True.intersection(idxs_tmp)
        else:
            idxs_True = np.where(result>=0)[0]  
        cov_subdomain_info.append((target_vol_right, len(idxs_True)/len(sample_label_right)))
    else:
        cov_subdomain_info.append((0,0))
        print('No samples evaluted follow this right relu split history')
    return cov_subdomain_info, A_b_dict_multi             

def calc_relu_refine_approx_coverage_image_over(A, label, sample_left_idx,sample_right_idx,spec_dim=1):
    A_b_dict_multi = post_process_greedy_A(A)
    # poly_dict = dict([('lA', []), ('lbias', [])])
    dataset_tp = arguments.Config["data"]["dataset"]    
    sample_num = arguments.Config["preimage"]["sample_num"]   
    sample_dir = arguments.Config["preimage"]["sample_dir"]
    atk_tp = arguments.Config["preimage"]["atk_tp"]
    upper_time_loss = arguments.Config["preimage"]["upper_time_loss"]
    if arguments.Config["specification"]["type"] == "bound":
        if arguments.Config["model"]["onnx_path"] is None:
            samples = torch.load(os.path.join(sample_dir, 'sample_{}_{}.pt'.format(arguments.Config["data"]["dataset"], atk_tp)))
        else:
            samples = np.load(os.path.join(sample_dir, f"{dataset_tp}/sample_{dataset_tp}.npy"))
            # if arguments.Config["data"]["dataset"] == "vcas":
            #     samples = np.load(os.path.join(sample_dir, "sample_{}_{}.npy".format(dataset_tp, upper_time_loss)))
            # else:
            #     samples = np.load(os.path.join(sample_dir, f"sample_{dataset_tp}.npy"))
            samples = np.squeeze(samples, axis=1)
    elif arguments.Config["specification"]["type"] == 'lp':
        samples = torch.load(os.path.join(arguments.Config['preimage']["sample_dir"], 'sample_{}_{}.pt'.format(dataset_tp, arguments.Config["preimage"]["atk_tp"])))
    if arguments.Config["model"]["onnx_path"] is None:   
        if "Customized" in dataset_tp:
            model = load_model(weights_loaded=False)
        else:
            model = load_model()
        device = arguments.Config["general"]["device"]
        model = model.to(device)      
        predicted = model(samples).argmax(dim=1).cpu().detach().numpy()
        sample_label_idxs = set(np.where(predicted==label)[0])
    else:
        if dataset_tp == "vcas":
            predicted = np.load(os.path.join(sample_dir, f"{dataset_tp}/pred_output_{dataset_tp}.npy"))
            predicted = predicted.argmax(axis=1)
            sample_label_idxs = set(np.where(predicted==label)[0])
        elif dataset_tp == 'dubinsrejoin':
            predicted = np.load(os.path.join(sample_dir, f"{dataset_tp}/pred_output_{dataset_tp}.npy"))
            # predicted = np.squeeze(predicted, axis=1)
            if label == 0:
                labels_group_0 = np.argmax(predicted[:, :4], axis=1)
            elif label == 4:
                labels_group_0 = np.argmax(predicted[:, 4:], axis=1)
            sample_label_idxs = set(np.where(labels_group_0==label)[0])
            # labels_group_1 = np.argmax(predicted[:, 4:8], axis=1)
            # label_truth_idx_0 = np.where(labels_group_0==0)[0]
            # label_truth_idx_1 = np.where(labels_group_1==4)[0]
            # sample_label_idxs = np.intersect1d(label_truth_idx_0, label_truth_idx_1)
        else:
            predicted = np.load(os.path.join(sample_dir, f"{dataset_tp}/pred_output_{dataset_tp}.npy"))
            predicted = np.squeeze(predicted, axis=1).argmax(axis=1)
            sample_label_idxs = set(np.where(predicted==label)[0])


    # sample_left_idx_file = os.path.join(sample_dir,'sample_left_{}.pt'.format(dataset_tp))
    # sample_right_idx_file = os.path.join(sample_dir,'sample_right_{}.pt'.format(dataset_tp))
    # if not os.path.exists(sample_left_idx_file):
    #     acti_vecs = load_act_vecs(dataset_tp)
    #     sample_left_idx = calc_history_idxs(acti_vecs,left_history)
    #     sample_right_idx = calc_history_idxs(acti_vecs, right_history)
    # else:
    #     with open(sample_left_idx_file, 'rb') as f:
    #         sample_left_idx = pickle.load(f)
    #     with open(sample_right_idx_file, 'rb') as f:
    #         sample_right_idx = pickle.load(f)
    cov_subdomain_info = []  
    sample_label_left = sample_label_idxs.intersection(sample_left_idx) 
    sample_label_right = sample_label_idxs.intersection(sample_right_idx)            
    if len(sample_label_left) > 0:
        target_vol_left = len(sample_label_left)/sample_num
        print('Label: {}, Split left, Num: {}'.format(label, len(sample_label_left)))
        sample_temp = samples[list(sample_left_idx)]
        if arguments.Config["model"]["onnx_path"] is None: 
            sample_temp = sample_temp.reshape(sample_temp.shape[0], -1)
            sample_temp = sample_temp.cpu().detach().numpy()
        mat = A_b_dict_multi['uA'][0]
        bias = A_b_dict_multi['ubias'][0]
        print('mat shape: {}, samples_T shape: {}'.format(mat.shape, sample_temp.T.shape))
        result = np.matmul(mat, sample_temp.T)+bias
        # NOTE spec dim to be dealt with future
        if len(result) > 1:
            idxs_True = None
            for const_dim in range(len(result)):
                idxs_tmp = set(np.where(result[const_dim]>=0)[0])
                if idxs_True is None:
                    idxs_True = idxs_tmp
                else:
                    idxs_True = idxs_True.intersection(idxs_tmp)
        else:
            idxs_True = np.where(result>=0)[0] 
        cov_subdomain_info.append((target_vol_left, len(idxs_True)/len(sample_label_left)))         
    else:
        cov_subdomain_info.append((0,0))
        print('No samples evaluted follow this left relu split history')
    if len(sample_label_right) > 0:
        target_vol_right = len(sample_label_right)/sample_num
        print('Label: {}, Split right, Num: {}'.format(label, len(sample_label_right)))
        sample_temp = samples[list(sample_right_idx)]
        if arguments.Config["model"]["onnx_path"] is None: 
            sample_temp = sample_temp.reshape(sample_temp.shape[0], -1)
            sample_temp = sample_temp.cpu().detach().numpy()
        mat = A_b_dict_multi['uA'][1]
        bias = A_b_dict_multi['ubias'][1]
        print('mat shape: {}, samples_T shape: {}'.format(mat.shape, sample_temp.T.shape))
        result = np.matmul(mat, sample_temp.T)+bias
        # NOTE spec dim to be dealt with future
        if len(result) > 1:
            idxs_True = None
            for const_dim in range(len(result)):
                idxs_tmp = set(np.where(result[const_dim]>=0)[0])
                if idxs_True is None:
                    idxs_True = idxs_tmp
                else:
                    idxs_True = idxs_True.intersection(idxs_tmp)
        else:
            idxs_True = np.where(result>=0)[0]  
        cov_subdomain_info.append((target_vol_right, len(idxs_True)/len(sample_label_right)))
    else:
        cov_subdomain_info.append((0,0))
        print('No samples evaluted follow this right relu split history')
    return cov_subdomain_info, A_b_dict_multi    
def calc_relu_refine_approx_coverage_image_final(A, A_dict_relus, left_history, right_history, split_relu_indices, label, spec_dim=1):
    A_b_dict_multi = post_process_greedy_A(A)
    A_b_dict_relus = post_process_A_dict_relu(A_dict_relus)
    left_relu_A, left_relu_b = calc_extra_A_b(A_b_dict_relus, left_history, split_relu_indices, indicator=0)
    right_relu_A, right_relu_b = calc_extra_A_b(A_b_dict_relus, right_history, split_relu_indices, indicator=1)
    poly_dict = dict([('lA', []), ('lbias', [])])
    model_tp = arguments.Config["model"]["name"]    
    sample_num = arguments.Config["preimage"]["sample_num"]   
    if arguments.Config["specification"]["type"] == "bound":
        samples = torch.load(os.path.join(arguments.Config['preimage']["sample_dir"], 'sample_{}.pt'.format(arguments.Config["data"]["dataset"])))
    elif arguments.Config["specification"]["type"] == 'lp':
        samples = torch.load(os.path.join(arguments.Config['preimage']["sample_dir"], 'sample_{}.pt'.format(arguments.Config["data"]["dataset"])))
        # X, labels, runnerup, data_max, data_min, perturb_epsilon, target_label = load_verification_dataset(eps_before_normalization=None)
        # # img_shape = X[0].cpu().detach().numpy().shape
        # img_shape = [-1] + list(X.shape[1:])
        # x_lower = (X[0] - perturb_epsilon).clamp(min=data_min).flatten(1)
        # x_upper = (X[0] + perturb_epsilon).clamp(max=data_max).flatten(1)
        # data_ub = torch.squeeze(x_upper).cpu().detach().numpy()
        # data_lb = torch.squeeze(x_lower).cpu().detach().numpy()
        # label = labels.cpu().detach().numpy()[0]
    # dm_vol = calc_total_sub_vol(data_lb, data_ub)   
    model = load_model()
    device = arguments.Config["general"]["device"]
    model = model.to(device)      
    cov_subdomain_info = []
    predicted = model(samples).argmax(dim=1).cpu().detach().numpy()
    sample_label_idxs = set(np.where(predicted==label)[0])
    act_file = os.path.join(arguments.Config['preimage']["sample_dir"], 'act_vec_{}.pkl'.format(arguments.Config["data"]["dataset"]))
    if os.path.exists(act_file):
        with open(act_file, 'rb') as f:
            activation = pickle.load(f)
        if "mnist" in model_tp:
            pre_relu_layer = ['2', '4', '6', '8', '10']
        elif "auto_park" in model_tp:
            pre_relu_layer = ['2']
    else:
        node_types = [m for m in list(model.modules())]
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        if "mnist" in model_tp:
            pre_relu_layer = ['2', '4', '6', '8', '10']
        elif "auto_park" in model_tp:
            pre_relu_layer = ['2']
        for i, layer in enumerate(pre_relu_layer):
            model_module = node_types[int(layer)]
            model_module.register_forward_hook(get_activation(layer))
            output = model(samples)
            print(activation[layer].shape)      
    acti_vecs = []
    for i, layer in enumerate(pre_relu_layer):
        act_vec = activation[layer].cpu().detach().numpy()
        acti_vecs.append(act_vec)
    sample_left = None
    sample_right = None
    # NOTE the 
    # acti_vecs outputs should be corresponding to relus recorded
    for i, layer_info in enumerate(left_history): # enumerate over relu layers
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
            if sample_left is None:
                sample_left = set(temp_idx)
            else:
                sample_left = sample_left.intersection(set(temp_idx))
    for i, layer_info in enumerate(right_history): # enumerate over relu layers
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
            if sample_right is None:
                sample_right = set(temp_idx)
            else:
                sample_right = sample_right.intersection(set(temp_idx))  
    sample_label_left = sample_label_idxs.intersection(sample_left) 
    sample_label_right = sample_label_idxs.intersection(sample_right)            
    if len(sample_label_left) > 0:
        target_vol_left = len(sample_label_left)/sample_num
        print('Label: {}, Split left, Num: {}'.format(label, len(sample_label_left)))
        sample_temp = samples[list(sample_label_left)]
        sample_temp = sample_temp.reshape(sample_temp.shape[0], -1)
        sample_temp = sample_temp.cpu().detach().numpy()
        mat = A_b_dict_multi['lA'][0]
        bias = A_b_dict_multi['lbias'][0]
        print('mat shape: {}, samples_T shape: {}'.format(mat.shape, sample_temp.T.shape))
        mat = np.vstack((mat, left_relu_A))
        bias = np.vstack((bias, left_relu_b))
        print('mat shape: {}, bias shape: {}'.format(mat.shape, bias.shape))
        poly_dict['lA'].append(mat)
        poly_dict['lbias'].append(bias)
        result = np.matmul(mat, sample_temp.T)+bias
        # NOTE spec dim to be dealt with future
        if len(result) > 1:
            idxs_True = None
            for const_dim in range(len(result)):
                idxs_tmp = set(np.where(result[const_dim]>=0)[0])
                if idxs_True is None:
                    idxs_True = idxs_tmp
                else:
                    idxs_True = idxs_True.intersection(idxs_tmp)
        else:
            idxs_True = np.where(result>=0)[0] 
        cov_subdomain_info.append((target_vol_left, len(idxs_True)/len(sample_label_left)))         
    else:
        cov_subdomain_info.append((0,0))
        poly_dict['lA'].append(None)
        poly_dict['lbias'].append(None)
        print('No samples evaluted follow this left relu split history')
    if len(sample_label_right) > 0:
        target_vol_right = len(sample_label_right)/sample_num
        print('Label: {}, Split right, Num: {}'.format(label, len(sample_label_right)))
        sample_temp = samples[list(sample_label_right)]
        sample_temp = sample_temp.reshape(sample_temp.shape[0], -1)
        sample_temp = sample_temp.cpu().detach().numpy()
        mat = A_b_dict_multi['lA'][1]
        bias = A_b_dict_multi['lbias'][1]
        print('mat shape: {}, samples_T shape: {}'.format(mat.shape, sample_temp.T.shape))
        mat = np.vstack((mat, right_relu_A))
        bias = np.vstack((bias, right_relu_b))
        print('mat shape: {}, bias shape: {}'.format(mat.shape, bias.shape))
        poly_dict['lA'].append(mat)
        poly_dict['lbias'].append(bias)
        result = np.matmul(mat, sample_temp.T)+bias
        # NOTE spec dim to be dealt with future
        if len(result) > 1:
            idxs_True = None
            for const_dim in range(len(result)):
                idxs_tmp = set(np.where(result[const_dim]>=0)[0])
                if idxs_True is None:
                    idxs_True = idxs_tmp
                else:
                    idxs_True = idxs_True.intersection(idxs_tmp)
        else:
            idxs_True = np.where(result>=0)[0]  
        cov_subdomain_info.append((target_vol_right, len(idxs_True)/len(sample_label_right)))
    else:
        cov_subdomain_info.append((0,0))
        poly_dict['lA'].append(None)
        poly_dict['lbias'].append(None)
        print('No samples evaluted follow this right relu split history')
    return cov_subdomain_info, poly_dict                    
def calc_relu_refine_approx_coverage(A, A_dict_relus, left_history, right_history, split_relu_indices, spec_dim=1):
    A_b_dict_multi=post_process_greedy_A(A)
    A_b_dict_relus = post_process_A_dict_relu(A_dict_relus)
    left_relu_A, left_relu_b = calc_extra_A_b(A_b_dict_relus, left_history, split_relu_indices, indicator=0)
    right_relu_A, right_relu_b = calc_extra_A_b(A_b_dict_relus, right_history, split_relu_indices, indicator=1)
    poly_dict = dict([('lA', []), ('lbias', [])])
    np.random.seed(0)
    dataset_tp = arguments.Config["data"]["dataset"]  
    sample_num = arguments.Config["preimage"]["sample_num"]   
    _, label, _, data_ub, data_lb, _, _ = load_verification_dataset(eps_before_normalization=None)
    data_ub = torch.squeeze(data_ub).cpu().detach().numpy()
    data_lb = torch.squeeze(data_lb).cpu().detach().numpy()
    label = label.cpu().detach().numpy()[0]
    dm_vol = calc_total_sub_vol(data_lb, data_ub)                
    samples = np.random.uniform(low=data_lb, high=data_ub, size=(sample_num, len(data_lb)))
    if "Customized" in dataset_tp:
        model = build_model_activation(dataset_tp)
        model.eval()
    else:
        model = load_model()    
    cov_subdomain_info = []
    data = torch.tensor(samples, dtype=torch.get_default_dtype())
    predicted, acti_vecs = model(data)
    predicted = predicted.argmax(dim=1)
    sample_label_idxs = set(np.where(predicted==label)[0])
    sample_left = None
    sample_right = None
    # NOTE the build_model_activation outputs should be corresponding to relus recorded
    for i, layer_info in enumerate(left_history): # enumerate over relu layers
        neuron_idxs = layer_info[0]
        neuron_signs = layer_info[1]
        for j, neuron_id in enumerate(neuron_idxs):
            neuron_sign = neuron_signs[j]
            if neuron_sign == +1:
                if acti_vecs.shape[0] == sample_num:
                    temp_idx = np.where(acti_vecs[:, neuron_id]>=0)[0]
                else:
                    temp_idx = np.where(acti_vecs[i][:, neuron_id]>=0)[0]
            elif neuron_sign == -1:
                if acti_vecs.shape[0] == sample_num:
                    temp_idx = np.where(acti_vecs[:, neuron_id]<0)[0]
                else:
                    temp_idx = np.where(acti_vecs[i][:, neuron_id]<0)[0]
            else:
                print("neuron sign assignment error")
            if sample_left is None:
                sample_left = set(temp_idx)
            else:
                sample_left = sample_left.intersection(set(temp_idx))
    for i, layer_info in enumerate(right_history): # enumerate over relu layers
        neuron_idxs = layer_info[0]
        neuron_signs = layer_info[1]
        for j, neuron_id in enumerate(neuron_idxs):
            neuron_sign = neuron_signs[j]
            if neuron_sign == +1:
                if acti_vecs.shape[0] == sample_num:
                    temp_idx = np.where(acti_vecs[:, neuron_id]>=0)[0]
                else:
                    temp_idx = np.where(acti_vecs[i][:, neuron_id]>=0)[0]
            elif neuron_sign == -1:
                if acti_vecs.shape[0] == sample_num:
                    temp_idx = np.where(acti_vecs[:, neuron_id]<0)[0]
                else:
                    temp_idx = np.where(acti_vecs[i][:, neuron_id]<0)[0]
            else:
                print("neuron sign assignment error")
            if sample_right is None:
                sample_right = set(temp_idx)
            else:
                sample_right = sample_right.intersection(set(temp_idx))  
    sample_label_left = sample_label_idxs.intersection(sample_left) 
    sample_label_right = sample_label_idxs.intersection(sample_right)            
    if len(sample_label_left) > 0:
        target_vol_left = dm_vol * len(sample_label_left)/sample_num
        print('Label: {}, Split left, Num: {}'.format(label, len(sample_label_left)))
        sample_temp = samples[list(sample_label_left)]
        mat = A_b_dict_multi['lA'][0]
        bias = A_b_dict_multi['lbias'][0]
        print('mat shape: {}, samples_T shape: {}'.format(mat.shape, sample_temp.T.shape))
        mat = np.vstack((mat, left_relu_A))
        bias = np.vstack((bias, left_relu_b))
        print('mat shape: {}, bias shape: {}'.format(mat.shape, bias.shape))
        poly_dict['lA'].append(mat)
        poly_dict['lbias'].append(bias)
        result = np.matmul(mat, sample_temp.T)+bias
        # NOTE spec dim to be dealt with future
        if len(result) > 1:
            idxs_True = None
            for const_dim in range(len(result)):
                idxs_tmp = set(np.where(result[const_dim]>=0)[0])
                if idxs_True is None:
                    idxs_True = idxs_tmp
                else:
                    idxs_True = idxs_True.intersection(idxs_tmp)
        else:
            idxs_True = np.where(result>=0)[0] 
        cov_subdomain_info.append((target_vol_left, len(idxs_True)/len(sample_label_left)))         
    else:
        cov_subdomain_info.append((0,0))
        print('No samples evaluted follow this left relu split history')
    if len(sample_label_right) > 0:
        target_vol_right = dm_vol * len(sample_label_right)/sample_num
        print('Label: {}, Split right, Num: {}'.format(label, len(sample_label_right)))
        sample_temp = samples[list(sample_label_right)]
        mat = A_b_dict_multi['lA'][1]
        bias = A_b_dict_multi['lbias'][1]
        print('mat shape: {}, samples_T shape: {}'.format(mat.shape, sample_temp.T.shape))
        mat = np.vstack((mat, right_relu_A))
        bias = np.vstack((bias, right_relu_b))
        print('mat shape: {}, bias shape: {}'.format(mat.shape, bias.shape))
        poly_dict['lA'].append(mat)
        poly_dict['lbias'].append(bias)
        result = np.matmul(mat, sample_temp.T)+bias
        # NOTE spec dim to be dealt with future
        if len(result) > 1:
            idxs_True = None
            for const_dim in range(len(result)):
                idxs_tmp = set(np.where(result[const_dim]>=0)[0])
                if idxs_True is None:
                    idxs_True = idxs_tmp
                else:
                    idxs_True = idxs_True.intersection(idxs_tmp)
        else:
            idxs_True = np.where(result>=0)[0]  
        cov_subdomain_info.append((target_vol_right, len(idxs_True)/len(sample_label_right)))
    else:
        cov_subdomain_info.append((0,0))
        print('No samples evaluted follow this right relu split history')
    return cov_subdomain_info, poly_dict
    
def calc_under_approx_coverage(A, args):
    # args = arguments().parse_args()
    # if os.path.exists(A_path):
    #     linear_rep_dict = load_A_dict(A_path)
    # else:
    linear_rep_dict = post_process_A(A)
    # with open(A_path, "wb") as f:
    #     pickle.dump(linear_rep_dict, f)
    # coeffs and biases are already saved and loaded here
    if args.add_layer:
        # addtional nn layer for under-approximation calculation
        print(linear_rep_dict)
        pass
    else:
        # need addtional operation to calculate under-approximation
        A_b_dict = calc_input_Hrep(linear_rep_dict)
        save_A_dict(A_b_dict, "vcas_1_init", args.result_dir)
        cov_quota_dict = calc_Hrep_coverage(A_b_dict, args)
    return cov_quota_dict, A_b_dict

def test_alpha_optim_quality(A, args, theta):
    A_dir = os.path.join(args.result_dir, '{}_{}_{}_{}'.format(args.model_name, args.dataset, args.base, args.label))
    if not os.path.exists(A_dir):
        os.makedirs(A_dir)
    linear_rep_dict = dict()
    for output_node, linear_rep in A.items():
        for input_node, param_dict in linear_rep.items():
            for k, v in param_dict.items():
                if v is not None:
                    linear_rep_dict[k] = torch.squeeze(v).cpu().detach().numpy()
    # polyA = linear_rep_dict["lA"]
    # polyb = linear_rep_dict["lbias"]
    cov_quota = calc_input_coverage_single_label(linear_rep_dict)
    
    
    
    
def eval_under_approx_quality(A, args, theta):
    # Check whether the under-approx has reached minimum coverage theta
    if "Customized" in args.model_name:
        args.model_name = "Customized_model"
    # if args.add_layer:    
    #     A_file = '{}_{}_{}_label_{}_init'.format(args.model_name, args.dataset, args.base, args.label)
    # else:
    #     A_file = '{}_{}_{}_init'.format(args.model_name, args.dataset, args.base)
    A_dir = os.path.join(args.result_dir, '{}_{}_{}'.format(args.model_name, args.dataset, args.base))
    if not os.path.exists(A_dir):
        os.makedirs(A_dir)
    under_approx_coverage, poly_A_b_dict = calc_under_approx_coverage(A, args)
    decision_labels = list(under_approx_coverage.keys())
    under_approx_to_refine = []
    # lAs = dict()
    if len(decision_labels) > 1:
        for l in decision_labels:
            if under_approx_coverage[l] < theta:
                under_approx_to_refine.append(l)
                # lA = process_lA(under_approx_A_b[l])
                # lAs[l] = lA
            else:
                A_file = 'polytope_label_{}'.format(l)
                save_A_dict(poly_A_b_dict[l], A_file, A_dir)
    else:
        print('Currently use the multi-label C to have linear approx through one run')
        print('Support for one label to be added')
    return under_approx_to_refine
        
def calc_input_Hrep(linear_param_dict):
    for k, v in linear_param_dict.items():
        print("Param: {}, Shape: {}".format(k, v.shape))
    lA = linear_param_dict["lA"]
    uA = linear_param_dict["uA"]
    lbias = linear_param_dict["lbias"]
    ubias = linear_param_dict["ubias"]
    output_dim = lbias.shape[0]
    A_b_dict = dict()
    # calculate input H representation for each label (multiple label's function is calculated)
    for i in range(output_dim):
        # calculate A of cdd mat for class i
        tA = np.reshape(lA[i], [1, -1])
        tA_rep = np.repeat(tA, output_dim-1, axis=0)
        uA_del = np.delete(uA, i, axis=0)
        # print(tA_rep.shape, uA_del.shape)
        assert (tA_rep.shape == uA_del.shape)
        polyA = tA_rep - uA_del
        # calculate b of cdd mat for class i
        tbias_rep = np.repeat(lbias[i], output_dim-1)
        ubias_del = np.delete(ubias, i)
        # print(tbias_rep.shape, ubias_del.shape)
        assert (tbias_rep.shape == ubias_del.shape)
        polyb = tbias_rep - ubias_del
        print('polyA, polyb', polyA, polyb)
        A_b_dict[i] = (polyA, polyb)
    return A_b_dict


def calc_Hrep_coverage_multi_spec_pairwise_over(A_b_dict, dm_l_all, dm_u_all, spec_dim):
    torch.manual_seed(arguments.Config["general"]["seed"])
    dataset_tp = arguments.Config["data"]["dataset"]  
    sample_num = arguments.Config["preimage"]["sample_num"]
    label = arguments.Config["preimage"]["label"]
    pair_num = int(len(dm_l_all)/(spec_dim * 2))
    bisec_sample_num = int(sample_num / 2)
    if "Customized" in dataset_tp:
        model = load_model(weights_loaded=False)
    else:
        model = load_model()
    if dataset_tp == 'dubinsrejoin':
        label_T = arguments.Config["preimage"]["runner_up"] - 4
    else:
        label_T = None
    device = arguments.Config["general"]["device"]
    model = model.to(device)
    cov_input_idx_all = [[] for _ in range(pair_num)] 
    for i in range(pair_num):
        cov_num = 0
        total_num = 0
        total_target_vol = 0
        total_loss = 0
        for j in range(2):
            dm_l, dm_u = dm_l_all[2*i*spec_dim+j*spec_dim], dm_u_all[2*i*spec_dim+j*spec_dim]
        # dm_l_1, dm_u_1 = dm_l_all[2*i+1], dm_u_all[2*i+1]
        # if samples is None:
            # samples = np.random.uniform(low=dm_l,high=dm_u,size=(sub_sample_num, len(dm_l)))
        # else:
            dm_vol = calc_total_sub_vol(dm_l, dm_u)
            # total_vol += dm_vol
            samples = Uniform(dm_l, dm_u).sample([bisec_sample_num])
            # tmp_samples = np.random.uniform(low=dm_l,high=dm_u,size=(bisec_sample_num, len(dm_l)))
            # data = torch.tensor(tmp_samples, dtype=torch.get_default_dtype())
            samples = samples.to(device)
            if dataset_tp == 'dubinsrejoin':
                prediction = model(samples)
                pred_R = prediction[:,:4]
                pred_T = prediction[:, 4:]
                pred_label_R = pred_R.argmax(dim=1).cpu().detach().numpy() 
                pred_label_T = pred_T.argmax(dim=1).cpu().detach().numpy()
                samples_idxs_R = np.where(pred_label_R == label)[0]
                samples_idxs_T = np.where(pred_label_T == label_T)[0]
                samples_eval_idxs = np.intersect1d(samples_idxs_R, samples_idxs_T)
            else:
                predicted = model(samples).argmax(dim=1).cpu().detach().numpy()    
                samples_eval_idxs = np.where(predicted==label)[0]
            target_num = len(samples_eval_idxs)
            target_vol = dm_vol * target_num / bisec_sample_num
            # cov_input_idx_all[i].append(target_vol)
            total_target_vol += target_vol
            if target_num > 0:
                samples_eval = samples.cpu().detach().numpy()
                if spec_dim == 1:
                    mat = A_b_dict['uA'][2*i+j]
                    bias = A_b_dict['ubias'][2*i+j]
                else:
                    mat = A_b_dict['uA'][2*i*spec_dim+j*spec_dim : 2*i*spec_dim+(j+1)*spec_dim]
                    bias = A_b_dict['ubias'][2*i*spec_dim+j*spec_dim : 2*i*spec_dim+(j+1)*spec_dim]
                    if dataset_tp != 'cartpole':
                        mat = np.squeeze(mat, axis=1)   
                print('Pair {}, subsection {}, mat shape: {}, samples_eval_T shape: {}'.format(i, j, mat.shape, samples_eval.T.shape))
                result = np.matmul(mat, samples_eval.T)+bias
                sub_tight_loss = calc_over_tightness(result, preimg_idx=samples_eval_idxs)
                if spec_dim > 1:
                    idxs_True = None
                    for s_dim in range(spec_dim):
                        idxs_tmp = set(np.where(result[s_dim]>=0)[0])
                        if idxs_True is None:
                            idxs_True = idxs_tmp
                        else:
                            idxs_True = idxs_True.intersection(idxs_tmp)
                else:
                    idxs_True = np.where(result>=0)[0]
                cov_input_idx_all[i].append((target_vol, len(idxs_True)/target_num, sub_tight_loss))
                cov_num += len(idxs_True)
                total_num += target_num
                total_loss += sub_tight_loss
            else:
                cov_input_idx_all[i].append((0,0,2))
                print('Pair {}, subsection {}, No samples of NN on: dm_l {}, dm_u {}'.format(i, j, dm_l, dm_u))# In this case, the subdomain will not lead to the target label, no need for further branching. set the cov_quota as 1, uncov_vol will be 0
            # however, when evaluating the generall coverage volume, the coverage quota is not calculated as 1, instead making an impact by not adding to the total number
        if total_num > 0:
            cov_quota = cov_num / total_num
            print("Pair {}, Coverage quota {}/{}:  {:.3f}, S-loss {:.2f}".format(i, cov_num, total_num, cov_quota, total_loss))
        else:
            cov_quota = 0
            print("Pair {}, Coverage quota {}/{}:  {:.3f}, S-loss {:.2f}".format(i, 0, 0, cov_quota, total_loss))
        # total_target_vol = total_vol * total_num / sample_num
        cov_input_idx_all[i].append((total_target_vol,cov_quota,total_loss))
        # Therefore, for each idx i, it consists of the cov_ratio for each bisection and the overall of splitting wrt i-th input feat.
    return cov_input_idx_all        

# In the all potential feature split case, we need the cov_quota for each pairwise subdomain, not the overall for all domains
def calc_Hrep_coverage_multi_spec_pairwise_under(A_b_dict, dm_l_all, dm_u_all, spec_dim):
    torch.manual_seed(arguments.Config["general"]["seed"])
    dataset_tp = arguments.Config["data"]["dataset"]  
    sample_num = arguments.Config["preimage"]["sample_num"]
    label = arguments.Config["preimage"]["label"]
    pair_num = int(len(dm_l_all)/(spec_dim * 2))
    bisec_sample_num = int(sample_num / 2)
    if "Customized" in dataset_tp:
        model = load_model(weights_loaded=False)
    else:
        model = load_model()
    if dataset_tp == 'dubinsrejoin':
        label_T = arguments.Config["preimage"]["runner_up"] - 4
    else:
        label_T = None
    device = arguments.Config["general"]["device"]
    model = model.to(device)
    cov_input_idx_all = [[] for _ in range(pair_num)] 
    for i in range(pair_num):
        cov_num = 0
        total_num = 0
        total_target_vol = 0
        total_reward = 0
        for j in range(2):
            dm_l, dm_u = dm_l_all[2*i*spec_dim+j*spec_dim], dm_u_all[2*i*spec_dim+j*spec_dim]
        # dm_l_1, dm_u_1 = dm_l_all[2*i+1], dm_u_all[2*i+1]
        # if samples is None:
            # samples = np.random.uniform(low=dm_l,high=dm_u,size=(sub_sample_num, len(dm_l)))
        # else:
            dm_vol = calc_total_sub_vol(dm_l, dm_u)
            # total_vol += dm_vol
            samples = Uniform(dm_l, dm_u).sample([bisec_sample_num])
            # tmp_samples = np.random.uniform(low=dm_l,high=dm_u,size=(bisec_sample_num, len(dm_l)))
            # data = torch.tensor(tmp_samples, dtype=torch.get_default_dtype())
            samples = samples.to(device)
            if dataset_tp == 'dubinsrejoin':
                prediction = model(samples)
                pred_R = prediction[:,:4]
                pred_T = prediction[:, 4:]
                pred_label_R = pred_R.argmax(dim=1).cpu().detach().numpy() 
                pred_label_T = pred_T.argmax(dim=1).cpu().detach().numpy()
                samples_idxs_R = np.where(pred_label_R == label)[0]
                samples_idxs_T = np.where(pred_label_T == label_T)[0]
                samples_eval_idxs = np.intersect1d(samples_idxs_R, samples_idxs_T)
            else:
                predicted = model(samples).argmax(dim=1).cpu().detach().numpy()    
                samples_eval_idxs = np.where(predicted==label)[0]
            target_num = len(samples_eval_idxs)
            target_vol = dm_vol * target_num / bisec_sample_num
            # cov_input_idx_all[i].append(target_vol)
            total_target_vol += target_vol
            if target_num > 0:
                samples_eval = samples[samples_eval_idxs]
                samples_eval = samples_eval.cpu().detach().numpy()
                if spec_dim == 1:
                    mat = A_b_dict['lA'][2*i+j]
                    bias = A_b_dict['lbias'][2*i+j]
                else:
                    mat = A_b_dict['lA'][2*i*spec_dim+j*spec_dim : 2*i*spec_dim+(j+1)*spec_dim]
                    bias = A_b_dict['lbias'][2*i*spec_dim+j*spec_dim : 2*i*spec_dim+(j+1)*spec_dim]
                    if dataset_tp != 'cartpole':
                        mat = np.squeeze(mat, axis=1)   
                print('Pair {}, subsection {}, mat shape: {}, samples_eval_T shape: {}'.format(i, j, mat.shape, samples_eval.T.shape))
                result = np.matmul(mat, samples_eval.T)+bias
                sub_tight_reward = calc_over_tightness(result, preimg_idx=None)
                if spec_dim > 1:
                    idxs_True = None
                    for s_dim in range(spec_dim):
                        idxs_tmp = set(np.where(result[s_dim]>=0)[0])
                        if idxs_True is None:
                            idxs_True = idxs_tmp
                        else:
                            idxs_True = idxs_True.intersection(idxs_tmp)
                else:
                    idxs_True = np.where(result>=0)[0]
                cov_input_idx_all[i].append((target_vol, len(idxs_True)/target_num, sub_tight_reward))
                cov_num += len(idxs_True)
                total_num += target_num
                total_reward += sub_tight_reward
            else:
                cov_input_idx_all[i].append((0,0,-2))
                print('Pair {}, subsection {}, No samples of NN on: dm_l {}, dm_u {}'.format(i, j, dm_l, dm_u))# In this case, the subdomain will not lead to the target label, no need for further branching. set the cov_quota as 1, uncov_vol will be 0
            # however, when evaluating the generall coverage volume, the coverage quota is not calculated as 1, instead making an impact by not adding to the total number
        if total_num > 0:
            cov_quota = cov_num / total_num
            print("Pair {}, Coverage quota {}/{}:  {:.3f}, S-reward {:.2f}".format(i, cov_num, total_num, cov_quota, total_reward))
        else:
            cov_quota = 0
            print("Pair {}, Coverage quota {}/{}:  {:.3f}, S-reward {:.2f}".format(i, 0, 0, cov_quota, total_reward))
        # total_target_vol = total_vol * total_num / sample_num
        cov_input_idx_all[i].append((total_target_vol, cov_quota, total_reward))
        # Therefore, for each idx i, it consists of the cov_ratio for each bisection and the overall of splitting wrt i-th input feat.
    return cov_input_idx_all    

def calc_mc_coverage_multi_spec_pairwise_under(A_b_dict, dm_l_all, dm_u_all, spec_dim):
    torch.manual_seed(arguments.Config["general"]["seed"])
    dataset_tp = arguments.Config["data"]["dataset"]  
    sample_num = arguments.Config["preimage"]["sample_num"]
    label = arguments.Config["preimage"]["label"]
    pair_num = int(len(dm_l_all)/(spec_dim * 2))
    bisec_sample_num = int(sample_num / 2)
    if "Customized" in dataset_tp:
        model = load_model(weights_loaded=False)
    else:
        model = load_model()
    if dataset_tp == 'dubinsrejoin':
        label_T = arguments.Config["preimage"]["runner_up"] - 4
    else:
        label_T = None
    device = arguments.Config["general"]["device"]
    model = model.to(device)
    cov_input_idx_all = [[] for _ in range(pair_num)] 
    for i in range(pair_num):
        cov_num = 0
        total_num = 0
        total_target_vol = 0
        total_reward = 0
        for j in range(2):
            dm_l, dm_u = dm_l_all[2*i*spec_dim+j*spec_dim], dm_u_all[2*i*spec_dim+j*spec_dim]
        # dm_l_1, dm_u_1 = dm_l_all[2*i+1], dm_u_all[2*i+1]
        # if samples is None:
            # samples = np.random.uniform(low=dm_l,high=dm_u,size=(sub_sample_num, len(dm_l)))
        # else:
            # dm_vol = calc_total_sub_vol(dm_l, dm_u)
            # total_vol += dm_vol
            dm_vol = np.prod(dm_u.cpu().detach().numpy() - dm_l.cpu().detach().numpy())
            samples = Uniform(dm_l, dm_u).sample([bisec_sample_num])
            # tmp_samples = np.random.uniform(low=dm_l,high=dm_u,size=(bisec_sample_num, len(dm_l)))
            # data = torch.tensor(tmp_samples, dtype=torch.get_default_dtype())
            samples = samples.to(device)
            if dataset_tp == 'dubinsrejoin':
                prediction = model(samples)
                pred_R = prediction[:,:4]
                pred_T = prediction[:, 4:]
                pred_label_R = pred_R.argmax(dim=1).cpu().detach().numpy() 
                pred_label_T = pred_T.argmax(dim=1).cpu().detach().numpy()
                samples_idxs_R = np.where(pred_label_R == label)[0]
                samples_idxs_T = np.where(pred_label_T == label_T)[0]
                samples_eval_idxs = np.intersect1d(samples_idxs_R, samples_idxs_T)
            else:
                predicted = model(samples).argmax(dim=1).cpu().detach().numpy()    
                samples_eval_idxs = np.where(predicted==label)[0]
            target_num = len(samples_eval_idxs)
            target_vol = dm_vol * target_num / bisec_sample_num
            # cov_input_idx_all[i].append(target_vol)
            total_target_vol += target_vol
            if target_num > 0:
                samples_eval = samples[samples_eval_idxs]
                samples_eval = samples_eval.cpu().detach().numpy()
                if spec_dim == 1:
                    mat = A_b_dict['lA'][2*i+j]
                    bias = A_b_dict['lbias'][2*i+j]
                else:
                    mat = A_b_dict['lA'][2*i*spec_dim+j*spec_dim : 2*i*spec_dim+(j+1)*spec_dim]
                    bias = A_b_dict['lbias'][2*i*spec_dim+j*spec_dim : 2*i*spec_dim+(j+1)*spec_dim]
                    if dataset_tp != 'cartpole':
                        mat = np.squeeze(mat, axis=1)   
                print('Pair {}, subsection {}, mat shape: {}, samples_eval_T shape: {}'.format(i, j, mat.shape, samples_eval.T.shape))
                result = np.matmul(mat, samples_eval.T)+bias
                sub_tight_reward = calc_over_tightness(result, preimg_idx=None)
                if spec_dim > 1:
                    idxs_True = None
                    for s_dim in range(spec_dim):
                        idxs_tmp = set(np.where(result[s_dim]>=0)[0])
                        if idxs_True is None:
                            idxs_True = idxs_tmp
                        else:
                            idxs_True = idxs_True.intersection(idxs_tmp)
                else:
                    idxs_True = np.where(result>=0)[0]
                cov_input_idx_all[i].append((target_vol, len(idxs_True)/target_num, sub_tight_reward))
                cov_num += len(idxs_True)
                total_num += target_num
                total_reward += sub_tight_reward
            else:
                cov_input_idx_all[i].append((0,0,-2))
                print('Pair {}, subsection {}, No samples of NN on: dm_l {}, dm_u {}'.format(i, j, dm_l, dm_u))# In this case, the subdomain will not lead to the target label, no need for further branching. set the cov_quota as 1, uncov_vol will be 0
            # however, when evaluating the generall coverage volume, the coverage quota is not calculated as 1, instead making an impact by not adding to the total number
        if total_num > 0:
            cov_quota = cov_num / total_num
            print("Pair {}, Coverage quota {}/{}:  {:.3f}, S-reward {:.2f}".format(i, cov_num, total_num, cov_quota, total_reward))
        else:
            cov_quota = 0
            print("Pair {}, Coverage quota {}/{}:  {:.3f}, S-reward {:.2f}".format(i, 0, 0, cov_quota, total_reward))
        # total_target_vol = total_vol * total_num / sample_num
        cov_input_idx_all[i].append((total_target_vol, cov_quota, total_reward))
        # Therefore, for each idx i, it consists of the cov_ratio for each bisection and the overall of splitting wrt i-th input feat.
    return cov_input_idx_all


def sigmoid(z):
    return 1 / (1 + np.exp(-z))   

def calc_over_tightness(sample_res, preimg_idx): #A, b, samples
    if preimg_idx is None:
        res_min_spec = np.min(sample_res, axis=0)
    else:
        res_exact_preimg = sample_res[:, preimg_idx]
        res_min_spec = np.min(res_exact_preimg, axis=0)
    # res_min_spec = np.min(sample_res, axis=0)
    res_sigmoid = sigmoid(res_min_spec) 
    mean_res = np.mean(res_sigmoid)  
    return mean_res

def calc_input_coverage_initial_input_over(A_b_dict):
    torch.manual_seed(arguments.Config["general"]["seed"])
    dataset_tp = arguments.Config["data"]["dataset"]    
    sample_num = arguments.Config["preimage"]["sample_num"]
    label = arguments.Config["preimage"]["label"]
    # sample_dir = arguments.Config['preimage']["sample_dir"]
    # sample_path = os.path.join(sample_dir, 'sample_{}.pt'.format(dataset_tp))
    X, labels, data_max, data_min, perturb_epsilon = load_input_bounds(dataset_tp, label, quant=False, trans=False)
    # if not os.path.exists(sample_path):
    samples = Uniform(data_min, data_max).sample([sample_num])
    samples = torch.squeeze(samples, 1)
    # torch.save(samples, sample_path)
    # else:
    #     samples = torch.load(sample_path)
    if "Customized" in dataset_tp:
        model = load_model(weights_loaded=False)
    else:
        model = load_model()
    if dataset_tp == 'dubinsrejoin':
        label_T = arguments.Config["preimage"]["runner_up"]-4
    else:
        label_T = None
    dm_vol = calc_total_sub_vol(data_min, data_max)
    device = arguments.Config["general"]["device"]
    samples = samples.to(device)
    model = model.to(device)
    if dataset_tp == 'dubinsrejoin':
        prediction = model(samples)
        pred_R = prediction[:,:4]
        pred_T = prediction[:, 4:]
        pred_label_R = pred_R.argmax(dim=1).cpu().detach().numpy() 
        pred_label_T = pred_T.argmax(dim=1).cpu().detach().numpy()
        samples_idxs_R = np.where(pred_label_R == label)[0]
        samples_idxs_T = np.where(pred_label_T == label_T)[0]
        idxs = np.intersect1d(samples_idxs_R, samples_idxs_T)
    else:
        predicted = model(samples).argmax(dim=1).cpu().detach().numpy()
        # volume estimation for exact preimage 
        idxs = np.where(predicted==label)[0] 
    if len(idxs)>0:
        samples_eval = samples.cpu().detach().numpy()
        target_vol = dm_vol * len(idxs)/sample_num
        # print('Label: {}, Num: {}'.format(label, len(idxs)))   
        # samples_tmp = samples[idxs]
        mat = A_b_dict["uA"]
        bias = A_b_dict["ubias"]
        # print('mat shape: {}, sample_tmp_T shape: {}'.format(mat.shape, samples_tmp.T.shape))   
         
        if dataset_tp != 'cartpole':
            mat = np.squeeze(mat, axis=1)            
        result = np.matmul(mat, samples_eval.T)+bias
        spec_dim = result.shape[0]
        if spec_dim > 1:
            idxs_True = None
            for i in range(spec_dim):
                idxs_tmp = set(np.where(result[i]>=0)[0])
                if idxs_True is None:
                    idxs_True = idxs_tmp
                else:
                    idxs_True = idxs_True.intersection(idxs_tmp)
        else:
            idxs_True = np.where(result>=0)[0]
        cov_quota = len(idxs_True)/len(idxs)
        print("Coverage quota {}/{}:  {:.3f}".format(len(idxs_True), len(idxs), cov_quota))
    else:
        target_vol = 0
        cov_quota = 0
    return target_vol, cov_quota

def calc_input_coverage_initial_input_under(A_b_dict):
    torch.manual_seed(arguments.Config["general"]["seed"])
    dataset_tp = arguments.Config["data"]["dataset"]    
    sample_num = arguments.Config["preimage"]["sample_num"]
    label = arguments.Config["preimage"]["label"]
    # sample_dir = arguments.Config['preimage']["sample_dir"]
    # sample_path = os.path.join(sample_dir, 'sample_{}.pt'.format(dataset_tp))
    # if not os.path.exists(sample_path):
    
    X, labels, data_max, data_min, perturb_epsilon = load_input_bounds(dataset_tp, label, quant=False, trans=False)
    dm_vol = calc_total_sub_vol(data_min, data_max)
    # dm_vol = np.prod(data_max.cpu().detach().numpy() - data_min.cpu().detach().numpy())
    samples = Uniform(data_min, data_max).sample([sample_num])
    samples = torch.squeeze(samples, 1)
    # torch.save(samples, sample_path)
    # else:
    #     samples = torch.load(sample_path)
    if "Customized" in dataset_tp:
        model = load_model(weights_loaded=False)
    else:
        model = load_model()
    if dataset_tp == 'dubinsrejoin':
        label_T = arguments.Config["preimage"]["runner_up"]-4
    else:
        label_T = None
    device = arguments.Config["general"]["device"]
    samples = samples.to(device)
    model = model.to(device)
    if dataset_tp == 'dubinsrejoin':
        prediction = model(samples)
        pred_R = prediction[:,:4]
        pred_T = prediction[:, 4:]
        pred_label_R = pred_R.argmax(dim=1).cpu().detach().numpy() 
        pred_label_T = pred_T.argmax(dim=1).cpu().detach().numpy()
        samples_idxs_R = np.where(pred_label_R == label)[0]
        samples_idxs_T = np.where(pred_label_T == label_T)[0]
        idxs = np.intersect1d(samples_idxs_R, samples_idxs_T)
    else:
        predicted = model(samples).argmax(dim=1).cpu().detach().numpy()
        idxs = np.where(predicted==label)[0]
    if len(idxs)>0:
        target_vol = dm_vol * len(idxs)/sample_num
        print('Label: {}, Num: {}'.format(label, len(idxs)))
        # for i in range(output_num):    
        samples_tmp = samples[idxs]
        samples_tmp = samples_tmp.cpu().detach().numpy()
        mat = A_b_dict["lA"]
        bias = A_b_dict["lbias"]
        # print('mat shape: {}, sample_tmp_T shape: {}'.format(mat.shape, samples_tmp.T.shape))    
        if dataset_tp != 'cartpole':
            mat = np.squeeze(mat, axis=1)            
        result = np.matmul(mat, samples_tmp.T)+bias
        spec_dim = result.shape[0]
        if spec_dim > 1:
            idxs_True = None
            for i in range(spec_dim):
                idxs_tmp = set(np.where(result[i]>=0)[0])
                if idxs_True is None:
                    idxs_True = idxs_tmp
                else:
                    idxs_True = idxs_True.intersection(idxs_tmp)
        else:
            idxs_True = np.where(result>=0)[0]
        cov_quota = len(idxs_True)/len(idxs)
        print("Coverage quota {}/{}:  {:.3f}".format(len(idxs_True), len(idxs), cov_quota))
    else:
        target_vol = 0
        cov_quota = 0
    return target_vol, cov_quota
def is_inside_polytope(A, b, point):
    return np.all(np.dot(A, point) + b >= 0)
         
def calc_mc_esti_coverage_initial_input_under(A_b_dict):
    torch.manual_seed(arguments.Config["general"]["seed"])
    dataset_tp = arguments.Config["data"]["dataset"]    
    sample_num = arguments.Config["preimage"]["sample_num"]
    label = arguments.Config["preimage"]["label"]
    # sample_dir = arguments.Config['preimage']["sample_dir"]
    # sample_path = os.path.join(sample_dir, 'sample_{}.pt'.format(dataset_tp))
    # if not os.path.exists(sample_path):
    
    X, labels, data_max, data_min, perturb_epsilon = load_input_bounds(dataset_tp, label, quant=False, trans=False)
    # dm_vol = calc_total_sub_vol(data_min, data_max)
    dm_vol = np.prod(data_max.cpu().detach().numpy() - data_min.cpu().detach().numpy())
    samples = Uniform(data_min, data_max).sample([sample_num])
    samples = torch.squeeze(samples, 1)
    # torch.save(samples, sample_path)
    # else:
    #     samples = torch.load(sample_path)
    if "Customized" in dataset_tp:
        model = load_model(weights_loaded=False)
    else:
        model = load_model()
    if dataset_tp == 'dubinsrejoin':
        label_T = arguments.Config["preimage"]["runner_up"]-4
    else:
        label_T = None
    device = arguments.Config["general"]["device"]
    samples = samples.to(device)
    model = model.to(device)
    if dataset_tp == 'dubinsrejoin':
        prediction = model(samples)
        pred_R = prediction[:,:4]
        pred_T = prediction[:, 4:]
        pred_label_R = pred_R.argmax(dim=1).cpu().detach().numpy() 
        pred_label_T = pred_T.argmax(dim=1).cpu().detach().numpy()
        samples_idxs_R = np.where(pred_label_R == label)[0]
        samples_idxs_T = np.where(pred_label_T == label_T)[0]
        idxs = np.intersect1d(samples_idxs_R, samples_idxs_T)
    else:
        predicted = model(samples).argmax(dim=1).cpu().detach().numpy()
        idxs = np.where(predicted==label)[0]
        
    if len(idxs)>0:
        target_vol = dm_vol * len(idxs)/sample_num
        print('Label: {}, Num: {}'.format(label, len(idxs)))
        # for i in range(output_num):    
        samples_tmp = samples[idxs]
        samples_tmp = samples_tmp.cpu().detach().numpy()
        mat = A_b_dict["lA"]
        bias = A_b_dict["lbias"]
        # print('mat shape: {}, sample_tmp_T shape: {}'.format(mat.shape, samples_tmp.T.shape))    
        if dataset_tp != 'cartpole':
            mat = np.squeeze(mat, axis=1)          
        # count_inside = sum(is_inside_polytope(mat, bias, point) for point in samples_tmp)  
        result = np.matmul(mat, samples_tmp.T)+bias
        spec_dim = result.shape[0]
        if spec_dim > 1:
            idxs_True = None
            for i in range(spec_dim):
                idxs_tmp = set(np.where(result[i]>=0)[0])
                if idxs_True is None:
                    idxs_True = idxs_tmp
                else:
                    idxs_True = idxs_True.intersection(idxs_tmp)
        else:
            idxs_True = np.where(result>=0)[0]
        cov_quota = len(idxs_True) / len(idxs)
        print("Coverage quota {}/{}:  {:.3f}".format(len(idxs_True), len(idxs), cov_quota))
    else:
        target_vol = 0
        cov_quota = 0
    return target_vol, cov_quota

# Note: deprecated
# def calc_input_coverage_single_label(A_b_dict, args):
#     # args = get_args()
#     data_lb, data_ub, output_num = load_input_bounds_numpy(args.dataset, args.quant, args.trans)
#     dm_vol = calc_total_sub_vol(data_lb, data_ub)
#     sample_num = args.sample_num
#     np.random.seed(0)
#     samples = np.random.uniform(low=data_lb, high=data_ub, size=(sample_num, len(data_lb)))
#     # print(samples)
#     if args.depth:
#         MODEL_DIR = "/home/xiyue/LinInv/model_dir/VCAS_8_{}".format(args.hidden_layer_num)
#     elif args.width:
#         MODEL_DIR = "/home/xiyue/LinInv/model_dir/VCAS_{}".format(args.hidden_dim)
#     else:
#         if args.dataset == 'vcas':
#             MODEL_DIR = "/home/xiyue/LinInv/model_dir/VCAS_21"        
#         else:
#             MODEL_DIR = "/home/xiyue/LinInv/model_dir/"
#     model_path = args.model
#     # if args.dataset == 'vcas' or args.dataset == 'cartpole':
#     if model_path[-4:] == 'onnx':
#         model_path = os.path.join(MODEL_DIR,model_path[:-4]+'pt')
#         model = torch.load(model_path)
#         # print('check model', model)
#         model.eval()
#     else:
#         model_path = os.path.join(MODEL_DIR, args.model)
#         if 'auto_park' in args.dataset:
#             model_info = {'hidden_dim': args.hidden_dim, 'hidden_layer_num': args.hidden_layer_num}
#         else:
#             model_info = None
#         if args.dataset == 'demo':
#             model = load_model_simple(args.model_name, model_path, model_info, weights_loaded=False)
#         else:
#             model = load_model_simple(args.model_name, model_path, model_info, weights_loaded=True)
#     # data = torch.tensor(samples,).float()
#     data = torch.tensor(samples, dtype=torch.get_default_dtype())
#     predicted = model(data).argmax(dim=1)
#     # print(predicted)
#     # samples_idx_dict = dict()
#     # for i in range(output_num):
#     idxs = np.where(predicted==args.label)[0]
#         # samples_idx_dict[i] = idxs
#     if len(idxs)>0:
#         target_vol = dm_vol * len(idxs)/sample_num
#         print('Label: {}, Num: {}'.format(args.label, len(idxs)))
#         # for i in range(output_num):    
#         samples_tmp = samples[idxs]
#         # Already multiplied with C
#         mat = A_b_dict["lA"]
#         bias = A_b_dict["lbias"]
#         # print('mat shape: {}, sample_tmp_T shape: {}'.format(mat.shape, samples_tmp.T.shape))    
#         result = np.matmul(mat, samples_tmp.T)+bias
#         spec_dim = result.shape[0]
#         if spec_dim > 1:
#             idxs_True = None
#             for i in range(spec_dim):
#                 idxs_tmp = set(np.where(result[i]>=0)[0])
#                 if idxs_True is None:
#                     idxs_True = idxs_tmp
#                 else:
#                     idxs_True = idxs_True.intersection(idxs_tmp)
#         else:
#             idxs_True = np.where(result>=0)[0]
#         cov_quota = len(idxs_True)/len(idxs)
#         print("Coverage quota {}/{}:  {:.3f}".format(len(idxs_True), len(idxs), cov_quota))
        
#         # if args.save_process:
#         #     A_b_dict['dm_l'] = data_lb 
#         #     A_b_dict['dm_u'] = data_ub
#         #     save_path = os.path.join(args.result_dir, 'run_example')
#         #     save_file = os.path.join(save_path,'{}_spec_{}_init_poly_{}'.format(args.dataset, args.label, args.base))
#         #     with open(save_file, 'wb') as f:
#         #         pickle.dump(A_b_dict, f)            
#     # idxs_False = np.where(result<0)[0]
#     # print("Sample points not included", samples_tmp[idxs_False][:5])
#     else:
#         target_vol = 0
#         cov_quota = 0
#     return target_vol, cov_quota



    
def calc_Hrep_coverage(A_b_dict, args):
    data_lb, data_ub, output_num = load_input_bounds_numpy(args.dataset, args.quant, args.trans)
    sample_num = args.sample_num
    samples = np.random.uniform(low=data_lb, high=data_ub, size=(sample_num, len(data_lb)))
    # print(samples)
    model_path = args.model
    ext = model_path.split('.')[-1]
    if ext == 'pt':
        model = load_model_simple(args.model_name, args.model, weights_loaded=True)
    elif ext == 'onnx':
        onnx_folder = "/home/xiyue/vcas-code/acas/networks/onnx"
        onnx_path = os.path.join(onnx_folder, model_path)
        # xy: test onnx_path and vnnlib_path
        # onnx_path = "../vnncomp2021/benchmarks/cifar10_resnet/onnx/resnet_2b.onnx"
        # vnnlib_path = "../vnncomp2021/benchmarks/cifar10_resnet/vnnlib_properties_pgd_filtered/resnet2b_pgd_filtered/prop_10_eps_0.008.vnnlib"
        # vnnlib_path = None
        from preimage_utils import load_model_onnx_simple
        model = load_model_onnx_simple(onnx_path)
        # shape = (-1, *onnx_shape) 
        
    # data = torch.tensor(samples,).float()
    data = torch.tensor(samples, dtype=torch.get_default_dtype())
    predicted = model(data).argmax(dim=1)
    # print(predicted)
    cov_quota_dict = dict()
    samples_idx_dict = dict()
    for i in range(output_num):
        idxs = np.where(predicted==i)[0]
        samples_idx_dict[i] = idxs
        print('Label: {}, Num: {}'.format(i, len(idxs)))
    for i in range(output_num):    
        samples_tmp = samples[samples_idx_dict[i]]
        mat = A_b_dict[i][0]
        bias = A_b_dict[i][1]
        print('mat shape: {}, sample_tmp_T shape: {}'.format(mat.shape, samples_tmp.T.shape))
        result = np.matmul(mat, samples_tmp.T).T+bias #[np.newaxis,:]
        idxs_True = np.where(result>=0)[0]
        if len(samples_tmp) > 0:
            cov_quota = len(idxs_True)/len(samples_idx_dict[i])
            print("Coverage quota {}/{}:  {:.3f}".format(len(idxs_True), len(samples_idx_dict[i]), cov_quota))
            cov_quota_dict[i] = cov_quota
            idxs_False = np.where(result<0)[0]
            print("Sample points not included", samples_tmp[idxs_False][:5])
        else:
            cov_quota_dict[i] = 2
    return cov_quota_dict

def build_cdd(linear_param_dict):
    '''
    This will return the H-representation for each specific label
    '''
    for k, v in linear_param_dict.items():
        print("Param: {}, Shape: {}".format(k, v.shape))
    lA = linear_param_dict["lA"]
    uA = linear_param_dict["uA"]
    lbias = linear_param_dict["lbias"]
    ubias = linear_param_dict["ubias"]
    output_dim = lbias.shape[0]
    # cdd requires a specific H-representation format
    cdd_mat_dict = dict()
    for i in range(output_dim):
        # calculate A of cdd mat for class i
        tA = np.reshape(lA[i], [1, -1])
        tA_rep = np.repeat(tA, output_dim-1, axis=0)
        uA_del = np.delete(uA, i, axis=0)
        # print(tA_rep.shape, uA_del.shape)
        assert (tA_rep.shape == uA_del.shape)
        polyA = tA_rep - uA_del
        # calculate b of cdd mat for class i
        tbias_rep = np.repeat(lbias[i], output_dim-1)
        ubias_del = np.delete(ubias, i)
        # print(tbias_rep.shape, ubias_del.shape)
        assert (tbias_rep.shape == ubias_del.shape)
        polyb = tbias_rep - ubias_del
        cdd_mat = np.column_stack((polyb, polyA))
        print('check cdd', cdd_mat.shape, cdd_mat)
        cdd_mat_dict[i] = cdd_mat.tolist()
    return cdd_mat_dict


def test_multi_label_constraint(args):
    A_dict_path = os.path.join(args.result_dir,"vcas_1_init")
    A_dict = load_A_dict(A_dict_path)
    calc_Hrep_coverage(A_dict, args)
    
    
