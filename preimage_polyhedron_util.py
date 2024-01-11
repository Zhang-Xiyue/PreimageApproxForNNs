import torch
import pickle
import os
import numpy as np
from collections import defaultdict
from preimage_model_utils import load_input_bounds_numpy, load_model_simple



def load_A_dict(A_file):
    with open(A_file, 'rb') as f:
        linear_rep_dict = pickle.load(f)
    return linear_rep_dict


def save_A_dict(A, A_path, A_dir='./results'):
    # A_file= os.path.join(RES_DIR, 'A_dict_BaB_init')
    if not os.path.exists(A_dir):
        os.makedirs(A_dir)
    A_file= os.path.join(A_dir, A_path)
    with open(A_file, 'wb') as f:
        pickle.dump(A, f)
        
def post_process_A(A_dict):
    linear_rep_dict = dict()
    for output_node, linear_rep in A_dict.items():
        for input_node, param_dict in linear_rep.items():
            for k, v in param_dict.items():
                # if v is not None: -- only care about the linear representation
                if k == 'lA':
                    linear_rep_dict[k] = torch.squeeze(v).cpu().detach().numpy()
                elif k == 'lbias':
                    # NOTE: we dont squeeze bias as it can help with the batch addition
                    linear_rep_dict[k] = v.cpu().detach().numpy()
    return linear_rep_dict

def post_process_multiple_A(A_dict):
    # linear_rep_dict_multi.setdefault()
    linear_rep_dict_multi = defaultdict(list)
    for output_node, linear_rep in A_dict.items():
        for input_node, param_dict in linear_rep.items():
            for k, v in param_dict.items():
                if k == 'lA':
                    for i in range(v.shape[0]):
                        linear_rep_dict_multi[k].append(torch.squeeze(v[i]).cpu().detach().numpy())
                elif k == 'lbias':
                    for i in range(v.shape[0]):
                        bias = v[i].cpu().detach().numpy()
                        linear_rep_dict_multi[k].append(bias[0])
    # print(linear_rep_dict_multi)
    return linear_rep_dict_multi
# def save_linear_rep_dict(A, A_path, args):
#     linear_rep_dict = post_process_A(A)
#     # if args is not None:
#     #     save_path = '{}_{}_{}_label_{}'.format(args.model_name, args.dataset, args.base, args.label)
#     # save_path = os.path.join(args.result_dir, save_path)
#     with open(A_path, "wb") as f:
#         pickle.dump(linear_rep_dict, f)
#     return linear_rep_dict

def calc_refine_under_approx_coverage(A, dm_l_all, dm_u_all, spec_dim, args):
    dm_l_all_np = dm_l_all.cpu().detach().numpy()
    dm_u_all_np = dm_u_all.cpu().detach().numpy()
    # print(dm_l_all_np, dm_u_all_np)
    A_b_dict_multi = post_process_multiple_A(A)
    # now changed it to pairwise one, for preset branching depth, use multi
    # cov_quota = calc_Hrep_coverage_multi(A_b_dict_multi, dm_l_all_np, dm_u_all_np)
    # List of list (3 elements)
    cov_input_idx_all = calc_Hrep_coverage_multi_spec_pairwise(A_b_dict_multi, dm_l_all_np, dm_u_all_np, spec_dim, args)
    return cov_input_idx_all, A_b_dict_multi


    # if A is None:
    #     # This if branch is for debugging
    #     dm_l_all = torch.load(os.path.join(RES_DIR, 'dm_l_tensor_level_{}.pt'.format(split_depth)))
    #     dm_u_all = torch.load(os.path.join(RES_DIR, 'dm_u_tensor_level_{}.pt'.format(split_depth)))
    #     dm_l_all_np = dm_l_all.cpu().detach().numpy()
    #     dm_u_all_np = dm_u_all.cpu().detach().numpy()
    #     print(dm_l_all_np, dm_u_all_np)
    #     A_file= os.path.join(RES_DIR, 'A_dict_BaB_init_level_{}'.format(split_depth))
    #     A = load_A_dict(A_file)
    #     A_b_dict_multi = post_process_multiple_A(A)
    #     # sample = [0.3, 0.5] 
    #     # idx = check_subdomain(sample, dm_l_all_np, dm_u_all_np)
    #     # print(idx)
    #     cov_quota = calc_Hrep_coverage_multi(A_b_dict_multi, dm_l_all_np, dm_u_all_np)
    # else:
# def check_subdomain(sample, dm_l_all, dm_u_all):
#     for idx in range(len(dm_l_all)):
#         dm_l = dm_l_all[idx]
#         dm_u = dm_u_all[idx]
#         dec = True
#         for i in range(len(sample)):
#             dec = dec and (sample[i] >= dm_l[i]) and (sample[i] <= dm_u[i])
#         if dec == True:
#             break
#     return idx
    
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


# Calculate the subregion volume    
def calc_total_sub_vol(dm_l, dm_u):
    total_sub_vol = 1
    in_dim = dm_l.shape[-1]
    dm_shape = dm_l.shape
    # print("check dm_l, dm_u shape", dm_shape)
    assert len(dm_shape) == 2 or len(dm_shape) == 1
    if len(dm_shape) == 2:
        for i in range(in_dim):
            total_sub_vol = total_sub_vol * (dm_u[0][i] - dm_l[0][i])
    elif len(dm_shape) == 1:
        for i in range(in_dim):
            total_sub_vol = total_sub_vol * (dm_u[i] - dm_l[i])
    return total_sub_vol

# In the all potential feature split case, we need the cov_quota for each pairwise subdomain, not the overall for all domains
def calc_Hrep_coverage_multi_spec_pairwise(A_b_dict, dm_l_all, dm_u_all, spec_dim, args):
    # args = get_args()
    np.random.seed(0)
    sample_num = args.sample_num
    pair_num = int(len(dm_l_all)/(spec_dim * 2))
    bisec_sample_num = int(sample_num / 2)
    # samples = None
    if args.depth:
        MODEL_DIR = "./model_dir/VCAS_8_{}".format(args.hidden_layer_num)
    elif args.width:
        MODEL_DIR = "./model_dir/VCAS_{}".format(args.hidden_dim)
    else:
        if args.dataset == 'vcas':
            MODEL_DIR = "./model_dir/VCAS_21"        
        else:
            MODEL_DIR = "./model_dir/"
    model_path = args.model
    # if args.dataset == 'vcas':
    if model_path[-4:] == 'onnx':
        model_path = os.path.join(MODEL_DIR,model_path[:-4]+'pt')
        model = torch.load(model_path)
        # print('check model', model)
        model.eval()
    else:
        model_path = os.path.join(MODEL_DIR, args.model)
        if 'auto_park' in args.dataset:
            model_info = {'hidden_dim': args.hidden_dim, 'hidden_layer_num': args.hidden_layer_num}
        model = load_model_simple(args.model_name, model_path, model_info, weights_loaded=True) 
        
    # samples = defaultdict(list)
    # samples = dict()
    cov_input_idx_all = [[] for _ in range(pair_num)] 
    for i in range(pair_num):
        cov_num = 0
        total_num = 0
        total_target_vol = 0
        for j in range(2):
            dm_l, dm_u = dm_l_all[2*i*spec_dim+j*spec_dim], dm_u_all[2*i*spec_dim+j*spec_dim]
        # dm_l_1, dm_u_1 = dm_l_all[2*i+1], dm_u_all[2*i+1]
        # if samples is None:
            # samples = np.random.uniform(low=dm_l,high=dm_u,size=(sub_sample_num, len(dm_l)))
        # else:
            dm_vol = calc_total_sub_vol(dm_l, dm_u)
            # total_vol += dm_vol
            tmp_samples = np.random.uniform(low=dm_l,high=dm_u,size=(bisec_sample_num, len(dm_l)))
            data = torch.tensor(tmp_samples, dtype=torch.get_default_dtype())
            predicted = model(data).argmax(dim=1)    
            tmp_samples_eval_idxs = np.where(predicted==args.label)[0]
            target_num = len(tmp_samples_eval_idxs)
            target_vol = dm_vol * target_num / bisec_sample_num
            # cov_input_idx_all[i].append(target_vol)
            total_target_vol += target_vol
            if target_num > 0:
                samples_eval = tmp_samples[tmp_samples_eval_idxs]
                if spec_dim == 1:
                    mat = A_b_dict['lA'][2*i+j]
                    bias = A_b_dict['lbias'][2*i+j]
                else:
                    mat = np.vstack(A_b_dict['lA'][2*i*spec_dim+j*spec_dim : 2*i*spec_dim+(j+1)*spec_dim])
                    bias = np.vstack(A_b_dict['lbias'][2*i*spec_dim+j*spec_dim : 2*i*spec_dim+(j+1)*spec_dim])
                print('Pair {}, subsection {}, mat shape: {}, samples_eval_T shape: {}'.format(i, j, mat.shape, samples_eval.T.shape))
                result = np.matmul(mat, samples_eval.T)+bias
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
                cov_input_idx_all[i].append((target_vol, len(idxs_True)/target_num))
                
                cov_num += len(idxs_True)
                total_num += target_num
            else:
                cov_input_idx_all[i].append((0,0))
                print('Pair {}, subsection {}, No samples of NN on: dm_l {}, dm_u {}'.format(i, j, dm_l, dm_u))# In this case, the subdomain will not lead to the target label, no need for further branching. set the cov_quota as 1, uncov_vol will be 0
            # however, when evaluating the generall coverage volume, the coverage quota is not calculated as 1, instead making an impact by not adding to the total number
        if total_num > 0:
            cov_quota = cov_num / total_num
            print("Pair {}, Coverage quota {}/{}:  {:.3f}".format(i, cov_num, total_num, cov_quota))
        else:
            cov_quota = 0
            print("Pair {}, Coverage quota {}/{}:  {:.3f}".format(i, 0, 0, cov_quota))
        # total_target_vol = total_vol * total_num / sample_num
        cov_input_idx_all[i].append((total_target_vol,cov_quota))
        # Therefore, for each idx i, it consists of the cov_ratio for each bisection and the overall of splitting wrt i-th input feat.
    # if args.save_process:
    #     select_idx = -1
    #     max_cov_vol = -1
    #     for i, cov_info in enumerate(cov_input_idx_all):
    #         if cov_info[0]*cov_info[1] > max_cov_vol:
    #             select_idx = i
    #             max_cov_vol = cov_info[0]*cov_info[1]

    #     save_path = os.path.join(args.result_dir, 'run_example')
    #     save_file = os.path.join(save_path,'{}_spec_{}_iter_{}'.format(args.dataset, args.label, num_iter))
        # with open(save_file, 'wb') as f:
        #     pickle.dump((preimage_dict_all, dm_rec_all), f)
    return cov_input_idx_all        



         



def calc_input_coverage_single_label(A_b_dict, args):
    # args = get_args()
    data_lb, data_ub, output_num = load_input_bounds_numpy(args.dataset, args.quant)
    dm_vol = calc_total_sub_vol(data_lb, data_ub)
    sample_num = args.sample_num
    np.random.seed(0)
    samples = np.random.uniform(low=data_lb, high=data_ub, size=(sample_num, len(data_lb)))
    # print(samples)
    if args.depth:
        MODEL_DIR = "./model_dir/VCAS_8_{}".format(args.hidden_layer_num)
    elif args.width:
        MODEL_DIR = "./model_dir/VCAS_{}".format(args.hidden_dim)
    else:
        if args.dataset == 'vcas':
            MODEL_DIR = "./model_dir/VCAS_21"        
        else:
            MODEL_DIR = "./model_dir/"
    model_path = args.model
    # if args.dataset == 'vcas' or args.dataset == 'cartpole':
    if model_path[-4:] == 'onnx':
        model_path = os.path.join(MODEL_DIR,model_path[:-4]+'pt')
        model = torch.load(model_path)
        # print('check model', model)
        model.eval()
    else:
        model_path = os.path.join(MODEL_DIR, args.model)
        if 'auto_park' in args.dataset:
            model_info = {'hidden_dim': args.hidden_dim, 'hidden_layer_num': args.hidden_layer_num}
        model = load_model_simple(args.model_name, model_path, model_info, weights_loaded=True)
    # data = torch.tensor(samples,).float()
    data = torch.tensor(samples, dtype=torch.get_default_dtype())
    predicted = model(data).argmax(dim=1)
    # print(predicted)
    # samples_idx_dict = dict()
    # for i in range(output_num):
    idxs = np.where(predicted==args.label)[0]
        # samples_idx_dict[i] = idxs
    if len(idxs)>0:
        target_vol = dm_vol * len(idxs)/sample_num
        print('Label: {}, Num: {}'.format(args.label, len(idxs)))
        # for i in range(output_num):    
        samples_tmp = samples[idxs]
        # Already multiplied with C
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
        
        # if args.save_process:
        #     A_b_dict['dm_l'] = data_lb 
        #     A_b_dict['dm_u'] = data_ub
        #     save_path = os.path.join(args.result_dir, 'run_example')
        #     save_file = os.path.join(save_path,'{}_spec_{}_init_poly'.format(args.dataset, args.label))
        #     with open(save_file, 'wb') as f:
        #         pickle.dump(A_b_dict, f)            
    # idxs_False = np.where(result<0)[0]
    # print("Sample points not included", samples_tmp[idxs_False][:5])
    else:
        target_vol = 0
        cov_quota = 0
    return target_vol, cov_quota



    

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
    
    
# if __name__ == "__main__":
#     args = get_args()
#     # calc_refine_under_approx_coverage(A=None, dm_l_all=None, dm_u_all=None)
#     test_multi_label_constraint(args)