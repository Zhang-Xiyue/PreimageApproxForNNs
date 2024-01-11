##################################################################################
## This file is the main entrance for preimage approximation of neural networks ##
##################################################################################

import socket
import random
import os
import sys
from pathlib import Path
cwd = Path.cwd()
if str(cwd).endswith("PreimgApprox"):
    sys_path =  os.path.join(str(cwd), 'alpha-beta-CROWN')
    sys_path2 = os.path.join(str(cwd), 'alpha-beta-CROWN/complete_verifier')
sys.path.append(sys_path)
sys.path.append(sys_path2)

import time
import gc
import torch
import numpy as np

from preimage_parse_args import get_args
import preimage_arguments
from auto_LiRPA import BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from jit_precompile import precompile_jit_kernels
# Model wrapper file
from preimage_CROWN_solver import LiRPAConvNet

from preimage_utils import construct_vnnlib, load_model_onnx_simple
from preimage_model_utils import load_input_info, load_model_simple
from preimage_approx_batch_input_split import input_approx_parallel_multi, input_approx_parallel_multi_no_prioritization
from preimage_read_vnnlib import batch_vnnlib

def bab_preimage(args, unwrapped_model, data, targets, y, data_ub, data_lb,
        lower_bounds=None, upper_bounds=None, reference_slopes=None,
        attack_images=None, c=None, all_prop=None, cplex_processes=None,
        activation_opt_params=None, reference_lA=None, rhs=None, 
        model_incomplete=None, timeout=None, refined_betas=None):

    norm = preimage_arguments.Config["specification"]["norm"]
    eps = preimage_arguments.Globals["lp_perturbation_eps"]  # epsilon for non Linf perturbations, None for all other cases.
    if norm != float("inf"):
        # For non Linf norm, upper and lower bounds do not make sense, and they should be set to the same.
        assert torch.allclose(data_ub, data_lb)

    model = LiRPAConvNet(unwrapped_model, in_size=data.shape if not targets.size > 1 else [len(targets)] + list(data.shape[1:]),
                         c=c, cplex_processes=cplex_processes)

    data = data.to(model.device)
    data_lb, data_ub = data_lb.to(model.device), data_ub.to(model.device)
    output = model.net(data).flatten()

    
    if preimage_arguments.Config['attack']['check_clean']:
        clean_rhs = c.matmul(output)
        print(f'Clean RHS: {clean_rhs}')
        if (clean_rhs < rhs).any():
            return -np.inf, np.inf, None, None, 'unsafe'

    ptb = PerturbationLpNorm(norm=norm, eps=eps, x_L=data_lb, x_U=data_ub)
    x = BoundedTensor(data, ptb).to(data_lb.device)
    domain = torch.stack([data_lb.squeeze(0), data_ub.squeeze(0)], dim=-1)

    cut_enabled = preimage_arguments.Config["bab"]["cut"]["enabled"]

    if cut_enabled:
        model.set_cuts(model_incomplete.A_saved, x, lower_bounds, upper_bounds)


    if args.prioritize:
        covered, dm_record, preimage_dict, nb_visited, cov_dm, time_cost, iter_cov_quota, subdomain_num = input_approx_parallel_multi(
            model, domain, x, args, model_ori=unwrapped_model, all_prop=all_prop,
            rhs=rhs, timeout=timeout, branching_method=preimage_arguments.Config["bab"]["branching"]["method"])
    else:
        covered, dm_record, preimage_dict, nb_visited, cov_dm, time_cost, iter_cov_quota, subdomain_num = input_approx_parallel_multi_no_prioritization(
            model, domain, x, args, model_ori=unwrapped_model, all_prop=all_prop,
            rhs=rhs, timeout=timeout, branching_method=preimage_arguments.Config["bab"]["branching"]["method"])            


    
    return covered, dm_record, preimage_dict, nb_visited, cov_dm, time_cost, iter_cov_quota, subdomain_num



def update_parameters(model, data_min, data_max):
    if 'vggnet16_2022' in preimage_arguments.Config['general']['root_path']:
        perturbed = (data_max - data_min > 0).sum()
        if perturbed > 10000:
            print('WARNING: prioritizing attack due to too many perturbed pixels on VGG')
            print('Setting test_arguments.Config["attack"]["pgd_order"] to "before"')
            preimage_arguments.Config['attack']['pgd_order'] = 'before'


def sort_targets_cls(batched_vnnlib, init_global_lb, init_global_ub, scores, reference_slopes, lA, final_node_name, reverse=False):

    assert len(batched_vnnlib) == init_global_lb.shape[0] and init_global_lb.shape[1] == 1
    sorted_idx = scores.argsort(descending=reverse)
    batched_vnnlib = [batched_vnnlib[i] for i in sorted_idx]
    init_global_lb = init_global_lb[sorted_idx]
    init_global_ub = init_global_ub[sorted_idx]

    if reference_slopes is not None:
        for m, spec_dict in reference_slopes.items():
            for spec in spec_dict:
                if spec == final_node_name:
                    if spec_dict[spec].size()[1] > 1:
                        # correspond to multi-x case
                        spec_dict[spec] = spec_dict[spec][:, sorted_idx]
                    else:
                        spec_dict[spec] = spec_dict[spec][:, :, sorted_idx]

    if lA is not None:
        lA = [lAitem[:, sorted_idx] for lAitem in lA]

    return batched_vnnlib, init_global_lb, init_global_ub, lA, sorted_idx


def verify_workflow(
        args, model_ori, model_incomplete, batched_vnnlib, vnnlib, vnnlib_shape,
        init_global_lb, lower_bounds, upper_bounds, index,
        timeout_threshold, bab_ret=None, lA=None, cplex_processes=None,
        reference_slopes=None, activation_opt_params=None, refined_betas=None, attack_images=None,
        attack_margins=None):
    start_time = time.time()
    cplex_cuts = preimage_arguments.Config["bab"]["cut"]["enabled"] and preimage_arguments.Config["bab"]["cut"]["cplex_cuts"]
    sort_targets = preimage_arguments.Config["bab"]["sort_targets"]
    bab_attack_enabled = preimage_arguments.Config["bab"]["attack"]["enabled"]


    for property_idx, properties in enumerate(batched_vnnlib):  # loop of x

        print(f'\nProperties batch {property_idx}, size {len(properties[0])}')

        start_time_bab = time.time()

        x_range = torch.tensor(properties[0], dtype=torch.get_default_dtype())
        data_min = x_range.select(-1, 0).reshape(vnnlib_shape)
        data_max = x_range.select(-1, 1).reshape(vnnlib_shape)
        x = x_range.mean(-1).reshape(vnnlib_shape)  # only the shape of x is important.

        target_label_arrays = list(properties[1])  # properties[1]: (c, rhs, y, pidx)

        print("check length of target label arrays", len(target_label_arrays))
        c, rhs, y, pidx = target_label_arrays[0]


        this_spec_attack_images = None
 

        print('##### Instance {} first 10 spec matrices: {}\nthresholds: {} ######'.format(index, c[:10],  rhs.flatten()[:10]))

        if np.array(pidx == y).any():
            raise NotImplementedError

        torch.cuda.empty_cache()
        gc.collect()

        c = torch.tensor(c, dtype=torch.get_default_dtype(), device=preimage_arguments.Config["general"]["device"])
        rhs = torch.tensor(rhs, dtype=torch.get_default_dtype(), device=preimage_arguments.Config["general"]["device"])

        # extract cplex cut filename
        if cplex_cuts:
            assert preimage_arguments.Config["bab"]["initial_max_domains"] == 1


        assert preimage_arguments.Config["general"]["complete_verifier"] == "bab"  # for MIP and BaB-Refine.
        assert not preimage_arguments.Config["bab"]["attack"]["enabled"], "BaB-attack must be used with incomplete verifier."
        # input split also goes here directly
        covered, dm_record, preimage_dict, nb_visited, cov_dm, time_cost, iter_cov_quota, subdomain_num = bab_preimage(
            args, model_ori, x, pidx, y, data_ub=data_max, data_lb=data_min, c=c,
            all_prop=target_label_arrays, cplex_processes=cplex_processes,
            rhs=rhs, timeout=timeout_threshold, attack_images=this_spec_attack_images)
        print("#Subdomain: {}, \n dm_record: {}, \n Coverage: {}, \n Time cost: {}".format(subdomain_num, dm_record, cov_dm, time_cost))
        

    return subdomain_num, time_cost, iter_cov_quota




def process_lA(poly_A_b):
    """transform polyA, polyb into lA format
    return lA in tensor format
    Args:
        poly_A_b (_tuple_): a tuple of (polyA, polyb) for a label 
    """
    polyA, polyb = poly_A_b
    lA = dict()
    lA.update({"lA": torch.from_numpy(polyA), 
               "lbias": torch.from_numpy(polyb)})
    return lA
    

def main(args=None):
    # Crown arguments set to default. Use args to update necessary argument dict
    preimage_arguments.Config.parse_config() # Crown default arguments
    assert args is not None

    MODEL_DIR = os.path.join(str(cwd), 'model_dir')
    if not os.path.exists(MODEL_DIR):
        print('Required model dir does not exist')
        return
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    model_name = args.model_name
    model_path = os.path.join(MODEL_DIR, args.model)
    dataset_tp = args.dataset
    vcas_idx = args.vcas
    ext = model_path.split('.')[-1] # model formats identified by file extension
    preimage_arguments.Config["data"]["num_outputs"] = args.output_dim
    preimage_arguments.Config["solver"]["bound_prop_method"] = args.base
    preimage_arguments.Config["bab"]["initial_max_domains"] = args.initial_max_domains
    if ext == 'pt':
        preimage_arguments.Config['model']['name'] = model_name
        preimage_arguments.Config["model"]["path"] = model_path
        preimage_arguments.Config["model"]["onnx_path"] = None
    elif ext == 'onnx':
        preimage_arguments.Config["model"]["onnx_path"] = model_path
        preimage_arguments.Config["model"]["path"] = None
    else:
        print("The model format is not supported. Convert to pytorch or onnx.")
        return 
    print(f'Model name {model_name}, dataset {dataset_tp}, num of outputs {args.output_dim}, label {args.label}, bound prop method {args.base}')

    print(f'Experiments at {time.ctime()} on {socket.gethostname()}')
    # set seed 
    torch.manual_seed(preimage_arguments.Config["general"]["seed"])
    random.seed(preimage_arguments.Config["general"]["seed"])
    np.random.seed(preimage_arguments.Config["general"]["seed"])
    torch.set_printoptions(precision=8)
    device = preimage_arguments.Config["general"]["device"]
    if device != 'cpu':
        torch.cuda.manual_seed_all(preimage_arguments.Config["general"]["seed"])
        # Always disable TF32 (precision is too low for verification).
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    if preimage_arguments.Config["general"]["deterministic"]:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    if preimage_arguments.Config["general"]["double_fp"]:
        torch.set_default_dtype(torch.float64)
    if preimage_arguments.Config["general"]["precompile_jit"]:
        precompile_jit_kernels()

    
    print('--- Set for Preimage Approximation ---')
    # Load Pytorch or ONNX model depends on the model path or onnx_path is given. 
    if ext == 'pt':
        if 'auto_park' in args.dataset: # different NNs for auto park case
            model_info = {'hidden_dim': args.hidden_dim, 'hidden_layer_num': args.hidden_layer_num}
        model_ori = load_model_simple(model_name, model_path, model_info, weights_loaded=True)
    elif ext == 'onnx':
        model_ori  = load_model_onnx_simple(model_path)    
    # always use the bound type and inf norm for specification
    assert preimage_arguments.Config["specification"]["type"] == 'bound'
    assert preimage_arguments.Config["specification"]["norm"] == float("inf")

    print(f'Target label {args.label} is used to construct c')
    X, labels, runnerup, data_max, data_min, perturb_epsilon, target_label = load_input_info(dataset_tp, args.label, args.quant)
    # print(f'Loaded datasets with per-element lower and upper bounds: max = {data_max.item()}, min = {data_min.item()}')
    if data_max.size(0) != X.size(0) or data_min.size(0) != X.size(0):
        raise ValueError("For 'bound' type specification, need per example lower and upper bounds.")
    example_idx_list = list(range(X.shape[0]))
    print('Task length:', len(example_idx_list))
    # The backward initailized mapping C is generated through construct_vnnlib; 
    vnnlib_all = construct_vnnlib(X, labels, runnerup, data_max, data_min, perturb_epsilon, target_label, example_idx_list, dataset_tp)

    vnnlib_shape = [-1] + list(X.shape[1:])


    model_ori = model_ori.to(device)
    model_ori.eval()

    

    bab_ret = []
    new_idx = 0
    vnnlib = vnnlib_all[new_idx]

    x_range = torch.tensor(vnnlib[0][0], dtype=torch.get_default_dtype())
    data_min = x_range.select(-1, 0).reshape(vnnlib_shape)
    data_max = x_range.select(-1, 1).reshape(vnnlib_shape)
    x = x_range.mean(-1).reshape(vnnlib_shape)  # only the shape of x is important.
    
    x, data_max, data_min = x.to(device), data_max.to(device), data_min.to(device)
    # print('x type', x.type())



    attack_margins = all_adv_candidates = None

    init_global_lb = saved_bounds = saved_slopes = y = lower_bounds = upper_bounds = None
    activation_opt_params = model_incomplete = lA = cplex_processes = None
    
    refined_betas = None
    

    timeout_threshold = preimage_arguments.Config["bab"]["timeout"]

    batched_vnnlib = batch_vnnlib(vnnlib)

    subdomain_num, time_cost, iter_cov_quota = verify_workflow(
        args, model_ori, model_incomplete, batched_vnnlib, vnnlib, vnnlib_shape,
        init_global_lb, lower_bounds, upper_bounds, new_idx,
        timeout_threshold=timeout_threshold,
        bab_ret=bab_ret, lA=lA, cplex_processes=cplex_processes,
        reference_slopes=saved_slopes, activation_opt_params=activation_opt_params,
        refined_betas=refined_betas, attack_images=all_adv_candidates, attack_margins=attack_margins)
    print('--- Preimage Generation ends ---')

    if args.effect:
        return subdomain_num, time_cost, iter_cov_quota[-1]     
    elif args.quant:
        log_file = os.path.join(args.result_dir, '{}_prio_{}_rule_{}_quant.txt'.format(args.dataset, args.prioritize, args.smart))
        with open(log_file, "a") as f:
            f.write("{}, Spec {} Target {} -- #Subdomain: {}, Coverage: {:.3f}, Time: {:.3f} \n".format(args.dataset,
            args.label, args.threshold, subdomain_num, iter_cov_quota[-1], time_cost))     
    elif args.depth:
        log_file = os.path.join(args.result_dir, '{}_depth_{}_LSE.txt'.format(args.dataset, args.hidden_layer_num))
        if args.dataset == 'vcas':
            with open(log_file, "a") as f:
                f.write("VCAS-{}, VCAS idx {}, Spec {} -- #Subdomain: {}, Coverage: {:.3f}, Time: {:.3f} \n".format(args.hidden_layer_num, vcas_idx,
                args.label, subdomain_num, iter_cov_quota[-1], time_cost))
    elif args.width:
        log_file = os.path.join(args.result_dir, '{}_width_{}_LSE.txt'.format(args.dataset, args.hidden_dim))
        if args.dataset == 'vcas':
            with open(log_file, "a") as f:
                f.write("VCAS-D1-{}, VCAS idx {}, Spec {} -- #Subdomain: {}, Coverage: {:.3f}, Time: {:.3f} \n".format(args.hidden_dim, vcas_idx,
                args.label, subdomain_num, iter_cov_quota[-1], time_cost))
    else:
        log_file = os.path.join(args.result_dir, '{}_prio_{}_rule_{}_LSE.txt'.format(args.dataset, args.prioritize, args.smart))
        if args.dataset == 'vcas':
            with open(log_file, "a") as f:
                f.write("VCAS-21- {}, Spec {} -- #Subdomain: {}, Coverage: {:.3f}, Time: {:.3f} \n".format(vcas_idx,
                args.label, subdomain_num, iter_cov_quota[-1], time_cost))
        else:
            with open(log_file, "a") as f:
                f.write("{}, Spec {} -- #Subdomain: {}, Coverage: {:.3f}, Time: {:.3f} \n".format(args.dataset,
                args.label, subdomain_num, iter_cov_quota[-1], time_cost))
    print('--- Log ends ---')
    

    
if __name__ == "__main__":
    args = get_args()
    if args.dataset == 'vcas':
        # Target Polytope coverage
        args.threshold = 0.9
        args.model_name = 'vcas'
        # Output number and specification number
        args.output_dim = 9
        args.initial_max_domains = 8
        if args.quant:
            # This branch is for quantitative verification
            args.vcas = 1
        elif args.depth:
            args.vcas = 1
            args.hidden_layer_num = 6
            args.model = 'VertCAS_{}.onnx'.format(args.vcas)                
            main(args)
        elif args.width:
            args.vcas = 1
            args.model = 'VertCAS_{}.onnx'.format(args.vcas)
            h_dim_lst = [8, 16, 32, 64, 128, 256]
            for h_dim in h_dim_lst:
                args.hidden_dim = h_dim                 
                main(args)
        else:
            if args.vcas == 0:
                # This is for quick running of all vcas models
                for model_idx in range(1,10):
                    args.vcas = model_idx
                    args.model = 'VertCAS_{}.onnx'.format(args.vcas)
                    main(args)
            else:
                args.model = 'VertCAS_{}.onnx'.format(args.vcas)                
                main(args)
    elif 'auto_park' in args.dataset:
        args.vcas = 0 # not vcas model, set to 0
        args.record = False # for target coverage impact analysis
        args.effect = False # for sample size impact analysis
        if args.quant:
            args.threshold = 0.95
            args.model = "model_auto_park_auto_park_model_20.pt"
            args.hidden_layer_num = 1
            args.hidden_dim = 20
            main(args)
        else:
            args.threshold = 0.9
            if args.hidden_layer_num == 2:
                args.model = 'model_auto_park_auto_park_model_10_2.pt'
            args.label = 1 # Specify the output label
            main(args)
    elif args.dataset == "cartpole":
        args.model_name = "cartpole"
        args.model = 'cartpole.onnx'
        args.output_dim = 2
        args.initial_max_domains = 1
        args.effect = False # For sample size impact analysis
        if args.effect:
            repeat_num = 20 # The number of multiple runs
            args.label = 0
            suite_size = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000]
            for size in suite_size:
                args.sample_num = size
                dm_sum = 0
                time_sum = 0
                cov_sum = 0
                for i in range(repeat_num):
                    dm_num, time_cost, cov_quota = main(args)
                    dm_sum += dm_num
                    time_sum += time_cost
                    cov_sum += cov_quota
                avr_dm = dm_sum / repeat_num
                avr_time = time_sum / repeat_num
                avr_quota = cov_sum / repeat_num
                log_file = os.path.join(args.result_dir, '{}_sample_eval_LSE_multi.txt'.format(args.dataset))
                with open(log_file, "a") as f:
                    f.write("{}, Spec {}, Sample {} -- #Subdomain: {:.1f}, Time: {:.3f}, Coverage: {:.3f} \n".format(args.dataset,
                    args.label, args.sample_num, avr_dm, avr_time, avr_quota))  
        else:
            main(args)
    elif args.dataset == "dubinsrejoin":
        args.record = False # for target coverage impact analysis
        args.effect = False
        args.model_name = "dubinsrejoin"
        args.model = 'dubinsrejoin.onnx'
        args.label = 0
        args.output_dim = 8
        args.initial_max_domains = 6
        if args.record:
            args.threshold = 0.9 # Set a higher coverage for coverage impact analysis
        main(args)
    elif args.dataset == "lunarlander":
        args.label = 1 # The label represents "fire main engine"
        args.record = False
        args.effect = False
        args.model_name = "lunarlander"
        args.model = 'lunarlander.onnx'
        args.output_dim = 4
        args.initial_max_domains = 3
        main(args) 
    else:
        print('The configured dataset is not supported.')   
