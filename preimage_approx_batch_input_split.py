import time
import numpy as np
import torch
import math
import os
import sys
from pathlib import Path
cwd = Path.cwd()
if str(cwd).endswith("PreimgApprox"):
    sys_path =  os.path.join(str(cwd), 'alpha-beta-CROWN')
sys.path.append(sys_path)
import pickle

import preimage_arguments
from preimage_polyhedron_util import calc_refine_under_approx_coverage, post_process_A, calc_input_coverage_single_label 
from auto_LiRPA.utils import stop_criterion_batch
from preimage_subdomain_queue import SortedInputDomainList, UnsortedInputDomainList
from preimage_branching_heuristics import input_split_all_feature_parallel, input_split_feature_edge
# from attack_pgd import pgd_attack_with_general_specs, test_conditions, gen_adv_example

Visited, Solve_slope, storage_depth = 0, False, 0


def batch_verification_input_split_baseline(
    args,
    d,
    net,
    batch,
    batch_spec,
    decision_thresh,
    shape=None,
    bounding_method="crown",
    branching_method="sb",
    stop_func=stop_criterion_batch,
):
    split_start_time = time.time()
    global Visited

    # STEP 1: find the neuron to split and create new split domains.
    pickout_start_time = time.time()
    ret = d.pick_out_batch(batch=1, device=net.x.device)
    dom_ub = dom_lb = None
    pickout_time = time.time() - pickout_start_time

    # STEP 2: find the neuron to split and create new split domains.
    decision_start_time = time.time()
    slopes, dm_l_all, dm_u_all, cs, thresholds, split_idx = ret

    # split_depth = get_split_depth(dm_l_all)
    new_dm_l_all, new_dm_u_all, cs, thresholds, split_depth = input_split_feature_edge(
        dm_l_all, dm_u_all, shape, cs, thresholds, split_depth=1, i_idx=split_idx)


    # slopes = slopes * (2 ** (split_depth - 1))
    subdomain_num = new_dm_l_all.shape[0]
    # spec_num = len(slopes)
    # if spec_num == 1:
        # print('Only 1 spec')
    # else:
    # slopes = slopes * (int(subdomain_num / 2))
    slopes = slopes * (int(subdomain_num/(batch_spec*2)))
    decision_time = time.time() - decision_start_time

    # STEP 3: Compute bounds for all domains.
    bounding_start_time = time.time()
    ret = net.get_lower_bound_naive(
        dm_l=new_dm_l_all, dm_u=new_dm_u_all, slopes=slopes,
        bounding_method=bounding_method, C=cs,
        stop_criterion_func=stop_func(thresholds),
    )
    dom_lb, dom_ub, slopes, lA, A_dict = ret
    bounding_time = time.time() - bounding_start_time

    
    cov_input_idx_all, A_b_dict_input_idx_all = calc_refine_under_approx_coverage(A_dict, new_dm_l_all, new_dm_u_all, batch_spec, args)


    select_idx = 0 
    # Update the attributes of subdomains that need to be added
    lA_list, lbias_list = [], []
    for j in range(2): # 2*i*spec_dim+j*spec_dim : 2*i*spec_dim+(j+1)*spec_dim
        lA_list.append(A_b_dict_input_idx_all['lA'][2*select_idx*batch_spec+j*batch_spec: 2*select_idx*batch_spec+(j+1)*batch_spec])
        lbias_list.append(A_b_dict_input_idx_all['lbias'][2*select_idx*batch_spec+j*batch_spec: 2*select_idx*batch_spec+(j+1)*batch_spec])
    cov_quota_list = [cov_info[1] for cov_info in cov_input_idx_all[select_idx][:2]]
    target_vol_list = [cov_info[0] for cov_info in cov_input_idx_all[select_idx][:2]]
    dom_lb = dom_lb[2*select_idx*batch_spec: 2*(select_idx+1)*batch_spec]
    new_dm_l_all = new_dm_l_all[2*select_idx*batch_spec: 2*(select_idx+1)*batch_spec]
    new_dm_u_all = new_dm_u_all[2*select_idx*batch_spec: 2*(select_idx+1)*batch_spec]
    cs = cs[2*select_idx*batch_spec: 2*(select_idx+1)*batch_spec]
    thresholds = thresholds[2*select_idx*batch_spec: 2*(select_idx+1)*batch_spec]

    if slopes is not None and type(slopes) != list:
        slope_dict = {}
        for key0 in slopes.keys():
            slope_dict[key0] = {}
            for key1 in slopes[key0].keys():
                slope_dict[key0][key1] = slopes[key0][key1][:, :, 2*select_idx*batch_spec : 2*(select_idx + 1)*batch_spec]
    # STEP 4: Add new domains back to domain list.
    adddomain_start_time = time.time()
    d.add_multi(
        cov_quota_list,
        target_vol_list,
        lA_list,
        lbias_list,
        dom_lb,
        new_dm_l_all.detach(),
        new_dm_u_all.detach(),
        slope_dict,
        cs,
        thresholds
    )
    adddomain_time = time.time() - adddomain_start_time

    total_time = time.time() - split_start_time
    print(
        f"Total time: {total_time:.4f}  pickout: {pickout_time:.4f}  decision: {decision_time:.4f}  bounding: {bounding_time:.4f}  add_domain: {adddomain_time:.4f}"
    )
    dom_len = len(d)
    print("length of domains:", dom_len)

    

    whole_vol = 0
    cov_vol = 0
    for i in range(dom_len):
        tmp_dm = d[i]
        if tmp_dm[-1] == 0:
            continue
        else: 
            whole_vol += tmp_dm[-1]
            cov_vol += tmp_dm[-1] * tmp_dm[0]
    cov_quota = cov_vol/whole_vol

    # the state indicates the nodes in the branching tree
    Visited += 2 
    print("{} domains visited\n".format(Visited))


    return cov_quota

def batch_verification_input_split(
    args,
    d,
    net,
    batch,
    batch_spec,
    decision_thresh,
    shape=None,
    bounding_method="crown",
    branching_method="sb",
    stop_func=stop_criterion_batch,
):
    split_start_time = time.time()
    global Visited

    # STEP 1: find the neuron to split and create new split domains.
    pickout_start_time = time.time()
    ret = d.pick_out_batch(batch=1, device=net.x.device)
    dom_ub = dom_lb = None
    pickout_time = time.time() - pickout_start_time

    # STEP 2: find the neuron to split and create new split domains.
    decision_start_time = time.time()
    slopes, dm_l_all, dm_u_all, cs, thresholds, split_idx = ret


    new_dm_l_all, new_dm_u_all, cs, thresholds, split_depth = input_split_all_feature_parallel(
        dm_l_all, dm_u_all, shape, cs, thresholds, split_depth=1, i_idx=split_idx)



    subdomain_num = new_dm_l_all.shape[0]

    if slopes is not None:
        slopes = slopes * (int(subdomain_num/(batch_spec*2)))
    decision_time = time.time() - decision_start_time

    # STEP 3: Compute bounds for all domains.
    bounding_start_time = time.time()
    ret = net.get_lower_bound_naive(
        dm_l=new_dm_l_all, dm_u=new_dm_u_all, slopes=slopes,
        bounding_method=bounding_method, C=cs,
        stop_criterion_func=stop_func(thresholds),
    )

    dom_lb, dom_ub, slopes, lA, A_dict = ret
    bounding_time = time.time() - bounding_start_time

    
    cov_input_idx_all, A_b_dict_input_idx_all = calc_refine_under_approx_coverage(A_dict, new_dm_l_all, new_dm_u_all, batch_spec, args)


    select_idx = 0
    max_cov_vol = -1
    for i, cov_info in enumerate(cov_input_idx_all):
        if cov_info[2][0]*cov_info[2][1] > max_cov_vol:
            select_idx = i
            max_cov_vol = cov_info[2][0]*cov_info[2][1]
 
    # Update the attributes of subdomains that need to be added
    lA_list, lbias_list = [], []
    for j in range(2): # 2*i*spec_dim+j*spec_dim : 2*i*spec_dim+(j+1)*spec_dim
        lA_list.append(A_b_dict_input_idx_all['lA'][2*select_idx*batch_spec+j*batch_spec: 2*select_idx*batch_spec+(j+1)*batch_spec])
        lbias_list.append(A_b_dict_input_idx_all['lbias'][2*select_idx*batch_spec+j*batch_spec: 2*select_idx*batch_spec+(j+1)*batch_spec])
    print('check lA lbias', lA_list, lbias_list)

    cov_quota_list = [cov_info[1] for cov_info in cov_input_idx_all[select_idx][:2]]
    target_vol_list = [cov_info[0] for cov_info in cov_input_idx_all[select_idx][:2]]
    dom_lb = dom_lb[2*select_idx*batch_spec: 2*(select_idx+1)*batch_spec]
    new_dm_l_all = new_dm_l_all[2*select_idx*batch_spec: 2*(select_idx+1)*batch_spec]
    new_dm_u_all = new_dm_u_all[2*select_idx*batch_spec: 2*(select_idx+1)*batch_spec]
    cs = cs[2*select_idx*batch_spec: 2*(select_idx+1)*batch_spec]
    thresholds = thresholds[2*select_idx*batch_spec: 2*(select_idx+1)*batch_spec]

    if slopes is not None and type(slopes) != list:
        slope_dict = {}
        for key0 in slopes.keys():
            slope_dict[key0] = {}
            for key1 in slopes[key0].keys():
                slope_dict[key0][key1] = slopes[key0][key1][:, :, 2*select_idx*batch_spec : 2*(select_idx + 1)*batch_spec]
    else:
        slope_dict = None
    # STEP 4: Add new domains back to domain list.
    adddomain_start_time = time.time()
    d.add_multi(
        cov_quota_list,
        target_vol_list,
        lA_list,
        lbias_list,
        dom_lb,
        new_dm_l_all.detach(),
        new_dm_u_all.detach(),
        slope_dict,
        cs,
        thresholds
    )
    adddomain_time = time.time() - adddomain_start_time

    total_time = time.time() - split_start_time
    print(
        f"Total time: {total_time:.4f}  pickout: {pickout_time:.4f}  decision: {decision_time:.4f}  bounding: {bounding_time:.4f}  add_domain: {adddomain_time:.4f}"
    )
    dom_len = len(d)
    print("length of domains:", dom_len)


    

    whole_vol = 0
    cov_vol = 0
    for i in range(dom_len):
        tmp_dm = d[i]
        # if tmp_dm[0][0] == 1 and tmp_dm[0][1] == 0:
        if tmp_dm[-1] == 0:
            continue
        else: 
            whole_vol += tmp_dm[-1]
            cov_vol += tmp_dm[-1] * tmp_dm[0]
    cov_quota = cov_vol/whole_vol

    # the state indicates the nodes in the branching tree
    Visited += 2 
    print("{} branch and bound domains visited\n".format(Visited))


    return cov_quota

def input_approx_parallel_multi(
    net,
    init_domain,
    x,
    args,
    model_ori=None,
    all_prop=None,
    rhs=None,
    timeout=None,
    branching_method="naive",
):
    global storage_depth
    # add the coverage stopping threshold
    cov_thre = args.threshold

    prop_mat_ori = net.c[0]

    start = time.time()
    # All supported test_arguments.
    global Visited, Flag_first_split, all_node_split, DFS_enabled

    timeout = timeout or preimage_arguments.Config["bab"]["timeout"]
    batch = preimage_arguments.Config["solver"]["batch_size"]

    bounding_method = preimage_arguments.Config["solver"]["bound_prop_method"]
    # added the budget argument
    branch_budget = preimage_arguments.Config['bab']['branching']['input_split']['branch_budget']

    stop_func = stop_criterion_batch
    spec_num = x.shape[0]
    Visited, Flag_first_split, global_ub = 0, True, np.inf
    # add the flag
    Flag_covered = False

    (
        global_ub,
        global_lb,
        _,
        _,
        primals,
        updated_mask,
        lA,
        A,
        lower_bounds,
        upper_bounds,
        pre_relu_indices,
        slope,
        history,
        attack_image,
    ) = net.build_the_model(
        init_domain,
        x,
        spec_num,
        stop_criterion_func=stop_func(rhs),
        bounding_method=bounding_method,
    )
    
                    
        
    if hasattr(net.net[net.net.input_name[0]], 'lA'):
        lA = net.net[net.net.input_name[0]].lA.transpose(0, 1)
    else:
        raise ValueError("sb heuristic cannot be used without lA.")
    dm_l = x.ptb.x_L
    dm_u = x.ptb.x_U

    # compute storage depth, now we consider one selected domain with splitting on all features
    use_slope = preimage_arguments.Config["solver"]["bound_prop_method"] == "alpha-crown"
    min_batch_size = (
        preimage_arguments.Config["solver"]["min_batch_size_ratio"]
        * preimage_arguments.Config["solver"]["batch_size"]
    )
    max_depth = max(int(math.log(max(min_batch_size, 1)) // math.log(2)), 1)
    storage_depth = min(max_depth, dm_l.shape[-1]) # it is feature size in general for RL tasks



    in_dim = dm_l.shape[-1]
    # print('check input dim for split index', in_dim)
    batch_spec = len(dm_l)
    split_idx = torch.arange(in_dim)
    split_idx = split_idx.repeat(batch_spec, 1)
    

    # the under approx quality evaluation for the polytope based on initial domain (the whole domain)
    # initial check whether satisfy the preimage underapproximation criterion
    initial_covered, cov_quota, target_vol, preimage_dict = initial_check_preimage_approx(A, cov_thre, args)
    if initial_covered:
        return (
            initial_covered,
            (dm_l, dm_u),
            preimage_dict,
            0,
            [cov_quota],
            time.time() - start,
            [cov_quota],
            1
        )
    elif target_vol == 0:
        return (
            initial_covered,
            (dm_l, dm_u),
            preimage_dict,
            0,
            [1],
            time.time() - start,
            [1],
            1
        )

    if args.save_process:
        preimage_dict['dm_l'] = dm_l.cpu().detach().numpy() 
        preimage_dict['dm_u'] = dm_u.cpu().detach().numpy()
        save_path = os.path.join(args.result_dir, 'run_example')
        save_file = os.path.join(save_path,'{}_spec_{}_init_{}'.format(args.dataset, args.label, args.base))
        with open(save_file, 'wb') as f:
            pickle.dump(preimage_dict, f)

    # use SortedInputDomain
    domains = SortedInputDomainList()
    

    cov_quota_list = [cov_quota]
    target_vol_list = [target_vol]
    domains.add_multi(
        cov_quota_list,
        target_vol_list,
        [preimage_dict['lA']],
        [preimage_dict['lbias']],
        global_lb,
        dm_l.detach(),
        dm_u.detach(),
        slope,
        net.c,
        rhs,
        split_idx
    )

    cov_record = [[time.time() - start, cov_quota]]
    iter_cov_quota = [cov_quota]
   
   
    num_iter = 1

    enhanced_bound_initialized = False

    
    while cov_quota < cov_thre:

        if Visited > branch_budget:
            time_cost = time.time() - start
            cov_dm_all, preimage_dict_all, dm_rec_all = get_preimage_info(domains)
            subdomain_num = len(domains)
            del domains
            return Flag_covered, dm_rec_all, preimage_dict_all, Visited, cov_dm_all, time_cost, iter_cov_quota, subdomain_num
        if args.smart:
            cov_quota = batch_verification_input_split(
                args,
                domains,
                net,
                batch,
                batch_spec,
                decision_thresh=rhs,
                shape=x.shape,
                bounding_method=bounding_method,
                branching_method=branching_method,
                stop_func=stop_func,
            )
        else:
            cov_quota = batch_verification_input_split_baseline(
                args,
                domains,
                net,
                batch,
                batch_spec,
                decision_thresh=rhs,
                shape=x.shape,
                bounding_method=bounding_method,
                branching_method=branching_method,
                stop_func=stop_func,
            )            

        print('--- Iteration {}, Cov quota {} ---'.format(num_iter, cov_quota))
        iter_cov_quota.append(cov_quota)
        if args.save_process:
            cov_dm_all, preimage_dict_all, dm_rec_all = get_preimage_info(domains)

            save_path = os.path.join(args.result_dir, 'run_example')
            save_file = os.path.join(save_path,'{}_spec_{}_iter_{}_{}'.format(args.dataset, args.label, num_iter, args.base))
            with open(save_file, 'wb') as f:
                pickle.dump((preimage_dict_all, dm_rec_all), f)

        num_iter += 1
        

        
        
    time_cost = time.time() - start
    # Before deleting the subdomains, record what we need for the final polyhedral rep 
    cov_dm_all, preimage_dict_all, dm_rec_all = get_preimage_info(domains)
    subdomain_num = len(domains)
    if args.quant:
        save_path = os.path.join(args.result_dir, 'quant_analysis')
        save_file = os.path.join(save_path,'{}_spec_{}_dm_{}'.format(args.dataset, args.label, subdomain_num))
        with open(save_file, 'wb') as f:
            pickle.dump((preimage_dict_all, dm_rec_all), f)
    del domains
    Flag_covered = True
    return Flag_covered, dm_rec_all, preimage_dict_all, Visited, cov_dm_all, time_cost, iter_cov_quota, subdomain_num

   
   
   
def input_approx_parallel_multi_no_prioritization(
    net,
    init_domain,
    x,
    args,
    model_ori=None,
    all_prop=None,
    rhs=None,
    timeout=None,
    branching_method="naive",
):
    global storage_depth

    # add the coverage stopping threshold
    cov_thre = args.threshold

    prop_mat_ori = net.c[0]

    start = time.time()
    # All supported test_arguments.
    global Visited, Flag_first_split, all_node_split, DFS_enabled

    timeout = timeout or preimage_arguments.Config["bab"]["timeout"]
    batch = preimage_arguments.Config["solver"]["batch_size"]

    bounding_method = preimage_arguments.Config["solver"]["bound_prop_method"]
    # added the branch budget argument
    branch_budget = preimage_arguments.Config['bab']['branching']['input_split']['branch_budget']

    stop_func = stop_criterion_batch

    Visited, Flag_first_split, global_ub = 0, True, np.inf
    # add the flag
    Flag_covered = False

    (
        global_ub,
        global_lb,
        _,
        _,
        primals,
        updated_mask,
        lA,
        A,
        lower_bounds,
        upper_bounds,
        pre_relu_indices,
        slope,
        history,
        attack_image,
    ) = net.build_the_model(
        init_domain,
        x,
        stop_criterion_func=stop_func(rhs),
        bounding_method=bounding_method,
    )
    if hasattr(net.net[net.net.input_name[0]], 'lA'):
        lA = net.net[net.net.input_name[0]].lA.transpose(0, 1)
    else:
        raise ValueError("sb heuristic cannot be used without lA.")
    dm_l = x.ptb.x_L
    dm_u = x.ptb.x_U

    # compute storage depth
    use_slope = preimage_arguments.Config["solver"]["bound_prop_method"] == "alpha-crown"
    min_batch_size = (
        preimage_arguments.Config["solver"]["min_batch_size_ratio"]
        * preimage_arguments.Config["solver"]["batch_size"]
    )
    max_depth = max(int(math.log(max(min_batch_size, 1)) // math.log(2)), 1)
    storage_depth = min(max_depth, dm_l.shape[-1])



    in_dim = dm_l.shape[-1]
    # print('check input dim for split index', in_dim)
    batch_spec = len(dm_l)
    split_idx = torch.arange(in_dim)
    split_idx = split_idx.repeat(batch_spec, 1)
    

    # the under approx quality evaluation for the polytope based on initial domain (the whole domain)
    # initial check whether satisfy the preimage underapproximation criterion
    initial_covered, cov_quota, target_vol, preimage_dict = initial_check_preimage_approx(A, cov_thre, args)
    if initial_covered:
        return (
            initial_covered,
            (dm_l, dm_u),
            preimage_dict,
            0,
            [cov_quota],
            time.time() - start,
            [cov_quota],
            1
        )
    elif target_vol == 0:
        return (
            initial_covered,
            (dm_l, dm_u),
            preimage_dict,
            0,
            [1],
            time.time() - start,
            [1],
            1
        )

    domains = UnsortedInputDomainList()
    cov_quota_list = [cov_quota]
    target_vol_list = [target_vol]
    domains.add_multi(
        cov_quota_list,
        target_vol_list,
        [preimage_dict['lA']],
        [preimage_dict['lbias']],
        global_lb,
        dm_l.detach(),
        dm_u.detach(),
        slope,
        net.c,
        rhs,
        split_idx
    )

    cov_record = [[time.time() - start, cov_quota]]
    iter_cov_quota = [cov_quota]
   
   
    num_iter = 1
    enhanced_bound_initialized = False

    
    while cov_quota < cov_thre:
        if Visited > branch_budget:
            time_cost = time.time() - start
            cov_dm_all, preimage_dict_all, dm_rec_all = get_preimage_info(domains)
            subdomain_num = len(domains)
            del domains
            return Flag_covered, dm_rec_all, preimage_dict_all, Visited, cov_dm_all, time_cost, iter_cov_quota, subdomain_num
        cov_quota = batch_verification_input_split(
            args,
            domains,
            net,
            batch,
            batch_spec,
            decision_thresh=rhs,
            shape=x.shape,
            bounding_method=bounding_method,
            branching_method=branching_method,
            stop_func=stop_func,
        )
        print('--- Iteration {}, Cov quota {} ---'.format(num_iter, cov_quota))
        iter_cov_quota.append(cov_quota)
        num_iter += 1
    time_cost = time.time() - start
    # Before deleting the subdomains, record what we need for the final polyhedral rep 
    cov_dm_all, preimage_dict_all, dm_rec_all = get_preimage_info(domains)
    subdomain_num = len(domains)
    del domains
    Flag_covered = True
    return Flag_covered, dm_rec_all, preimage_dict_all, Visited, cov_dm_all, time_cost, iter_cov_quota, subdomain_num


def get_preimage_info(domains):
    dm_rec_all = []
    preimage_dict_all = []
    cov_dm_all = []
    for idx, dom in enumerate(domains):
        cov_dm_all.append(dom[0])
        # change the attribute lower bound into preimage_dict
        preimage_dict_all.append((dom[1],dom[2]))
        dm_rec_all.append((dom[3].cpu().detach().numpy(), dom[4].cpu().detach().numpy()))
    return cov_dm_all, preimage_dict_all, dm_rec_all
    
    


def initial_check_preimage_approx(A_dict, thre, args):
    """check whether optimization on initial domain is successful"""
    preimage_dict = post_process_A(A_dict)
    target_vol, cov_quota = calc_input_coverage_single_label(preimage_dict, args)
    if cov_quota >= thre:  # check whether the preimage approx satisfies the criteria
        print("Reached by optmization on the initial domain!")
        return True, cov_quota, target_vol, preimage_dict
    else:
        return False, cov_quota, target_vol, preimage_dict
