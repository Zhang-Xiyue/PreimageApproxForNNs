#########################################################################
##                                                                     ##
##     This program is licenced under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
"""Branch and bound for input space split."""

import time
import numpy as np
import torch
from torch import Tensor
import math
import os
import pickle
import random
# import logging

# import preimage_arguments
import arguments
from auto_LiRPA.utils import stop_criterion_batch
# from branching_domains_input_split import (
#     UnsortedInputDomainList,
#     SortedInputDomainList,
# )
from test_branching_subdomain_queue import SortedInputDomainList, UnsortedInputDomainList
# change to test_branching
from test_branching_heuristics import input_split_all_feature_parallel, input_split_feature_edge, input_split_branching, get_split_depth
from attack_pgd import pgd_attack_with_general_specs, test_conditions, gen_adv_example

from test_polyhedron_util import post_process_A, calc_input_coverage_initial_input_under, calc_input_coverage_initial_input_over
from test_polyhedron_util import post_process_greedy_A, calc_Hrep_coverage_multi_spec_pairwise_under, calc_Hrep_coverage_multi_spec_pairwise_over
from test_polyhedron_util import calc_mc_esti_coverage_initial_input_under, calc_mc_coverage_multi_spec_pairwise_under
Visited, Solve_slope, storage_depth = 0, False, 0

# logging.basicConfig(filename='experiment_{}.log'.format(arguments.Config["data"]["dataset"]), level=logging.INFO, 
                    # format='%(asctime)s - %(message)s')
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
    # xy: may not need this, we just split wrt all potential input features
    decision_time = time.time() - decision_start_time

    # STEP 3: Compute bounds for all domains.
    bounding_start_time = time.time()
    ret = net.get_lower_bound_naive(
        dm_l=new_dm_l_all, dm_u=new_dm_u_all, slopes=slopes,
        bounding_method=bounding_method, C=cs,
        stop_criterion_func=stop_func(thresholds),
    )
    # here slopes is a dict FIXME: check to ensure "ret" gives us A_dict
    dom_lb, dom_ub, slopes, lA, A_dict = ret
    bounding_time = time.time() - bounding_start_time

    
    from test_polyhedron_util import save_A_dict, calc_refine_under_approx_coverage
    # save_A_dict(A_dict, A_path="A_dict_BaB_init_level_{}".format(split_depth))
    # torch.save(new_dm_l_all, 'dm_l_tensor_level_{}.pt'.format(split_depth))
    # torch.save(new_dm_u_all, 'dm_u_tensor_level_{}.pt'.format(split_depth))
    cov_input_idx_all, A_b_dict_input_idx_all = calc_refine_under_approx_coverage(A_dict, new_dm_l_all, new_dm_u_all, batch_spec, args)

    # split_idx = input_split_branching(net, dom_lb, new_dm_l_all, new_dm_u_all, lA, thresholds, branching_method, None, shape, slopes, storage_depth)
        
    # select one bisection to keep which leads to best coverage
    # remeber that you take every possible bisection on input feats
    # select_idx = 0
    # max_cov_vol = -1
    # for i, cov_info in enumerate(cov_input_idx_all):
    #     if cov_info[2] > max_cov_vol:
    #         select_idx = i
    #         max_cov_vol = cov_info[2]
    select_idx = 0 
    # Update the attributes of subdomains that need to be added
    # xy: additionally we need the lA and lbias dict
    lA_list, lbias_list = [], []
    for j in range(2): # 2*i*spec_dim+j*spec_dim : 2*i*spec_dim+(j+1)*spec_dim
        lA_list.append(A_b_dict_input_idx_all['lA'][2*select_idx*batch_spec+j*batch_spec: 2*select_idx*batch_spec+(j+1)*batch_spec])
        lbias_list.append(A_b_dict_input_idx_all['lbias'][2*select_idx*batch_spec+j*batch_spec: 2*select_idx*batch_spec+(j+1)*batch_spec])
    # xy: cov_quota_list records the added domains, actually in the end you can always obtain the cov_quota, dm_l, dm_b into the class and obtain them later
    cov_quota_list = [cov_info[1] for cov_info in cov_input_idx_all[select_idx][:2]]
    target_vol_list = [cov_info[0] for cov_info in cov_input_idx_all[select_idx][:2]]
    dom_lb = dom_lb[2*select_idx*batch_spec: 2*(select_idx+1)*batch_spec]
    new_dm_l_all = new_dm_l_all[2*select_idx*batch_spec: 2*(select_idx+1)*batch_spec]
    new_dm_u_all = new_dm_u_all[2*select_idx*batch_spec: 2*(select_idx+1)*batch_spec]
    cs = cs[2*select_idx*batch_spec: 2*(select_idx+1)*batch_spec]
    thresholds = thresholds[2*select_idx*batch_spec: 2*(select_idx+1)*batch_spec]
    # split_idx = split_idx[2*select_idx, 2*(select_idx+1)]
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

    
    # NOTE: In our case, len(d) will never be 0, set an early exit when reach a budget 
    # if dom_len == 0:
    #     print("No domains left, verification finished!")
    #     if dom_lb is not None:
    #         print(f"The lower bound of last batch is {dom_lb.min().item()}")
    #     return decision_thresh.max() + 1e-7
    # else:
        # FIXME worst_val [0] and [-1], we can check the volume coverage oc each subregion
        # worst_idx = d.get_topk_indices().item()
        # worst_val = d[worst_idx]
        # global_lb = worst_val[0] - worst_val[-1]
        
        # xy: now we calculate the overall cov_quota according to subdomain info for the while loop
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

    # xy: here the state indicates the nodes in the branching tree
    # Visited += len(new_dm_l_all)
    Visited += 2 
    # FIXME: we can print the current attained coverage ratio, for the part where there is no points of the property, then we need to process accordingly
    # print(f"Current (lb-rhs): {global_lb.max().item()}")
    print("{} branch and bound domains visited\n".format(Visited))


    return cov_quota

def batch_verification_input_split(
    d,
    net,
    batch,
    batch_spec,
    decision_thresh,
    shape=None,
    bounding_method="crown",
    branching_method="sb",
    stop_func=stop_criterion_batch,
    bound_lower=False,
    bound_upper=False
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
    slopes, dm_l_all, dm_u_all, cs, thresholds, _ = ret

    # split_depth = get_split_depth(dm_l_all)
    # if arguments.Config["data"]["dataset"] == "vcas":
    #     split_idx = torch.tensor([0,1,2,3])
    # else:
    split_idx = torch.arange(dm_l_all.shape[1])        
    new_dm_l_all, new_dm_u_all, cs, thresholds, split_depth = input_split_all_feature_parallel(
        dm_l_all, dm_u_all, shape, cs, thresholds, split_depth=1, i_idx=split_idx)


    # slopes = slopes * (2 ** (split_depth - 1))
    subdomain_num = new_dm_l_all.shape[0]
    # spec_num = len(slopes)
    # if spec_num == 1:
        # print('Only 1 spec')
    # else:
    # slopes = slopes * (int(subdomain_num / 2))
    if slopes is not None:
        slopes = slopes * (int(subdomain_num/(batch_spec*2)))
    # xy: may not need this, we just split wrt all potential input features
    decision_time = time.time() - decision_start_time

    # STEP 3: Compute bounds for all domains.
    bounding_start_time = time.time()
    ret = net.get_lower_bound_naive(
        dm_l=new_dm_l_all, dm_u=new_dm_u_all, slopes=slopes,
        bounding_method=bounding_method, C=cs,
        stop_criterion_func=stop_func(thresholds),
        bound_lower=bound_lower, bound_upper=bound_upper
    )
    # here slopes is a dict FIXME: check to ensure "ret" gives us A_dict
    dom_lb, dom_ub, slopes, A_dict = ret
    bounding_time = time.time() - bounding_start_time

    
    
    A_b_dict_input_idx_all = post_process_greedy_A(A_dict)
    if bound_lower:
        if arguments.Config['preimage']["quant"]:
            cov_input_idx_all = calc_mc_coverage_multi_spec_pairwise_under(A_b_dict_input_idx_all, new_dm_l_all, new_dm_u_all, batch_spec)
        else:
            cov_input_idx_all = calc_Hrep_coverage_multi_spec_pairwise_under(A_b_dict_input_idx_all, new_dm_l_all, new_dm_u_all, batch_spec)
        # select one bisection to keep which leads to best coverage
        # remeber that the algorithm takes every possible bisection on input feats
        select_idx = None
        max_under_reward = None
        max_under_cov = None
        under_cov_all = []
        cov_reward_feat_all = []
        if arguments.Config["preimage"]["smooth_val"]:
            for i, cov_info in enumerate(cov_input_idx_all):
                if select_idx is None:
                    select_idx = i
                    # min_over_cov = cov_info[2][1]
                    max_under_reward = cov_info[2][2]
                else:
                    # if cov_info[2][0]*cov_info[2][1] > max_cov_vol:
                    if cov_info[2][2] > max_under_reward:
                        select_idx = i
                        # max_cov_vol = cov_info[2][0]*cov_info[2][1]
                        max_under_reward = cov_info[2][2]
                cov_reward_feat_all.append(cov_info[2][2])
            cov_reward_feat_all = np.array(cov_reward_feat_all)
            if (np.max(cov_reward_feat_all) - np.min(cov_reward_feat_all)) < 0.1:
                print("use the longest length")
                select_idx = torch.topk(dm_u_all[0] - dm_l_all[0], k=1, dim=-1).indices
                select_idx = select_idx[0]
        else:
            for i, cov_info in enumerate(cov_input_idx_all):
                if select_idx is None:
                    select_idx = i
                    max_under_cov = cov_info[2][1]
                else:
                    if cov_info[2][1] > max_under_cov:
                    # if cov_info[2][2] > max_under_reward:
                        select_idx = i
                        max_under_cov = cov_info[2][1]
                        # max_under_reward = cov_info[2][2]
                under_cov_all.append(cov_info[2][1])
            under_cov_all = np.array(under_cov_all)
            if (np.max(under_cov_all) - np.min(under_cov_all)) < 0.01:
                select_idx = random.randint(0, len(cov_input_idx_all)-1)
            #     print("use the longest length")
            #     select_idx = torch.topk(dm_u_all[0] - dm_l_all[0], k=1, dim=-1).indices
            #     select_idx = select_idx[0]
        print('selected feature', select_idx)
        # Update the attributes of subdomains that need to be added
        lA_list, lbias_list = [], []
        for j in range(2): # 2*i*spec_dim+j*spec_dim : 2*i*spec_dim+(j+1)*spec_dim
            lA_list.append(A_b_dict_input_idx_all['lA'][2*select_idx*batch_spec+j*batch_spec: 2*select_idx*batch_spec+(j+1)*batch_spec])
            lbias_list.append(A_b_dict_input_idx_all['lbias'][2*select_idx*batch_spec+j*batch_spec: 2*select_idx*batch_spec+(j+1)*batch_spec])
        # print('check lA lbias', lA_list, lbias_list)
        uA_list = None
        ubias_list = None
        dom_lb = dom_lb[2*select_idx*batch_spec: 2*(select_idx+1)*batch_spec]
    elif bound_upper:
        cov_input_idx_all = calc_Hrep_coverage_multi_spec_pairwise_over(A_b_dict_input_idx_all, new_dm_l_all, new_dm_u_all, batch_spec)
        select_idx = None
        # min_over_cov = None
        min_over_loss = None
        over_cov_all = []
        cov_loss_feat_all = []
        if arguments.Config["data"]["dataset"] == "auto_park_part":
            select_idx = 1
            # min_over_cov = cov_input_idx_all[select_idx][2][1]
        else:
            if arguments.Config["preimage"]["smooth_val"]:
                for i, cov_info in enumerate(cov_input_idx_all):
                    if select_idx is None:
                        select_idx = i
                        # min_over_cov = cov_info[2][1]
                        min_over_loss = cov_info[2][2]
                        # cov_quota_feat_all.append(cov_info[2][1])
                    else:
                        # if cov_info[2][1] < min_over_cov:
                        if cov_info[2][2] < min_over_loss:
                            select_idx = i
                            # min_over_cov = cov_info[2][1]
                            min_over_loss = cov_info[2][2]
                            # cov_quota_feat_all.append(cov_info[2][1])
                    cov_loss_feat_all.append(cov_info[2][2])
                # cov_quota_feat_all = np.array(cov_quota_feat_all)
                cov_loss_feat_all = np.array(cov_loss_feat_all)
                if (np.max(cov_loss_feat_all) - np.min(cov_loss_feat_all)) < 0.1:
                    print("use the longest length")
                    select_idx = torch.topk(dm_u_all[0] - dm_l_all[0], k=1, dim=-1).indices
                    select_idx = select_idx[0]
            else:
                for i, cov_info in enumerate(cov_input_idx_all):
                    if select_idx is None:
                        select_idx = i
                        min_over_cov = cov_info[2][1]
                        # cov_quota_feat_all.append(cov_info[2][1])
                    else:
                        if cov_info[2][1] < min_over_cov:
                            select_idx = i
                            min_over_cov = cov_info[2][1]
                            # cov_quota_feat_all.append(cov_info[2][1])
                    over_cov_all.append(cov_info[2][1])
                over_cov_all = np.array(over_cov_all)
                # cov_loss_feat_all = np.array(cov_loss_feat_all)
                if (np.max(over_cov_all) - np.min(over_cov_all)) < 0.1:
                    select_idx = random.randint(0, len(cov_input_idx_all)-1)
                #     print("use the longest length")
                #     select_idx = torch.topk(dm_u_all[0] - dm_l_all[0], k=1, dim=-1).indices
                #     select_idx = select_idx[0]
        print('selected feature', select_idx)
        lA_list = None
        lbias_list = None
        uA_list, ubias_list = [], []
        for j in range(2): # 2*i*spec_dim+j*spec_dim : 2*i*spec_dim+(j+1)*spec_dim
            uA_list.append(A_b_dict_input_idx_all['uA'][2*select_idx*batch_spec+j*batch_spec: 2*select_idx*batch_spec+(j+1)*batch_spec])
            ubias_list.append(A_b_dict_input_idx_all['ubias'][2*select_idx*batch_spec+j*batch_spec: 2*select_idx*batch_spec+(j+1)*batch_spec]) 
        dom_ub = dom_ub[2*select_idx*batch_spec: 2*(select_idx+1)*batch_spec]
    # xy: cov_quota_list records the added domains, actually in the end you can always obtain the cov_quota, dm_l, dm_b into the class and obtain them later
    cov_quota_list = [cov_info[1] for cov_info in cov_input_idx_all[select_idx][:2]]
    target_vol_list = [cov_info[0] for cov_info in cov_input_idx_all[select_idx][:2]]
    
    new_dm_l_all = new_dm_l_all[2*select_idx*batch_spec: 2*(select_idx+1)*batch_spec]
    new_dm_u_all = new_dm_u_all[2*select_idx*batch_spec: 2*(select_idx+1)*batch_spec]
    cs = cs[2*select_idx*batch_spec: 2*(select_idx+1)*batch_spec]
    thresholds = thresholds[2*select_idx*batch_spec: 2*(select_idx+1)*batch_spec]
    # split_idx = split_idx[2*select_idx, 2*(select_idx+1)]
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
        new_dm_l_all.detach(),
        new_dm_u_all.detach(),
        slope_dict,
        cs,
        thresholds,
        lA_list=lA_list,
        lbias_list=lbias_list,
        lb=dom_lb,
        uA_list=uA_list, 
        ubias_list=ubias_list,
        ub=dom_ub
    )
    adddomain_time = time.time() - adddomain_start_time

    total_time = time.time() - split_start_time
    print(
        f"Total time: {total_time:.4f}  pickout: {pickout_time:.4f}  decision: {decision_time:.4f}  bounding: {bounding_time:.4f}  add_domain: {adddomain_time:.4f}"
    )
    dom_len = len(d)
    print("length of domains:", dom_len)


    
    # NOTE: In our case, len(d) will never be 0, set an early exit when reach a budget 
    # if dom_len == 0:
    #     print("No domains left, verification finished!")
    #     if dom_lb is not None:
    #         print(f"The lower bound of last batch is {dom_lb.min().item()}")
    #     return decision_thresh.max() + 1e-7
    # else:
        # FIXME worst_val [0] and [-1], we can check the volume coverage oc each subregion
        # worst_idx = d.get_topk_indices().item()
        # worst_val = d[worst_idx]
        # global_lb = worst_val[0] - worst_val[-1]
        
        # xy: now we calculate the overall cov_quota according to subdomain info for the while loop
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
    if whole_vol == 0:
        print('no exact preimage exists')
        return 1
    else:
        cov_quota = cov_vol/whole_vol

    # xy: here the state indicates the nodes in the branching tree
    # Visited += len(new_dm_l_all)
    Visited += 2 
    # FIXME: we can print the current attained coverage ratio, for the part where there is no points of the property, then we need to process accordingly
    # print(f"Current (lb-rhs): {global_lb.max().item()}")
    print("{} branch and bound domains visited\n".format(Visited))


    return cov_quota

def input_bab_approx_parallel_multi(
    net,
    init_domain,
    x,
    model_ori=None,
    all_prop=None,
    rhs=None,
    timeout=None,
    branching_method="naive",
):
    global storage_depth

    # NOTE add arguments required for preimage generation
    cov_thre = arguments.Config["preimage"]["threshold"]
    branch_budget = arguments.Config['preimage']['branch_budget']
    result_dir = arguments.Config['preimage']['result_dir']
    bound_lower = arguments.Config["preimage"]["under_approx"]
    bound_upper = arguments.Config["preimage"]["over_approx"] 
    input_split_enabled = arguments.Config["bab"]["branching"]["input_split"]["enable"]
    if input_split_enabled:
        opt_input_poly = True
        opt_relu_poly = False
    else:
        opt_input_poly = False
        opt_relu_poly = True        
    # the crown_lower/upper_bounds are dummy test_arguments here --- similar to refined_lower/upper_bounds, they are not used
    """ run input split bab """
    prop_mat_ori = net.c[0]

    start = time.time()
    # All supported test_arguments.
    global Visited, Flag_first_split, all_node_split, DFS_enabled

    timeout = timeout or arguments.Config["bab"]["timeout"]
    batch = arguments.Config["solver"]["batch_size"]
    bounding_method = arguments.Config["solver"]["bound_prop_method"]

    stop_func = stop_criterion_batch
    spec_num = x.shape[0]
    Visited, Flag_first_split, global_ub = 0, True, np.inf
    # xy: add the flag
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
        opt_input_poly=opt_input_poly, 
        opt_relu_poly=opt_relu_poly
    )
    # if hasattr(net.net[net.net.input_name[0]], 'lA'):
    #     lA = net.net[net.net.input_name[0]].lA.transpose(0, 1)
    # else:
    #     raise ValueError("sb heuristic cannot be used without lA.")
    dm_l = x.ptb.x_L
    dm_u = x.ptb.x_U
    # xy: lA used below should contain lA and lbias
    # xy: the under approx quality evaluation for the polytope based on initial domain (the whole domain)
    # xy: initial check whether satisfy the preimage underapproximation criterion
    

    
    initial_covered, cov_quota, target_vol, preimage_dict = initial_check_preimage_approx(A, cov_thre)
    
    if arguments.Config["preimage"]["save_process"]:
        preimage_dict['dm_l'] = dm_l.cpu().detach().numpy() 
        preimage_dict['dm_u'] = dm_u.cpu().detach().numpy()
        save_path = os.path.join(result_dir, 'run_example')
        if bound_upper:
            save_file = os.path.join(save_path,'{}_spec_{}_init_{}'.format(arguments.Config["data"]["dataset"], arguments.Config["preimage"]["label"], bounding_method))
        elif bound_lower:
            save_file = os.path.join(save_path,'{}_spec_{}_init_under_region_1'.format(arguments.Config["data"]["dataset"], arguments.Config["preimage"]["label"]))
        with open(save_file, 'wb') as f:
            pickle.dump(preimage_dict, f)
        print('gen the initial plane for whole region')
        return
    
    if initial_covered:
        return (
            initial_covered,
            preimage_dict,
            0,
            time.time() - start,
            [cov_quota],
            1
        )
    elif target_vol == 0:
        return (
            initial_covered,
            preimage_dict,
            0,
            time.time() - start,
            [1],
            1
        )    
            
    # split_depth = get_split_depth(dm_l)

    # compute storage depth, now we consider one selected domain with splitting on all features
    min_batch_size = (
        arguments.Config["solver"]["min_batch_size_ratio"]
        * arguments.Config["solver"]["batch_size"]
    )
    max_depth = max(int(math.log(max(min_batch_size, 1)) // math.log(2)), 1)
    storage_depth = min(max_depth, dm_l.shape[-1]) # it is feature size in general for RL tasks


    # compute initial split idx
    # split_idx = input_split_branching(net, global_lb, dm_l, dm_u, lA, rhs, branching_method, None, None, None, storage_depth)
    # xy: do not order the input idx, select in a greedy manner from all possible ones
    if arguments.Config["data"]["dataset"] == 'vcas':
        in_dim = dm_l.shape[-1]
        # print('check input dim for split index', in_dim)
        batch_spec = len(dm_l)
        split_idx = torch.tensor([0, 1, 3])
        split_idx = split_idx.repeat(batch_spec, 1)
    else:
        in_dim = dm_l.shape[-1]
        # print('check input dim for split index', in_dim)
        batch_spec = len(dm_l)
        split_idx = torch.arange(in_dim)
        split_idx = split_idx.repeat(batch_spec, 1)



    # if test_arguments.Config["bab"]["batched_domain_list"]:
    #     domains = UnsortedInputDomainList(storage_depth, use_slope=use_slope)
    # else:
    # xy: use SortedInputDomain
    domains = SortedInputDomainList()
    
    # FIXME Change the InputDomain sorting conditions 
    cov_quota_list = [cov_quota]
    target_vol_list = [target_vol]
    if bound_lower:
        domains.add_multi(
            cov_quota_list,
            target_vol_list,
            dm_l.detach(),
            dm_u.detach(),
            slope,
            net.c,
            rhs,
            split_idx,
            lA_list=[preimage_dict['lA']],
            lbias_list=[preimage_dict['lbias']],
            lb=global_lb
        )
    elif bound_upper:
        domains.add_multi(
            cov_quota_list,
            target_vol_list,
            dm_l.detach(),
            dm_u.detach(),
            slope,
            net.c,
            rhs,
            split_idx,
            uA_list=[preimage_dict['uA']],
            ubias_list=[preimage_dict['ubias']],
            ub=global_ub
        )
    # glb_record = [[time.time() - start, (global_lb - rhs).max().item()]]
    cov_record = [[time.time() - start, cov_quota]]
    iter_cov_quota = [cov_quota]
   
   
    num_iter = 1
    # sort_domain_iter = test_arguments.Config["bab"]["branching"]["input_split"]["sort_domain_interval"]
    enhanced_bound_initialized = False
    # while len(domains) > 0:
    if bound_lower:
        while cov_quota < cov_thre:
            # FIXME implement a budget for early stopping
            # last_glb = global_lb.max()
            if Visited > branch_budget:
                time_cost = time.time() - start
                cov_dm_all, preimage_dict_all, dm_rec_all = get_preimage_info(domains, bound_lower, bound_upper)
                subdomain_num = len(domains)
                del domains
                if arguments.Config["preimage"]["save_cov"]:
                    save_path = os.path.join(result_dir, 'iteration_cov')
                    if bound_lower:
                        save_file = os.path.join(save_path,'{}_iter_cov_under.pt'.format(arguments.Config["data"]["dataset"]))
                    else:
                        save_file = os.path.join(save_path,'{}_iter_cov_over.pt'.format(arguments.Config["data"]["dataset"]))
                    iter_cov_quota = torch.tensor(iter_cov_quota)
                    torch.save(iter_cov_quota, save_file)
                if arguments.Config["preimage"]["quant"]:
                    save_path = os.path.join(result_dir, 'quant_analysis')
                    save_file = os.path.join(save_path,'{}_spec_{}_dm_{}'.format(arguments.Config["data"]["dataset"], arguments.Config["preimage"]["label"], subdomain_num))
                    with open(save_file, 'wb') as f:
                        pickle.dump((preimage_dict_all, dm_rec_all), f)                
                return Flag_covered, preimage_dict_all, Visited, time_cost, iter_cov_quota, subdomain_num
            # if args.smart:
            cov_quota = batch_verification_input_split(
                domains,
                net,
                batch,
                batch_spec,
                decision_thresh=rhs,
                shape=x.shape,
                bounding_method=bounding_method,
                branching_method=branching_method,
                stop_func=stop_func,
                bound_lower=bound_lower,
                bound_upper=bound_upper
            )
            
            # else:
            #     cov_quota = batch_verification_input_split_baseline(
            #         args,
            #         domains,
            #         net,
            #         batch,
            #         batch_spec,
            #         decision_thresh=rhs,
            #         shape=x.shape,
            #         bounding_method=bounding_method,
            #         branching_method=branching_method,
            #         stop_func=stop_func,
            #     )            
            # if record:
            #     cov_record.append([time.time() - start, cov_doms])
            print('--- Iteration {}, Cov quota {} ---'.format(num_iter, cov_quota))
            
            iter_cov_quota.append(cov_quota)
            if arguments.Config["preimage"]["save_process"]:
                cov_dm_all, preimage_dict_all, dm_rec_all = get_preimage_info(domains, bound_lower, bound_upper)
                save_path = os.path.join(result_dir, 'run_example')
                save_file = os.path.join(save_path,'{}_spec_{}_iter_{}_input'.format(arguments.Config["data"]["dataset"], arguments.Config["preimage"]["label"], num_iter))
                with open(save_file, 'wb') as f:
                    pickle.dump((preimage_dict_all, dm_rec_all), f)

            num_iter += 1
    elif bound_upper:    
        while cov_quota > cov_thre:
            # FIXME implement a budget for early stopping
            # last_glb = global_lb.max()
            if Visited > branch_budget:
                time_cost = time.time() - start
                cov_dm_all, preimage_dict_all, dm_rec_all = get_preimage_info(domains, bound_lower, bound_upper)
                subdomain_num = len(domains)
                del domains
                if arguments.Config["preimage"]["save_cov"]:
                    save_path = os.path.join(result_dir, 'iteration_cov')
                    if bound_lower:
                        save_file = os.path.join(save_path,'{}_iter_cov_under.pt'.format(arguments.Config["data"]["dataset"]))
                    else:
                        save_file = os.path.join(save_path,'{}_iter_cov_over.pt'.format(arguments.Config["data"]["dataset"]))
                    iter_cov_quota = torch.tensor(iter_cov_quota)
                    torch.save(iter_cov_quota, save_file)
                return Flag_covered, preimage_dict_all, Visited, time_cost, iter_cov_quota, subdomain_num
            # if args.smart:
            cov_quota = batch_verification_input_split(
                domains,
                net,
                batch,
                batch_spec,
                decision_thresh=rhs,
                shape=x.shape,
                bounding_method=bounding_method,
                branching_method=branching_method,
                stop_func=stop_func,
                bound_lower=bound_lower,
                bound_upper=bound_upper
            )
            # else:
            #     cov_quota = batch_verification_input_split_baseline(
            #         args,
            #         domains,
            #         net,
            #         batch,
            #         batch_spec,
            #         decision_thresh=rhs,
            #         shape=x.shape,
            #         bounding_method=bounding_method,
            #         branching_method=branching_method,
            #         stop_func=stop_func,
            #     )            
            # if record:
            #     cov_record.append([time.time() - start, cov_doms])
            print('--- Iteration {}, Cov quota {} ---'.format(num_iter, cov_quota))
            iter_cov_quota.append(cov_quota)
            if arguments.Config["preimage"]["save_process"]:
                cov_dm_all, preimage_dict_all, dm_rec_all = get_preimage_info(domains, bound_lower, bound_upper)
                save_path = os.path.join(result_dir, 'run_example')
                if bound_lower:
                    save_file = os.path.join(save_path,'{}_spec_{}_iter_{}_input_under'.format(arguments.Config["data"]["dataset"], arguments.Config["preimage"]["label"], num_iter))
                if bound_upper:
                    save_file = os.path.join(save_path,'{}_spec_{}_iter_{}_input_over'.format(arguments.Config["data"]["dataset"], arguments.Config["preimage"]["label"], num_iter))
                with open(save_file, 'wb') as f:
                    pickle.dump((preimage_dict_all, dm_rec_all), f)

 


            num_iter += 1
        
        
    time_cost = time.time() - start
    # xy: Before deleting the subdomains, record what we need for the final polyhedral rep 
    cov_dm_all, preimage_dict_all, dm_rec_all = get_preimage_info(domains, bound_lower, bound_upper)
    # print(f'Final domains: "{dm_rec_all}')
    subdomain_num = len(domains)
    if arguments.Config["preimage"]["quant"]:
        save_path = os.path.join(result_dir, 'quant_analysis')
        save_file = os.path.join(save_path,'{}_spec_{}_dm_{}'.format(arguments.Config["data"]["dataset"], arguments.Config["preimage"]["label"], subdomain_num))
        with open(save_file, 'wb') as f:
            pickle.dump((preimage_dict_all, dm_rec_all), f)
    del domains
    Flag_covered = True
    if arguments.Config["preimage"]["save_cov"]:
        save_path = os.path.join(result_dir, 'iteration_cov')
        if bound_lower:
            save_file = os.path.join(save_path,'{}_iter_cov_under.pt'.format(arguments.Config["data"]["dataset"]))
        else:
            save_file = os.path.join(save_path,'{}_iter_cov_over.pt'.format(arguments.Config["data"]["dataset"]))
        iter_cov_quota = torch.tensor(iter_cov_quota)
        torch.save(iter_cov_quota, save_file)
        # with open(save_file, 'wb') as f:
        #     pickle.dump(iter_cov_quota, f)        
    return Flag_covered, preimage_dict_all, Visited, time_cost, iter_cov_quota, subdomain_num
    # return Flag_covered, dm_rec_all, preimage_dict_all, Visited, cov_dm_all, time_cost, iter_cov_quota, subdomain_num
    # return global_lb.max(), None, glb_record, Visited, "safe"
   
   
   


def get_preimage_info(domains, bound_under, bound_over):
    dm_rec_all = []
    preimage_dict_all = []
    cov_dm_all = []
    if bound_under:
        for idx, dom in enumerate(domains):
            cov_dm_all.append(dom[0])
            # change the attribute lower bound into preimage_dict
            preimage_dict_all.append((dom[1],dom[2]))
            dm_rec_all.append((dom[5].cpu().detach().numpy(), dom[6].cpu().detach().numpy()))
    elif bound_over:
        for idx, dom in enumerate(domains):
            cov_dm_all.append(dom[0])
            # change the attribute lower bound into preimage_dict
            preimage_dict_all.append((dom[3],dom[4]))
            dm_rec_all.append((dom[5].cpu().detach().numpy(), dom[6].cpu().detach().numpy()))
            # print(dom[5], dom[6], dom[9], dom[0])
    return cov_dm_all, preimage_dict_all, dm_rec_all
    
    


def initial_check_preimage_approx(A_dict, thre):
    """check whether optimization on initial domain is successful"""
    # lbs: b, n_bounds (already multiplied with c in compute_bounds())
    preimage_dict = post_process_A(A_dict)
    assert (arguments.Config["preimage"]["under_approx"] or arguments.Config["preimage"]["over_approx"])
    if arguments.Config["preimage"]["under_approx"]:
        if arguments.Config['preimage']["quant"]:
            target_vol, cov_quota = calc_mc_esti_coverage_initial_input_under(preimage_dict)
        else:
            target_vol, cov_quota = calc_input_coverage_initial_input_under(preimage_dict)
        if cov_quota >= thre:  # check whether the preimage approx satisfies the criteria
            print("Reached by optmization on the initial domain!")
            return True, cov_quota, target_vol, preimage_dict
        else:
            return False, cov_quota, target_vol, preimage_dict
    else:
        target_vol, cov_quota = calc_input_coverage_initial_input_over(preimage_dict)
        if cov_quota <= thre:  # check whether the preimage approx satisfies the criteria
            print("Reached on the initial domain!")
            return True, cov_quota, target_vol, preimage_dict
        else:
            return False, cov_quota, target_vol, preimage_dict




def input_bab_approx_parallel_multi_no_prioritization(
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
    # FIXME maybe to add the coverage threshold to the global arguments, instead of setting it here.
    # xy: add the coverage stopping threshold
    cov_thre = args.threshold
    # the crown_lower/upper_bounds are dummy test_arguments here --- similar to refined_lower/upper_bounds, they are not used
    """ run input split bab """
    prop_mat_ori = net.c[0]

    start = time.time()
    # All supported test_arguments.
    global Visited, Flag_first_split, all_node_split, DFS_enabled

    timeout = timeout or preimage_arguments.Config["bab"]["timeout"]
    batch = preimage_arguments.Config["solver"]["batch_size"]
    # FIXME xy: added the record argument, we do not need it now 
    # record = test_arguments.Config["general"]["record_bounds"]
    bounding_method = preimage_arguments.Config["solver"]["bound_prop_method"]
    # FIXME xy: added the branch budget argument
    branch_budget = preimage_arguments.Config['bab']['branching']['input_split']['branch_budget']

    stop_func = stop_criterion_batch

    Visited, Flag_first_split, global_ub = 0, True, np.inf
    # xy: add the flag
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
    # split_depth = get_split_depth(dm_l)

    # compute storage depth
    use_slope = preimage_arguments.Config["solver"]["bound_prop_method"] == "alpha-crown"
    min_batch_size = (
        preimage_arguments.Config["solver"]["min_batch_size_ratio"]
        * preimage_arguments.Config["solver"]["batch_size"]
    )
    max_depth = max(int(math.log(max(min_batch_size, 1)) // math.log(2)), 1)
    storage_depth = min(max_depth, dm_l.shape[-1])


    # compute initial split idx
    # split_idx = input_split_branching(net, global_lb, dm_l, dm_u, lA, rhs, branching_method, None, None, None, storage_depth)
    # xy: do not order the input idx, select in a greedy manner from all possible ones
    in_dim = dm_l.shape[-1]
    # print('check input dim for split index', in_dim)
    batch_spec = len(dm_l)
    split_idx = torch.arange(in_dim)
    split_idx = split_idx.repeat(batch_spec, 1)
    
    # xy: lA used below should contain lA and lbias
    # xy: the under approx quality evaluation for the polytope based on initial domain (the whole domain)
    # xy: initial check whether satisfy the preimage underapproximation criterion
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


    # if test_arguments.Config["bab"]["batched_domain_list"]:
    #     domains = UnsortedInputDomainList(storage_depth, use_slope=use_slope)
    # else:
    # xy: use SortedInputDomain
    # domains = SortedInputDomainList()
    domains = UnsortedInputDomainList()
    # FIXME Change the InputDomain sorting conditions 
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

    # glb_record = [[time.time() - start, (global_lb - rhs).max().item()]]
    cov_record = [[time.time() - start, cov_quota]]
    iter_cov_quota = [cov_quota]
   
   
    num_iter = 1
    # sort_domain_iter = test_arguments.Config["bab"]["branching"]["input_split"]["sort_domain_interval"]
    enhanced_bound_initialized = False
    # while len(domains) > 0:
    
    while cov_quota < cov_thre:
        # FIXME implement a budget for early stopping
        # last_glb = global_lb.max()
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
        # if record:
        #     cov_record.append([time.time() - start, cov_doms])
        print('--- Iteration {}, Cov quota {} ---'.format(num_iter, cov_quota))
        iter_cov_quota.append(cov_quota)
        num_iter += 1
    time_cost = time.time() - start
    # xy: Before deleting the subdomains, record what we need for the final polyhedral rep 
    cov_dm_all, preimage_dict_all, dm_rec_all = get_preimage_info(domains)
    subdomain_num = len(domains)
    del domains
    Flag_covered = True
    return Flag_covered, dm_rec_all, preimage_dict_all, Visited, cov_dm_all, time_cost, iter_cov_quota, subdomain_num
    # return global_lb.max(), None, glb_record, Visited, "safe" 
