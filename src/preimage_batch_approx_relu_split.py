#########################################################################
##                                                                     ##
##     This program is licenced under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
"""Branch and bound for activation space split."""
import time
import os
import random
import numpy as np
import torch
import copy
import pickle
from collections import defaultdict

from auto_LiRPA.utils import stop_criterion_sum, stop_criterion_batch_any, stop_criterion_batch_topk
from branching_domains import merge_domains_params, SortedReLUDomainList, BatchedReLUDomainList
from branching_heuristics import choose_node_parallel_FSB, choose_node_parallel_crown, choose_node_parallel_kFSB, choose_node_parallel_preimg
from torch.distributions import Uniform

import arguments

from branching_domains import select_batch
# from adv_domains import AdvExamplePool
# from bab_attack import beam_mip_attack, find_promising_domains, bab_attack
from cut_utils import fetch_cut_from_cplex, generate_cplex_cuts, clean_net_mps_process, cplex_update_general_beta

from test_polyhedron_util import post_process_A, calc_input_coverage_initial_image_under, calc_input_coverage_initial_image_over
from test_polyhedron_util import calc_relu_refine_approx_coverage_image_under, calc_relu_refine_approx_coverage_image_over
from test_polyhedron_util import calc_relu_refine_approx_coverage
from preimage_model_utils import build_model_activation

Visited, Flag_first_split = 0, True
Use_optimized_split = False
all_node_split = False
total_pickout_time = total_decision_time = total_solve_time = total_add_time = 0.0


def build_history(history, split, orig_lbs, orig_ubs):
    """
    Generate fake history and fake lower and upper bounds for new domains
    history: [num_domain], history of the input domains
    split: [num_copy * num_domain], split decision for each new domain.
    orig_lbs, orig_ubs: [num_relu_layer, num_copy, num_domain, relu_layer.shape]
    """
    new_history = []
    num_domain = len(history)
    num_split = len(split)//num_domain

    num_layer = len(orig_lbs)

    def generate_history(heads, splits, orig_lbs, orig_ubs, domain_idx):
        '''
        Generate [num_copy] fake history and fake lower and upper bounds for an input domain.
        '''
        for pos in range(num_split-1):
            num_history = len(heads)
            for i in range(num_history):
                decision_layer = splits[pos*num_domain+domain_idx][0][0]
                decision_index = splits[pos*num_domain+domain_idx][0][1]

                for l in range(num_layer):
                    orig_ubs[l][num_history+i][domain_idx] = orig_ubs[l][i][domain_idx]
                    orig_lbs[l][num_history+i][domain_idx] = orig_lbs[l][i][domain_idx]

                orig_lbs[decision_layer][i][domain_idx].view(-1)[decision_index] = 0.0
                heads[i][decision_layer][0].append(decision_index)
                heads[i][decision_layer][1].append(1.0)
                heads.append(copy.deepcopy(heads[i]))
                orig_ubs[decision_layer][num_history+i][domain_idx].view(-1)[decision_index] = 0.0
                heads[-1][decision_layer][1][-1] = -1.0
        return heads
    new_history_list = []
    for i in range(num_domain):
        new_history_list.append(generate_history([history[i]], split, orig_lbs, orig_ubs, i))

    for i in range(len(new_history_list[0])):
        for j in range(num_domain):
            new_history.append(new_history_list[j][i])
    # num_copy * num_domain
    return new_history, orig_lbs, orig_ubs

def load_act_vecs(dataset_tp):
    if arguments.Config["model"]["onnx_path"] is None:
        act_file = os.path.join(arguments.Config['preimage']["sample_dir"], 'act_vec_{}_{}.pkl'.format(dataset_tp, arguments.Config["preimage"]["atk_tp"]))
    else:
        act_file = os.path.join(arguments.Config['preimage']["sample_dir"], '{}/act_vec_{}.pkl'.format(dataset_tp, dataset_tp))
        # if dataset_tp == "vcas":
        #     act_file = os.path.join(arguments.Config['preimage']["sample_dir"], 'act_vec_{}_{}.pkl'.format(dataset_tp, arguments.Config["preimage"]["upper_time_loss"]))
        # else:
        #     act_file = os.path.join(arguments.Config['preimage']["sample_dir"], 'act_vec_{}.pkl'.format(dataset_tp))
    with open(act_file, 'rb') as f:
        activation = pickle.load(f)
    if "MNIST" in dataset_tp:
        if arguments.Config["model"]["name"] == "mnist_3_128":
            pre_relu_layer = ['1', '3']
        elif arguments.Config["model"]["name"] == "mnist_6_100":
            pre_relu_layer = ['2', '4', '6', '8', '10']
    elif dataset_tp == "auto_park":
        pre_relu_layer = ['2']
    elif dataset_tp == "auto_park_part":
        pre_relu_layer = ['2', '4']
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
def batch_verification(tot_ambi_nodes_sample, mask_sample_ori, score_relus_ori, d, y, net, batch, pre_relu_indices, growth_rate, fix_intermediate_layer_bounds=True,
                    stop_func=stop_criterion_sum, multi_spec_keep_func=lambda x: torch.all(x, dim=-1), bound_lower=True, bound_upper=False):
    global Visited, Flag_first_split
    global Use_optimized_split
    global total_pickout_time, total_decision_time, total_solve_time, total_add_time

    opt_intermediate_beta = False
    branching_method = arguments.Config['bab']['branching']['method']
    branching_reduceop = arguments.Config['bab']['branching']['reduceop']
    get_upper_bound = arguments.Config["bab"]["get_upper_bound"]
    branching_candidates = arguments.Config["bab"]["branching"]["candidates"]

    total_time = time.time()

    pickout_time = time.time()

    domains_params = d.pick_out(batch=batch, device=net.x.device)
    # Note that lAs is removed
    mask, orig_lbs, orig_ubs, slopes, betas, intermediate_betas, selected_domains, cs, rhs = domains_params
    history = [sd.history for sd in selected_domains]
    # history_sample = [sd.history_sample for sd in selected_domains]

    # if split_num == tot_ambi_nodes_sample:
    #     print('all nodes are split for the selected domain!!')
    #     print('{} domains visited'.format(Visited))
    #     selected_domains[0].valid = False
    #     d.add(selected_domains[0])
        
    # print('-' * 20)
    # print('mask shape:', [x.shape for x in mask])
    # print('lAs shape:', [x.shape for x in lAs])
    # print('orig_lbs shape:', [x.shape for x in orig_lbs])
    # print('orig_ubs shape:', [x.shape for x in orig_ubs])
    # print('slopes shape:', len(slopes), '*' dict([(x, dict([(xx, yy.shape) for xx, yy in y.items()])) for x, y in slope[0].items()]))
    # print('Cs.shape:', cs.shape)
    # print('-' * 20)
    
    pickout_time = time.time() - pickout_time
    total_pickout_time += pickout_time

    decision_time = time.time()


        
    split_history = [sd.split_history for sd in selected_domains]

    # Here we check the length of current domain list.
    # If the domain list is small, we can split more layers.
    # min_batch_size = min(arguments.Config["solver"]["min_batch_size_ratio"]*arguments.Config["solver"]["batch_size"], batch)

    # if orig_lbs[0].shape[0] < min_batch_size:
    #     # Split multiple levels, to obtain at least min_batch_size domains in this batch.
    #     split_depth = int(np.log(min_batch_size)/np.log(2))

    #     if orig_lbs[0].shape[0] > 0:
    #         split_depth = max(int(np.log(min_batch_size/orig_lbs[0].shape[0])/np.log(2)), 0)
    #     split_depth = max(split_depth, 1)
    # else:
    split_depth = 1
    # def update_mask_unstable_sample(history, mask_sample, score_relus):
    #     num_copy = len(history)
    #     new_mask_sample, new_score_relus = [], []
    #     for layer_mask in mask_sample:
    #         new_mask_sample.append(layer_mask.repeat(num_copy, *[1]*(len(layer_mask.shape)-1)))
    #     for layer_score in score_relus:
    #         new_score_relus.append(layer_score.repeat(num_copy, *[1]*(len(layer_score.shape)-1)))
    #     for d, dm_his in enumerate(history):
    #         for l, layer_his in enumerate(dm_his):
    #             if len(layer_his[0]) == 0:
    #                 continue
    #             neuron_idxs = layer_his[0]
    #             new_mask_sample[l][d][neuron_idxs] = 0
    #             new_score_relus[l][d][neuron_idxs] = -1
    #     return new_mask_sample, new_score_relus
    def update_mask_unstable_sample_ind(history, mask_sample, score_relus):
        num_copy = len(history)
        new_mask_sample, new_score_relus = copy.deepcopy(mask_sample), copy.deepcopy(score_relus)
        for l, layer_his in enumerate(history):
            if len(layer_his[0]) == 0:
                continue
            neuron_idxs = layer_his[0]
            print('check mask and score relus', new_mask_sample[l][0][neuron_idxs], new_score_relus[l][0][neuron_idxs])
            new_mask_sample[l][0][neuron_idxs] = 0
            new_score_relus[l][0][neuron_idxs] = -1
        return new_mask_sample, new_score_relus
    history_mask = False
    for l, layer_his in enumerate(history[0]):
        if len(layer_his[0]) > 0:
            history_mask = True
            break
    mask_sample = copy.deepcopy(mask_sample_ori)
    score_relus = copy.deepcopy(score_relus_ori)        
    if history_mask:
        # This is the case where the mask and relu scores need to be updated
        mask_sample, score_relus = update_mask_unstable_sample_ind(history[0], mask_sample, score_relus)
                
    print("batch: ", orig_lbs[0].shape, "pre split depth: ", split_depth)
    # Increase the maximum number of candidates for fsb and kfsb if there are more splits needed.
    branching_candidates = max(branching_candidates, split_depth)

    if branching_method == 'babsr':
        branching_decision, split_depth = choose_node_parallel_crown(orig_lbs, orig_ubs, mask, net, pre_relu_indices, lAs,
                                                        batch=batch, branching_reduceop=branching_reduceop, split_depth=split_depth, cs=cs, rhs=rhs)
    elif branching_method == 'fsb':
        branching_decision, split_depth = choose_node_parallel_FSB(orig_lbs, orig_ubs, mask, net, pre_relu_indices, lAs,
                                        branching_candidates=branching_candidates, branching_reduceop=branching_reduceop,
                                        slopes=slopes, betas=betas, history=history, split_depth=split_depth, cs=cs, rhs=rhs)
    elif branching_method.startswith('kfsb'):
        branching_decision, split_depth = choose_node_parallel_kFSB(orig_lbs, orig_ubs, mask, net, pre_relu_indices, lAs,
                                        branching_candidates=branching_candidates, branching_reduceop=branching_reduceop,
                                        slopes=slopes, betas=betas, history=history, split_depth=split_depth, cs=cs, rhs=rhs,
                                        method=branching_method)
    elif branching_method == 'preimg':
        branching_decision, split_depth = choose_node_parallel_preimg(mask_sample, score_relus, orig_lbs, orig_ubs, mask, net, pre_relu_indices,
                                                        batch=batch, branching_reduceop=branching_reduceop, split_depth=split_depth, cs=cs, rhs=rhs)
    else:
        raise ValueError(f'Unsupported branching method "{branching_method}" for relu splits.')
    if arguments.Config["data"]["dataset"] == "auto_park_part" and arguments.Config["preimage"]["save_process"]:
        if bound_lower:
            branching_decision = [(0,9)]
        if bound_upper:
            # branching_decision = [(1,6)]
            branching_decision = [(0,4)]
            
        split_depth = 1
    print("batch: ", orig_lbs[0].shape, "post split depth: ", split_depth)

    if len(branching_decision) < len(mask[0]):
        print('all nodes are split!!')
        print('{} domains visited'.format(Visited))
        global all_node_split
        all_node_split = True
        if not arguments.Config["solver"]["beta-crown"]["all_node_split_LP"]:
            global_lb = selected_domains[0].lower_bound - selected_domains[0].threshold
            for i in range(1, len(selected_domains)):
                if max(selected_domains[i].lower_bound - selected_domains[i].threshold) <= max(global_lb):
                    global_lb = selected_domains[i].lower_bound - selected_domains[i].threshold
            return global_lb, np.inf

    print('splitting decisions: ')
    for l in range(split_depth):
        print("split level {}".format(l), end=": ")
        for b in range(min(10, len(history))):
            print(branching_decision[l*len(history) + b], end=" ")
        print('')
    # print the first two split for first 10 domains.

    if not Use_optimized_split:
        split = {}
        # split["decision"]: selected domains (next batch/2)->node list->node: [layer, idx]
        split["decision"] = [[bd] for bd in branching_decision]
        # split["split"]: selected domains (next batch/2)->node list->float coefficients
        split["coeffs"] = [[1.] for i in range(len(branching_decision))]
    else:
        split = {}
        num_nodes = 3
        split["decision"] = [[[2, i] for i in range(num_nodes)] for bd in branching_decision]
        split["coeffs"] = [[random.random() * 0.001 - 0.0005 for j in range(num_nodes)] for i in
                            range(len(branching_decision))]

    decision_time = time.time() - decision_time
    total_decision_time += decision_time

    solve_time = time.time()
    single_node_split = True
    # copy the original lbs

    num_copy = (2**(split_depth-1))

    if num_copy > 1:
        orig_lbs = [lb.unsqueeze(0).repeat(num_copy, *[1]*len(lb.shape)) for lb in orig_lbs]
        orig_ubs = [ub.unsqueeze(0).repeat(num_copy, *[1]*len(ub.shape)) for ub in orig_ubs]
        # 4 * [num_copy, num_domain, xxx]

        num_domain = len(history)

        # create fake history for each branch.
        # TODO: set origlbs and orig_ubs
        history, orig_lbs, orig_ubs = build_history(history, split['decision'], orig_lbs, orig_ubs)
        # num_domains -> num_domains * (2**num_split_layer)

        # set the slopes for each branch
        for k, v in slopes.items():
            for kk, vv in v.items():
                v[kk] = torch.cat([vv] * num_copy, dim=2)

        # create fake split_history for each branch.
        split_history = split_history * num_copy

        # cs needs to repeat
        cs = torch.cat([cs] * num_copy, dim=0)

        new_betas = []
        new_intermediate_betas = []
        for i in range(num_copy):
            for j in range(len(betas)):
                new_betas.append(betas[j])
                new_intermediate_betas.append(intermediate_betas[j])
        betas = new_betas
        intermediate_betas = new_intermediate_betas

        orig_lbs = [lb.view(-1, *lb.shape[2:]) for lb in orig_lbs]
        orig_ubs = [ub.view(-1, *ub.shape[2:]) for ub in orig_ubs]

        # create split for num_copy * num_domain
        # we only keep the last split since the first few ones has been split with build_history
        split['decision'] = split['decision'][-num_domain:] * num_copy
        split['coeffs'] = split['coeffs'][-num_domain:] * num_copy

        branching_decision = branching_decision[-num_domain:] * num_copy
        rhs = torch.cat([rhs] * num_copy, dim=0)
    def calc_splitting_history(histories,branching_decisions):
        '''
        calculate the updated history for the two new subdomains
        '''
        # assert len(histories) == 1
        # This is the history of one selected domain
        num_relu_layer =  len(histories[0]) 
        print("# of relu layers:", num_relu_layer)
        left_history = copy.deepcopy(histories[0])
        relu_layer_idx = branching_decisions[0][0]
        left_history[relu_layer_idx][0].append(branching_decisions[0][1])
        left_history[relu_layer_idx][1].append(+1.0)
        # sanity check repeated split
        if branching_decisions[0][1] in histories[0][relu_layer_idx][0]:
            print('BUG!!! repeated split!')
            print(histories[0][relu_layer_idx])
            print(branching_decisions[0])
            raise RuntimeError
        right_history = copy.deepcopy(histories[0])
        right_history[relu_layer_idx][0].append(branching_decisions[0][1])
        right_history[relu_layer_idx][1].append(-1.0)
        return left_history, right_history


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
    acti_vecs = load_act_vecs(arguments.Config["data"]["dataset"])
    left_history, right_history = calc_splitting_history(history, branching_decision)
    sample_left_idx = calc_history_idxs(acti_vecs, left_history)
    sample_right_idx = calc_history_idxs(acti_vecs, right_history)
    if len(sample_left_idx) == 0 and len(sample_right_idx) == 0:
        print("check why")

    # left_history_sample, right_history_sample = calc_splitting_history(history_sample, branching_decision)
    # sample_left_idx = calc_history_idxs(acti_vecs, left_history_sample)
    # sample_right_idx = calc_history_idxs(acti_vecs, right_history_sample)

    # if len(sample_left_idx) == 0 and len(sample_right_idx) == 0:
    #     flag_infeasible = True
    # else:
    #     flag_infeasible = False
    #     if len(sample_left_idx) == 0:
    #         left_history_sample = copy.deepcopy(history_sample[0])
    #         sample_left_idx = calc_history_idxs(acti_vecs, left_history_sample)
    #     if len(sample_right_idx) == 0:
    #         right_history_sample = copy.deepcopy(history_sample[0])
    #         sample_right_idx = calc_history_idxs(acti_vecs, right_history_sample)
    # if flag_infeasible:
    #     left_history_sample = copy.deepcopy(history_sample[0])
    #     right_history_sample = copy.deepcopy(history_sample[0])
    #     sample_left_idx = calc_history_idxs(acti_vecs, left_history_sample)
    #     sample_right_idx = calc_history_idxs(acti_vecs, right_history_sample)
        
    # if len(sample_left_idx) == 0 and len(sample_right_idx) == 0:
    #     flag_next_split 
    # Caution: we use "all" predicate to keep the domain when multiple specs are present: all lbs should be <= threshold, otherwise pruned
    # maybe other "keeping" criterion needs to be passed here
    ret = net.get_lower_bound(orig_lbs, orig_ubs, split, slopes=slopes, history=history,sample_left_idx=sample_left_idx, sample_right_idx=sample_right_idx,
                                split_history=split_history, fix_intermediate_layer_bounds=fix_intermediate_layer_bounds, betas=betas,
                                single_node_split=single_node_split, intermediate_betas=intermediate_betas, cs=cs, decision_thresh=rhs, rhs=rhs,
                                stop_func=stop_func(torch.cat([rhs, rhs])), multi_spec_keep_func=multi_spec_keep_func, bound_lower=bound_lower, bound_upper=bound_upper)

    dom_ub, dom_lb, dom_ub_point, lAs, A, dom_lb_all, dom_ub_all, slopes, split_history, betas, intermediate_betas, primals, dom_cs = ret
    solve_time = time.time() - solve_time
    total_solve_time += solve_time
    add_time = time.time()
    batch = len(branching_decision)
    # If intermediate layers are not refined or updated, we do not need to check infeasibility when adding new domains.
    check_infeasibility = not (single_node_split and fix_intermediate_layer_bounds)

    depths = [domain.depth + split_depth - 1 for domain in selected_domains] * num_copy * 2

    old_d_len = len(d)
    


            
    # NOTE evaluate the coverage quality after splitting
    if bound_lower:
        cov_subdomain_info, A_b_dict = calc_relu_refine_approx_coverage_image_under(A, y, sample_left_idx, sample_right_idx)
    # cov_subdomain_info, A_b_dict = calc_relu_refine_approx_coverage_image(A, A_dict_relus, left_history, right_history, split_relu_indices, y)
    if bound_upper:
        cov_subdomain_info, A_b_dict = calc_relu_refine_approx_coverage_image_over(A, y, sample_left_idx, sample_right_idx)
    cov_vol_info = [cov[0] for cov in cov_subdomain_info]
    cov_quota_info = [cov[1] for cov in cov_subdomain_info]
    if arguments.Config["solver"]["beta-crown"]["all_node_split_LP"]:
        for domain_idx in range(len(depths)):
            # get tot_ambi_nodes
            dlb, dub = [dlbs[domain_idx: domain_idx + 1] for dlbs in dom_lb_all],  [dubs[domain_idx: domain_idx + 1] for dubs in dom_ub_all]
            decision_threshold = rhs.to(dom_lb[0].device, non_blocking=True)[domain_idx if domain_idx < (len(dom_lb)//2) else domain_idx - (len(dom_lb)//2)]
            # print(depths[domain_idx] + 1, dlb[-1], decision_threshold, torch.all(dlb[-1] <= decision_threshold))
            if depths[domain_idx] + 1 == net.tot_ambi_nodes  and torch.all(dlb[-1] <= decision_threshold):
                lp_status, dlb, adv = net.all_node_split_LP(dlb, dub, decision_threshold)
                print(f"using lp to solve all split node domain {domain_idx}/{len(dom_lb)}, results {dom_lb[domain_idx]} -> {dlb}, {lp_status}")
                # import pdb; pdb.set_trace()
                if lp_status == "unsafe":
                    # unsafe cases still needed to be handled! set to be unknown for now!
                    all_node_split = True
                    return dlb, np.inf
                dom_lb_all[-1][domain_idx] = dlb
                dom_lb[domain_idx] = dlb
    left_right_his = [left_history, right_history]
    # left_right_his_sample = [left_history_sample, right_history_sample]
    if bound_lower:
        d.add(tot_ambi_nodes_sample, cov_vol_info, cov_quota_info, A_b_dict["lA"], A_b_dict["lbias"], dom_lb, dom_ub, dom_lb_all, dom_ub_all, history, left_right_his, depths, slopes, betas, split_history,
                branching_decision, rhs, intermediate_betas, check_infeasibility, dom_cs, (2*num_copy)*batch)
    if bound_upper: 
        d.add(tot_ambi_nodes_sample, cov_vol_info, cov_quota_info, A_b_dict["uA"], A_b_dict["ubias"], dom_lb, dom_ub, dom_lb_all, dom_ub_all, history, left_right_his, depths, slopes, betas, split_history,
                branching_decision, rhs, intermediate_betas, check_infeasibility, dom_cs, (2*num_copy)*batch)
    total_vol = 0
    cov_vol = 0
    
    # if len(d) < old_d_len:
    #     print("check why")
    #     print("should not happen")
    for i, subdm in enumerate(d.domains):
        total_vol += subdm.preimg_vol
        cov_vol += subdm.preimg_vol * subdm.preimg_cov
        
    total_cov_quota = cov_vol / total_vol
    print('length of domains:', len(d))
    print('Coverage quota:', total_cov_quota)
    split_all = True
    for i, subdm in enumerate(d.domains):
        if subdm.valid:
            split_all = False
            break
    if split_all:
        print("exhausting search achieved")
        return total_cov_quota, split_all

    

    # Visited += (len(selected_domains) * num_copy) * 2 - (len(d) - old_d_len)
    Visited += 2
    # if len(d) > 0:
    #     if get_upper_bound:
    #         print('Current worst splitting domains [lb, ub] (depth):')
    #     else:
    #         print('Current worst splitting domains lb-rhs (depth):')
    #     if arguments.Config["bab"]["cut"]["enabled"] and arguments.Config["bab"]["cut"]["cplex_cuts"] and arguments.Config["bab"]["cut"]["cplex_cuts_revpickup"]:
    #         printed_d = d.get_min_domain(20, rev_order=True)
    #     else:
    #         printed_d = d.get_min_domain(20)
    #     for i in printed_d:
    #         if get_upper_bound:
    #             print(f'[{(i.lower_bound - i.threshold).max():.5f}, {(i.upper_bound - i.threshold).min():5f}] ({i.depth})', end=', ')
    #         else:
    #             print(f'{(i.lower_bound - i.threshold).max():.5f} ({i.depth})', end=', ')
    #     print()
    #     if hasattr(d, 'sublist'):
    #         print(f'Max depth domain: [{d.sublist[0].domain.lower_bound}, {d.sublist[0].domain.upper_bound}] ({d.sublist[0].domain.depth})')
    add_time = time.time() - add_time
    total_add_time += add_time

    total_time = time.time() - total_time
    print(f'Total time: {total_time:.4f}\t pickout: {pickout_time:.4f}\t decision: {decision_time:.4f}\t get_bound: {solve_time:.4f}\t add_domain: {add_time:.4f}')
    print(f'Accumulated time:\t pickout: {total_pickout_time:.4f}\t decision: {total_decision_time:.4f}\t get_bound: {total_solve_time:.4f}\t add_domain: {total_add_time:.4f}')

    # if len(d) > 0:
    #     if arguments.Config["bab"]["cut"]["enabled"] and arguments.Config["bab"]["cut"]["cplex_cuts"] and arguments.Config["bab"]["cut"]["cplex_cuts_revpickup"]:
    #         worst_domain = d.get_min_domain(1 ,rev_order=True)
    #         global_lb = worst_domain[-1].lower_bound - worst_domain[-1].threshold
    #     else:
    #         worst_domain = d.get_min_domain(1 ,rev_order=False)
    #         global_lb = worst_domain[0].lower_bound - worst_domain[0].threshold
    # else:
    #     print("No domains left, verification finished!")
    #     print('{} domains visited'.format(Visited))
    #     return torch.tensor(arguments.Config["bab"]["decision_thresh"] + 1e-7), np.inf

    # batch_ub = np.inf
    # if get_upper_bound:
    #     batch_ub = min(dom_ub)
    #     print(f"Current (lb-rhs): {global_lb.max()}, ub:{batch_ub}")
    # else:
    #     print(f"Current (lb-rhs): {global_lb.max()}")

    print('{} domains visited'.format(Visited))
    return total_cov_quota, split_all


def cut_verification(d, net, pre_relu_indices, fix_intermediate_layer_bounds=True):
    decision_thresh = arguments.Config["bab"]["decision_thresh"]
    get_upper_bound = arguments.Config["bab"]["get_upper_bound"]
    lp_cut_enabled = arguments.Config["bab"]["cut"]["lp_cut"]
    cplex_cuts = arguments.Config["bab"]["cut"]["cplex_cuts"]

    # construct the cut splits
    # change to only create one domain and make sure the other is infeasible
    split = {}
    if cplex_cuts:
        generate_cplex_cuts(net)

    if net.cutter.cuts is not None:
        split["cut"] = net.cutter.cuts
        split["cut_timestamp"] = net.cutter.cut_timestamp
    else:
        print('Cut is not present from cplex or predefined cut yet, direct return from cut init')
        return None, None
    return None, None

# NOTE
def initial_check_preimage_approx(A_dict, thre, label):
    """check whether optimization on initial domain is successful"""
    # lbs: b, n_bounds (already multiplied with c in compute_bounds())
    preimage_dict = post_process_A(A_dict)
    assert (arguments.Config["preimage"]["under_approx"] or arguments.Config["preimage"]["over_approx"])
    if arguments.Config["preimage"]["under_approx"]:
        target_vol, cov_quota = calc_input_coverage_initial_image_under(preimage_dict, label)
        if cov_quota >= thre:  # check whether the preimage approx satisfies the criteria
            print("Reached by optmization on the initial domain!")
            return True, cov_quota, target_vol, preimage_dict
        else:
            return False, cov_quota, target_vol, preimage_dict
    else:
        target_vol, cov_quota = calc_input_coverage_initial_image_over(preimage_dict, label)
        if cov_quota <= thre:  # check whether the preimage approx satisfies the criteria
            print("Reached by optmization on the initial domain!")
            return True, cov_quota, target_vol, preimage_dict
        else:
            return False, cov_quota, target_vol, preimage_dict
def calc_pixel_pos(attack_tp):
    if 'patch' in attack_tp:
        pixel_pos = []
        xs, ys = arguments.Config["preimage"]["patch_h"], arguments.Config["preimage"]["patch_v"]
        xe = xs + arguments.Config["preimage"]["patch_len"]
        ye = ys + arguments.Config["preimage"]["patch_width"]
        for i in range(xs, xe):
            for j in range(ys, ye):
                pixel_pos.append(i * 28 + j)  
    elif attack_tp == 'l0_rand':
        pixel_pos = []        
        random.seed(arguments.Config["general"]["seed"])
        l0_norm = arguments.Config["preimage"]["l0_norm"]
        length = 28
        width = 28
        for _ in range(l0_norm):
            x = random.randint(0, length-1)
            y = random.randint(0, width-1)
            pixel_pos.append(x * 28 + y)
    elif attack_tp == "l0_sensitive":
        grad_pix_file = os.path.join(arguments.Config["preimage"]["sample_dir"], "grad_pixel.pkl")
        with open(grad_pix_file, 'rb') as f:
            grad_pix_list = pickle.load(f)
        pixel_pos = []
        for grad_pix in grad_pix_list:
            x = grad_pix[1][0]
            y = grad_pix[1][1]
            pixel_pos.append(x * 28 + y)
    return pixel_pos
def get_act_vecs(prop_samples, model, dataset_tp):
    if "MNIST" in dataset_tp:
        if arguments.Config["model"]["name"] == "mnist_3_128":
            pre_relu_layer = ['1', '3']
        elif arguments.Config["model"]["name"] == "mnist_6_100":
            pre_relu_layer = ['2', '4', '6', '8', '10']
    elif dataset_tp == "auto_park":
        pre_relu_layer = ['2']
    elif dataset_tp == "auto_park_part":
        pre_relu_layer = ['2', '4']
    elif "vcas" in dataset_tp:
        pre_relu_layer = ['1']
    node_types = [m for m in list(model.modules())]
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    for i, layer in enumerate(pre_relu_layer):
        model_module = node_types[int(layer)]
        model_module.register_forward_hook(get_activation(layer))
        output = model(prop_samples)
        # print(activation[layer].shape)  
    return pre_relu_layer, activation
def calc_mask_concrete_samples_from_unstable_idx(x, net, y):
    sample_num = arguments.Config['preimage']["sample_num"]
    atk_tp = arguments.Config["preimage"]["atk_tp"]
    model_tp = arguments.Config["model"]["name"]
    spec_epsilon = arguments.Config["specification"]["epsilon"]
    multi_spec = arguments.Config["preimage"]["multi_spec"]
    x_L, x_U = x.ptb.x_L, x.ptb.x_U
    img_shape = [-1] + list(x.shape[1:])
    img = x.data.reshape(x.shape[0], -1)
    x_L = x_L.reshape(x_L.shape[0], -1)
    x_U = x_U.reshape(x_U.shape[0], -1)
    torch.manual_seed(arguments.Config["general"]["seed"])
    # torch.manual_seed(arguments.Config["general"]["seed"]) 
    if multi_spec:
        prop_samples = Uniform(x_L, x_U).sample([sample_num])
    else:
        if atk_tp == 'l_inf':
            prop_samples = Uniform(x_L, x_U).sample([sample_num])
            prop_samples = torch.squeeze(prop_samples).reshape(img_shape)
        else:
            prop_samples = img.repeat(sample_num,1)
            pixel_pos = calc_pixel_pos(atk_tp)
            pixel_pos = torch.tensor(pixel_pos).to(arguments.Config["general"]["device"])
            pixel_lower = x_L[0][pixel_pos]
            pixel_upper = x_U[0][pixel_pos]
            pixel_vals = Uniform(pixel_lower, pixel_upper).sample([sample_num])
            prop_samples[:, pixel_pos] = pixel_vals
            prop_samples = prop_samples.reshape(img_shape)
    torch.save(prop_samples, os.path.join(arguments.Config['preimage']["sample_dir"], 'sample_{}_{}.pt'.format(arguments.Config["data"]["dataset"],atk_tp)))
    model = net.model_ori
    model.eval()   
    # print(model.state_dict())
    device = arguments.Config["general"]["device"]
    model = model.to(device)
    orig_pred = model(x)
    orig_pred = orig_pred.argmax()
    print(f"orig_pred: {orig_pred}")
    if multi_spec:
        for i in range(prop_samples.shape[1]):
            predicted = model(prop_samples[:,i,:])
            predicted = predicted.argmax(dim=1).cpu().detach().numpy()
            idxs = np.where(predicted==y)[0]
            print(f'length of label {len(idxs)} preserved samples for spec {i}')             
    else:  
        predicted = model(prop_samples)
        predicted = predicted.argmax(dim=1).cpu().detach().numpy()
        idxs = np.where(predicted==y)[0]
        print('length of label preserved samples', len(idxs)) 
    pre_relu_layer, activation = get_act_vecs(prop_samples, model, arguments.Config["data"]["dataset"])
    act_file = os.path.join(arguments.Config['preimage']["sample_dir"], 'act_vec_{}_{}.pkl'.format(arguments.Config["data"]["dataset"], atk_tp))
    with open(act_file, 'wb') as f:
        pickle.dump(activation, f)
    mask_sample = []
    score_all = []
    if atk_tp == "l_inf":
        unstable_idx_file = os.path.join(arguments.Config["preimage"]["sample_dir"], f"unstable_idxs_all_{model_tp}_{spec_epsilon}")
        with open(unstable_idx_file, "rb") as f:
            unstable_indices = pickle.load(f)
    elif atk_tp == "patch":
        patch_len = arguments.Config["preimage"]["patch_len"]
        patch_wid = arguments.Config["preimage"]["patch_width"]
        patch_h = arguments.Config["preimage"]["patch_h"]
        if patch_h == 0:
            unstable_idx_file = os.path.join(arguments.Config["preimage"]["sample_dir"], f"unstable_idxs_all_{atk_tp}_{patch_len}_{patch_wid}_{patch_h}")
        else:
            unstable_idx_file = os.path.join(arguments.Config["preimage"]["sample_dir"], f"unstable_idxs_all_{atk_tp}_{patch_len}_{patch_wid}")
        with open(unstable_idx_file, "rb") as f:
            unstable_indices = pickle.load(f)
    for i, layer in enumerate(pre_relu_layer):
        act_vec = activation[layer]
        samples_lb = torch.min(act_vec, dim=0).values
        samples_ub = torch.max(act_vec, dim=0).values
        # 1 is unstable neuron, 0 is stable neuron.
        mask_tmp = torch.logical_and(samples_lb<0, samples_ub>0).float()
        unstable_idx = unstable_indices[i]
        mask_tmp[unstable_idx] = 1                
        mask_sample.append(torch.unsqueeze(mask_tmp,0))
        score_tmp = []
        # unstable_idx = torch.squeeze(mask_tmp.nonzero(),1)
        # unstable_indices.append(unstable_idx)
        if unstable_idx.shape[0]:
            pos_sample_mask = act_vec[:, unstable_idx] >= 0
            # neg_sample_mask = act_vec[:, unstable_idx] < 0
            for j in range(unstable_idx.shape[0]):
                pos_sample_idx_neuron = pos_sample_mask[:,j].nonzero()
                # neg_sample_idx_neuron = neg_sample_mask[:,j].nonzero()
                pos_freq = pos_sample_idx_neuron.shape[0]
                neg_freq = sample_num - pos_freq
                # print(pos_sample_idx_neuron.shape)
                score_tmp.append(sample_num-abs(pos_freq-neg_freq))
            score_all.append(score_tmp)                 
        else:
            score_all.append(score_tmp)                       
    return mask_sample, score_all, unstable_indices 

def calc_mask_concrete_samples(x, net, y):
    sample_num = arguments.Config['preimage']["sample_num"]
    atk_tp = arguments.Config["preimage"]["atk_tp"]
    multi_spec = arguments.Config["preimage"]["multi_spec"]
    x_L, x_U = x.ptb.x_L, x.ptb.x_U
    img_shape = [-1] + list(x.shape[1:])
    img = x.data.reshape(x.shape[0], -1)
    x_L = x_L.reshape(x_L.shape[0], -1)
    x_U = x_U.reshape(x_U.shape[0], -1)
    torch.manual_seed(arguments.Config["general"]["seed"])
    # torch.manual_seed(arguments.Config["general"]["seed"]) 
    if multi_spec:
        prop_samples = Uniform(x_L, x_U).sample([sample_num])
    else:
        if atk_tp == 'l_inf':
            prop_samples = Uniform(x_L, x_U).sample([sample_num])
            prop_samples = torch.squeeze(prop_samples).reshape(img_shape)
        else:
            prop_samples = img.repeat(sample_num,1)
            pixel_pos = calc_pixel_pos(atk_tp)
            pixel_pos = torch.tensor(pixel_pos).to(arguments.Config["general"]["device"])
            pixel_lower = x_L[0][pixel_pos]
            pixel_upper = x_U[0][pixel_pos]
            pixel_vals = Uniform(pixel_lower, pixel_upper).sample([sample_num])
            prop_samples[:, pixel_pos] = pixel_vals
            prop_samples = prop_samples.reshape(img_shape)
    torch.save(prop_samples, os.path.join(arguments.Config['preimage']["sample_dir"], 'sample_{}_{}.pt'.format(arguments.Config["data"]["dataset"],atk_tp)))
    model = net.model_ori
    model.eval()   
    # print(model.state_dict())
    device = arguments.Config["general"]["device"]
    model = model.to(device)
    orig_pred = model(x)
    orig_pred = orig_pred.argmax()
    print(f"orig_pred: {orig_pred}")
    if multi_spec:
        for i in range(prop_samples.shape[1]):
            predicted = model(prop_samples[:,i,:])
            predicted = predicted.argmax(dim=1).cpu().detach().numpy()
            idxs = np.where(predicted==y)[0]
            print(f'length of label {len(idxs)} preserved samples for spec {i}')             
    else:  
        predicted = model(prop_samples)
        predicted = predicted.argmax(dim=1).cpu().detach().numpy()
        idxs = np.where(predicted==y)[0]
        print('length of label preserved samples', len(idxs)) 
    pre_relu_layer, activation = get_act_vecs(prop_samples, model, arguments.Config["data"]["dataset"])
    act_file = os.path.join(arguments.Config['preimage']["sample_dir"], 'act_vec_{}_{}.pkl'.format(arguments.Config["data"]["dataset"], atk_tp))
    with open(act_file, 'wb') as f:
        pickle.dump(activation, f)
    mask_sample = []
    score_all = []
    unstable_indices = []
    for i, layer in enumerate(pre_relu_layer):
        act_vec = activation[layer]
        samples_lb = torch.min(act_vec, dim=0).values
        samples_ub = torch.max(act_vec, dim=0).values
        # 1 is unstable neuron, 0 is stable neuron.
        mask_tmp = torch.logical_and(samples_lb<0, samples_ub>0).float()
        mask_sample.append(torch.unsqueeze(mask_tmp,0))
        score_tmp = []
        unstable_idx = torch.squeeze(mask_tmp.nonzero(),1)
        unstable_indices.append(unstable_idx)
        if unstable_idx.shape[0]:
            pos_sample_mask = act_vec[:, unstable_idx] >= 0
            # neg_sample_mask = act_vec[:, unstable_idx] < 0
            for j in range(unstable_idx.shape[0]):
                pos_sample_idx_neuron = pos_sample_mask[:,j].nonzero()
                # neg_sample_idx_neuron = neg_sample_mask[:,j].nonzero()
                pos_freq = pos_sample_idx_neuron.shape[0]
                neg_freq = sample_num - pos_freq
                # print(pos_sample_idx_neuron.shape)
                score_tmp.append(sample_num-abs(pos_freq-neg_freq))
            score_all.append(score_tmp)                 
        else:
            score_all.append(score_tmp)                       
    return mask_sample, score_all, unstable_indices  
def restore_scores(score_all, unstable_indices, mask_sample):
    score_restore = []
    for i, unstable_idx in enumerate(unstable_indices):
        act_vec_shape = mask_sample[i].shape
        score_tmp = -torch.ones(act_vec_shape)
        score_tmp[:, unstable_idx] = torch.tensor(score_all[i],dtype=torch.float)
        # score_tmp = torch.unsqueeze(score_tmp, 0)
        score_restore.append(score_tmp)
    return score_restore 
def relu_bab_parallel(net, domain, x,y, use_neuron_set_strategy=False, refined_lower_bounds=None,
                      refined_upper_bounds=None, activation_opt_params=None,
                      reference_slopes=None, reference_lA=None, attack_images=None,
                      timeout=None, refined_betas=None, rhs=0):
    # the crown_lower/upper_bounds are present for initializing the unstable indx when constructing bounded module
    # it is ok to not pass them here, but then we need to go through a CROWN process again which is slightly slower
    start = time.time()
    # All supported arguments.
    global Visited, Flag_first_split, all_node_split 
    global total_pickout_time, total_decision_time, total_solve_time, total_add_time

    total_pickout_time = total_decision_time = total_solve_time = total_add_time = 0.0
    # NOTE add arguments required for preimage generation
    cov_thre = arguments.Config["preimage"]["threshold"]
    branch_budget = arguments.Config['preimage']['branch_budget']
    # result_dir = arguments.Config['preimage']['result_dir']
    bound_lower = arguments.Config["preimage"]["under_approx"]
    bound_upper = arguments.Config["preimage"]["over_approx"] 
    sample_dir = arguments.Config['preimage']["sample_dir"]
    sample_based_instability = arguments.Config['preimage']['instability']
    # model_tp = arguments.Config["model"]["name"] 
    # input_split_enabled = arguments.Config["bab"]["branching"]["input_split"]["enable"]
    # if input_split_enabled:
    #     opt_input_poly = True
    #     opt_relu_poly = False
    # else:
    #     opt_input_poly = False
    #     opt_relu_poly = True   

    timeout = timeout or arguments.Config["bab"]["timeout"]
    batch = arguments.Config["solver"]["batch_size"]
    opt_intermediate_beta = False
    use_bab_attack = arguments.Config["bab"]["attack"]["enabled"]
    max_dive_fix_ratio = arguments.Config["bab"]["attack"]["max_dive_fix_ratio"]
    min_local_free_ratio = arguments.Config["bab"]["attack"]["min_local_free_ratio"]
    cut_enabled = arguments.Config["bab"]["cut"]["enabled"]
    lp_cut_enabled = arguments.Config["bab"]["cut"]["lp_cut"]
    use_batched_domain = arguments.Config["bab"]["batched_domain_list"]
    

    if not isinstance(rhs, torch.Tensor):
        rhs = torch.tensor(rhs)
    decision_thresh = rhs

    # general (multi-bounds) output for one C matrix
    # any spec >= rhs, then this sample can be stopped; if all samples can be stopped, stop = True, o.w., False
    stop_criterion = stop_criterion_batch_any
    multi_spec_keep_func = lambda x: torch.all(x, dim=-1)

    Visited, Flag_first_split, global_ub = 0, True, np.inf
    betas = None
    Flag_covered = False

    y_label = y[0][0]
    # print('check data label', y_label)
    if arguments.Config["model"]["onnx_path"] is None:
        model_tp = arguments.Config["model"]["name"]    
        if model_tp == 'mnist_6_100':
            mask_sample_ori, score_all, unstable_indices = calc_mask_concrete_samples_from_unstable_idx(x, net, y_label)
        else:                   
            mask_sample_ori, score_all, unstable_indices = calc_mask_concrete_samples(x, net, y_label)
    else:
        dataset_tp = arguments.Config["data"]["dataset"]
        if "MNIST" in dataset_tp:
            sample_dir = os.path.join(sample_dir, 'mnist_6_100')
        mask_sample_file = os.path.join(sample_dir, f"{dataset_tp}/mask_sample_{dataset_tp}.pkl")
        score_all_file = os.path.join(sample_dir, f"{dataset_tp}/score_all_{dataset_tp}.pkl")
        unstable_indices_file = os.path.join(sample_dir, f"{dataset_tp}/unstable_indices_{dataset_tp}.pkl") 
        # if dataset_tp == "vcas":
            # upper_time_loss = arguments.Config["preimage"]["upper_time_loss"]
            # mask_sample_file = os.path.join(sample_dir, f"mask_sample_{dataset_tp}_{upper_time_loss}.pkl")
            # score_all_file = os.path.join(sample_dir, f"score_all_{dataset_tp}_{upper_time_loss}.pkl")
            # unstable_indices_file = os.path.join(sample_dir, f"unstable_indices_{dataset_tp}_{upper_time_loss}.pkl") 
        # else:
        #     mask_sample_file = os.path.join(sample_dir, f"mask_sample_{dataset_tp}.pkl")
        #     score_all_file = os.path.join(sample_dir, f"score_all_{dataset_tp}.pkl")
        #     unstable_indices_file = os.path.join(sample_dir, f"unstable_indices_{dataset_tp}.pkl")
        with open(mask_sample_file, 'rb') as f:
            mask_sample_ori = pickle.load(f)
        with open(score_all_file, 'rb') as f:
            score_all = pickle.load(f)
        with open(unstable_indices_file, 'rb') as f:
            unstable_indices = pickle.load(f)    
    score_restore_ori = restore_scores(score_all, unstable_indices, mask_sample_ori)
    tot_ambi_nodes_sample = 0
    for i, layer_mask in enumerate(mask_sample_ori):
        n_unstable = int(torch.sum(layer_mask).item())
        print(f'layer {i} size {layer_mask.shape[0]} unstable {n_unstable}')
        tot_ambi_nodes_sample += n_unstable
    print(f'-----------------\n# of unstable neurons (Sample): {tot_ambi_nodes_sample}\n-----------------\n')
      
    if arguments.Config["solver"]["alpha-crown"]["no_joint_opt"]:
        global_ub, global_lb, _, _, primals, updated_mask, lA, lower_bounds, upper_bounds, pre_relu_indices, slope, history, betas = net.build_the_model_with_refined_bounds(
            domain, x, None, None, stop_criterion_func=stop_criterion(decision_thresh), reference_slopes=None,
            cutter=net.cutter)
    elif refined_lower_bounds is None or refined_upper_bounds is None:
        assert arguments.Config["general"]["enable_incomplete_verification"] is False
        global_ub, global_lb, _, _, primals, updated_mask, lA, A, lower_bounds, upper_bounds, pre_relu_indices, slope, history, attack_image = net.build_the_model(
            domain, x, stop_criterion_func=stop_criterion(decision_thresh),opt_input_poly=False,opt_relu_poly=True)
    else:
        global_ub, global_lb, _, _, primals, updated_mask, lA, lower_bounds, upper_bounds, pre_relu_indices, slope, history, betas = net.build_the_model_with_refined_bounds(
            domain, x, refined_lower_bounds, refined_upper_bounds, activation_opt_params, reference_lA=reference_lA,
            stop_criterion_func=stop_criterion(decision_thresh), reference_slopes=reference_slopes,
            cutter=net.cutter, refined_betas=refined_betas)
        # release some storage to save memory
        if activation_opt_params is not None: del activation_opt_params
        torch.cuda.empty_cache()
    def update_lower_upper_bounds(unstable_indices, updated_mask, lower_bounds, upper_bounds):
        acti_vecs = load_act_vecs(arguments.Config["data"]["dataset"])
        for i, layer_mask in enumerate(updated_mask):
            unstable_idx_inter = torch.squeeze(layer_mask[0].nonzero(),1)
            # unstable_idx_inter = set(unstable_idx_inter)
            unstable_idx_inter = set(unstable_idx_inter.cpu().detach().numpy().tolist())
            # print("check len (interval)", len(unstable_idx_inter))
            unstable_idx_sample = unstable_indices[i].cpu().detach().numpy().tolist()
            unstable_idx_sample = set(unstable_idx_sample)
            # print("check len (sample)", len(unstable_idx_sample))
            unstable_set_diff = unstable_idx_inter-unstable_idx_sample
            # print("check len (diff)", len(unstable_set_diff))
            for idx in list(unstable_set_diff):
                updated_mask[i][0][idx] = 0
                if acti_vecs[i][0, idx] >= 0:
                    lower_bounds[i][0][idx] = 0.0
                else:
                    upper_bounds[i][0][idx] = 0.0
            
        return updated_mask, lower_bounds, upper_bounds
    tot_ambi_nodes = 0
    # only pick the first copy from possible multiple x
    updated_mask = [mask[0:1] for mask in updated_mask]
    # mask_sample = [mask[0:1] for mask in mask_sample]
    for i, layer_mask in enumerate(updated_mask):
        n_unstable = int(torch.sum(layer_mask).item())
        print(f'layer {i} size {layer_mask.shape[1:]} unstable {n_unstable}')
        tot_ambi_nodes += n_unstable

    print(f'-----------------\n# of unstable neurons (Interval): {tot_ambi_nodes}\n-----------------\n')
            
    if sample_based_instability:               
        updated_mask, lower_bounds, upper_bounds = update_lower_upper_bounds(unstable_indices, updated_mask, lower_bounds, upper_bounds)
    # NOTE check the first coarsest preimage without any splitting or optimization
    initial_covered, cov_quota, target_vol, preimage_dict = initial_check_preimage_approx(A, cov_thre, y_label)
    # NOTE second variable is intended for extra constraints
    if initial_covered:
        return (
            initial_covered,
            preimage_dict,
            Visited,
            time.time() - start,
            [cov_quota],
            1
        )
    if target_vol == 0:
        return (
            initial_covered,
            preimage_dict,
            Visited,
            time.time() - start,
            [1],
            1
        )
    # if arguments.Config["preimage"]["save_process"]:
    #     save_path = os.path.join(arguments.Config["preimage"]["result_dir"], 'run_example')
    #     save_file = os.path.join(save_path,'{}_spec_{}_init'.format(arguments.Config["data"]["dataset"], arguments.Config["preimage"]["runner_up"]))
    #     with open(save_file, 'wb') as f:
    #         pickle.dump(preimage_dict, f)
    if arguments.Config["solver"]["beta-crown"]["all_node_split_LP"]:
        timeout = arguments.Config["bab"]["timeout"]
        # mip_multi_proc = arguments.Config["solver"]["mip"]["parallel_solvers"]
        # mip_threads = arguments.Config["solver"]["mip"]["solver_threads"]
        # solver_pkg = arguments.Config["solver"]["intermediate_refinement"]["solver_pkg"]
        # adv_warmup = arguments.Config["solver"]["mip"]["adv_warmup"]
        net.build_solver_model(timeout, model_type="lp")

    if use_bab_attack:
        # Beam search based BaB enabled. We need to construct the MIP model.
        print('Building MIP for beam search...')
        _ = net.build_solver_model(
                    timeout=arguments.Config["bab"]["attack"]["mip_timeout"],
                    mip_multi_proc=arguments.Config["solver"]["mip"]["parallel_solvers"],
                    mip_threads=arguments.Config["solver"]["mip"]["solver_threads"],
                    model_type="mip")

    all_label_global_lb = global_lb
    all_label_global_lb = torch.min(all_label_global_lb - decision_thresh).item()
    all_label_global_ub = global_ub
    all_label_global_ub = torch.max(all_label_global_ub - decision_thresh).item()

    # if lp_test in ["LP", "MIP"]:
    #     return all_label_global_lb, all_label_global_ub, [[time.time()-start, global_lb]], 0, 'unknown'

    # if stop_criterion(decision_thresh)(global_lb).all():
    #     return all_label_global_lb, all_label_global_ub, [[time.time()-start, global_lb]], 0, 'safe'

    if not opt_intermediate_beta:
        # If we are not optimizing intermediate layer bounds, we do not need to save all the intermediate alpha.
        # We only keep the alpha for the last layer.
        if not arguments.Config['solver']['beta-crown'].get('enable_opt_interm_bounds', False):
            # new_slope shape: [dict[relu_layer_name, {final_layer: torch.tensor storing alpha}] for each sample in batch]
            new_slope = {}
            kept_layer_names = [net.net.final_name]
            kept_layer_names.extend(filter(lambda x: len(x.strip()) > 0, arguments.Config["bab"]["optimized_intermediate_layers"].split(",")))
            print(f'Keeping slopes for these layers: {kept_layer_names}')
            for relu_layer, alphas in slope.items():
                new_slope[relu_layer] = {}
                for layer_name in kept_layer_names:
                    if layer_name in alphas:
                        new_slope[relu_layer][layer_name] = alphas[layer_name]
                    else:
                        print(f'Layer {relu_layer} missing slope for start node {layer_name}')
        else:
            new_slope = slope



    # net.tot_ambi_nodes = tot_ambi_nodes

    if use_batched_domain:
        assert not use_bab_attack, "Please disable batched_domain_list to run BaB-Attack."
        DomainClass = BatchedReLUDomainList
    else:
        DomainClass = SortedReLUDomainList

    # This is the first (initial) domain.
    num_initial_domains = net.c.shape[0]
    if bound_lower:
        domains = DomainClass([target_vol], [cov_quota], [preimage_dict['lA']], [preimage_dict['lbias']],
                            global_lb, global_ub, lower_bounds, upper_bounds, new_slope,
                            copy.deepcopy(history), [0] * num_initial_domains, net.c, # "[0] * num_initial_domains" corresponds to initial domain depth
                            decision_thresh,
                            betas, num_initial_domains,
                            interm_transfer=arguments.Config["bab"]["interm_transfer"])
    elif bound_upper:
        domains = DomainClass([target_vol], [cov_quota], [preimage_dict['uA']], [preimage_dict['ubias']],
                            global_lb, global_ub, lower_bounds, upper_bounds, new_slope,
                            copy.deepcopy(history), [0] * num_initial_domains, net.c, # "[0] * num_initial_domains" corresponds to initial domain depth
                            decision_thresh,
                            betas, num_initial_domains,
                            interm_transfer=arguments.Config["bab"]["interm_transfer"])
    if use_bab_attack:
        # BaB-attack code still uses a legacy sorted domain list.
        domains = domains.to_sortedList()

    if not arguments.Config["bab"]["interm_transfer"]:
        # tell the AutoLiRPA class not to transfer intermediate bounds to save time
        net.interm_transfer = arguments.Config["bab"]["interm_transfer"]

    # after domains are added, we replace global_lb, global_ub with the multile targets "real" global lb and ub to make them scalars
    global_lb, global_ub = all_label_global_lb, all_label_global_ub

        
    if cut_enabled:
        print('======================Cut verification begins======================')
        start_cut = time.time()
        # enable lp solver
        if lp_cut_enabled:
            glb = net.build_the_model_lp()
        if arguments.Config["bab"]["cut"]["cplex_cuts"]:
            time.sleep(arguments.Config["bab"]["cut"]["cplex_cuts_wait"])
        global_lb_from_cut, batch_ub_from_cut = cut_verification(domains, net, pre_relu_indices, fix_intermediate_layer_bounds=not opt_intermediate_beta)
        if global_lb_from_cut is None and batch_ub_from_cut is None:
            # no available cut present --- we don't refresh global_lb and global_ub
            pass
        else:
            global_lb, batch_ub = global_lb_from_cut, batch_ub_from_cut
        print('Cut bounds before BaB:', float(global_lb))
        if len(domains) >= 1 and getattr(net.cutter, 'opt', False):
            # beta will be reused from split_history
            assert len(domains) == 1
            assert isinstance(domains[0].split_history['general_betas'], torch.Tensor)
            net.cutter.refine_cuts(split_history=domains[0].split_history)
        print('Cut time:', time.time() - start_cut)
        print('======================Cut verification ends======================')

    if arguments.Config["bab"]["attack"]["enabled"]:
        # Max number of fixed neurons during diving.
        max_dive_fix = int(max_dive_fix_ratio * tot_ambi_nodes)
        min_local_free = int(min_local_free_ratio * tot_ambi_nodes)
        adv_pool = AdvExamplePool(net.net, updated_mask, C=net.c)
        adv_pool.add_adv_images(attack_images)
        print(f'best adv in pool: {adv_pool.adv_pool[0].obj}, worst {adv_pool.adv_pool[-1].obj}')
        adv_pool.print_pool_status()
        find_promising_domains.counter = 0
        # find_promising_domains.current_method = "bottom-up"
        find_promising_domains.current_method = "top-down"
        find_promising_domains.topdown_status = "normal"
        find_promising_domains.bottomup_status = "normal"
        beam_mip_attack.started = False
        global_ub = min(all_label_global_ub, adv_pool.adv_pool[0].obj)

    glb_record = [[time.time()-start, global_lb]]
    iter_cov_quota = [cov_quota]
    # run_condition = len(domains) > 0
    num_iter = 0 

    if bound_lower:
        while cov_quota < cov_thre:
            global_lb = None

            if Visited >= branch_budget:
                time_cost = time.time() - start
                preimage_dict_all = get_preimage_info(domains)
                subdomain_num = len(domains)
                del domains            
                return False, preimage_dict_all, Visited, time_cost, iter_cov_quota, subdomain_num
            if use_bab_attack:
                max_dive_fix_ratio = arguments.Config["bab"]["attack"]["max_dive_fix_ratio"]
                min_local_free_ratio = arguments.Config["bab"]["attack"]["min_local_free_ratio"]
                max_dive_fix = int(max_dive_fix_ratio * tot_ambi_nodes)
                min_local_free = int(min_local_free_ratio * tot_ambi_nodes)
                global_lb, batch_ub, domains = bab_attack(
                        domains, net, batch, pre_relu_indices, 0,
                        fix_intermediate_layer_bounds=True,
                        adv_pool=adv_pool,
                        max_dive_fix=max_dive_fix, min_local_free=min_local_free)

            # if global_lb is None:
            # cut is enabled
            if arguments.Config["bab"]["cut"]["enabled"] and arguments.Config["bab"]["cut"]["cplex_cuts"]:
                fetch_cut_from_cplex(net)
            # Do two batch of neuron set bounds per 10000 domains
            if len(domains) > 80000 and len(domains) % 10000 < batch * 2 and use_neuron_set_strategy:
                # neuron set  bounds cost more memory, we set a smaller batch here
                cov_quota, all_node_split = batch_verification(tot_ambi_nodes_sample, mask_sample_ori, score_restore_ori, domains, net, int(batch/2), pre_relu_indices, 0,
                                        fix_intermediate_layer_bounds=False, stop_func=stop_criterion,
                                        multi_spec_keep_func=multi_spec_keep_func, bound_lower=bound_lower, bound_upper=bound_upper)
            else:
                cov_quota, all_node_split = batch_verification(tot_ambi_nodes_sample, mask_sample_ori, score_restore_ori, domains, y_label, net, batch, pre_relu_indices, 0,
                                        fix_intermediate_layer_bounds=not opt_intermediate_beta,
                                        stop_func=stop_criterion, multi_spec_keep_func=multi_spec_keep_func, bound_lower=bound_lower, bound_upper=bound_upper)


            print('--- Iteration {}, Cov quota {} ---'.format(num_iter+1, cov_quota))
            iter_cov_quota.append(cov_quota)
            # if num_iter == 163:
            #     print("start to check")
            #     print("check details")
            if arguments.Config["preimage"]["save_process"]:
                preimage_dict_all = get_preimage_info(domains)
                # history_list = []
                # for idx, dm in enumerate(domains):
                #     history_list.append(dm.history)
                # split_plane_list = get_extra_const(net, history_list)
                save_path = os.path.join(arguments.Config["preimage"]["result_dir"], 'run_example')
                save_file = os.path.join(save_path,'{}_spec_{}_iter_{}'.format(arguments.Config["data"]["dataset"], arguments.Config["preimage"]["runner_up"], num_iter))
                with open(save_file, 'wb') as f:
                    pickle.dump(preimage_dict_all, f) 
                # split_plane_file = os.path.join(save_path, '{}_split_iter_{}'.format(arguments.Config["data"]["dataset"], num_iter))
                # with open(split_plane_file, 'wb') as f:
                #     pickle.dump(split_plane_list, f)                    


            num_iter += 1
            if all_node_split:
                time_cost = time.time() - start
                preimage_dict_all = get_preimage_info(domains)
                subdomain_num = len(domains)
                del domains
                return True, preimage_dict_all, Visited, time_cost, iter_cov_quota, subdomain_num
        # if isinstance(global_lb, torch.Tensor):
        #     global_lb = global_lb.max().item()
        # if isinstance(global_ub, torch.Tensor):
        #     global_ub = global_ub.min().item()
    elif bound_upper:
        while cov_quota > cov_thre:
            global_lb = None

            if Visited >= branch_budget:
                time_cost = time.time() - start
                preimage_dict_all = get_preimage_info(domains)
                subdomain_num = len(domains)
                del domains            
                return False, preimage_dict_all, Visited, time_cost, iter_cov_quota, subdomain_num
            if use_bab_attack:
                max_dive_fix_ratio = arguments.Config["bab"]["attack"]["max_dive_fix_ratio"]
                min_local_free_ratio = arguments.Config["bab"]["attack"]["min_local_free_ratio"]
                max_dive_fix = int(max_dive_fix_ratio * tot_ambi_nodes)
                min_local_free = int(min_local_free_ratio * tot_ambi_nodes)
                global_lb, batch_ub, domains = bab_attack(
                        domains, net, batch, pre_relu_indices, 0,
                        fix_intermediate_layer_bounds=True,
                        adv_pool=adv_pool,
                        max_dive_fix=max_dive_fix, min_local_free=min_local_free)

            # if global_lb is None:
            # cut is enabled
            if arguments.Config["bab"]["cut"]["enabled"] and arguments.Config["bab"]["cut"]["cplex_cuts"]:
                fetch_cut_from_cplex(net)
            # Do two batch of neuron set bounds per 10000 domains
            if len(domains) > 80000 and len(domains) % 10000 < batch * 2 and use_neuron_set_strategy:
                # neuron set  bounds cost more memory, we set a smaller batch here
                cov_quota, all_node_split = batch_verification(tot_ambi_nodes_sample, mask_sample_ori, score_restore_ori, domains, net, int(batch/2), pre_relu_indices, 0,
                                        fix_intermediate_layer_bounds=False, stop_func=stop_criterion,
                                        multi_spec_keep_func=multi_spec_keep_func, bound_lower=bound_lower, bound_upper=bound_upper)
            else:
                cov_quota, all_node_split = batch_verification(tot_ambi_nodes_sample, mask_sample_ori, score_restore_ori, domains, y_label, net, batch, pre_relu_indices, 0,
                                        fix_intermediate_layer_bounds=not opt_intermediate_beta,
                                        stop_func=stop_criterion, multi_spec_keep_func=multi_spec_keep_func, bound_lower=bound_lower, bound_upper=bound_upper)


            print('--- Iteration {}, Cov quota {} ---'.format(num_iter+1, cov_quota))
            iter_cov_quota.append(cov_quota)
            # if num_iter == 163:
            #     print("start to check")
            #     print("check details")
            if arguments.Config["preimage"]["save_process"]:
                preimage_dict_all = get_preimage_info(domains)
                # history_list = []
                # for idx, dm in enumerate(domains):
                #     history_list.append(dm.history)
                # split_plane_list = get_extra_const(net, history_list)
                save_path = os.path.join(arguments.Config["preimage"]["result_dir"], 'run_example')
                if bound_lower:
                    save_file = os.path.join(save_path,'{}_spec_{}_iter_{}'.format(arguments.Config["data"]["dataset"], arguments.Config["preimage"]["runner_up"], num_iter))
                if bound_upper:
                    save_file = os.path.join(save_path,'{}_spec_{}_iter_{}_relu_over_dual_False_0_4'.format(arguments.Config["data"]["dataset"], arguments.Config["preimage"]["runner_up"], num_iter))
                with open(save_file, 'wb') as f:
                    pickle.dump(preimage_dict_all, f) 
                # split_plane_file = os.path.join(save_path, '{}_split_relu_over_1_6'.format(arguments.Config["data"]["dataset"]))
                # with open(split_plane_file, 'wb') as f:
                #     pickle.dump(split_plane_list, f)                    


            num_iter += 1    
            if all_node_split:
                time_cost = time.time() - start
                preimage_dict_all = get_preimage_info(domains)
                subdomain_num = len(domains)
                del domains
                return True, preimage_dict_all, Visited, time_cost, iter_cov_quota, subdomain_num
    time_cost = time.time() - start
    preimage_dict_all = get_preimage_info(domains)
    subdomain_num = len(domains)
    del domains
    return True, preimage_dict_all, Visited, time_cost, iter_cov_quota, subdomain_num

def get_preimage_info(domains):
    preimage_dict_all = []
    for idx, dom in enumerate(domains):
        preimage_dict_all.append((dom.preimg_A, dom.preimg_b))
    return preimage_dict_all
            
    
    
def get_extra_const(net, history_list):
    # NOTE obtain the intermediate neuron constraints
    split_relu_indices = []
    assert len(history_list) == 2
    for i, layer_split in enumerate(history_list[0]):
        if len(layer_split[0]) != 0:
            split_relu_indices.append(i)
    A_dict_relus = net.get_intermediate_constraints(split_relu_indices)
    from test_polyhedron_util import post_process_A_dict_relu, calc_extra_A_b
    A_b_dict_relus = post_process_A_dict_relu(A_dict_relus)
    left_relu_A, left_relu_b = calc_extra_A_b(A_b_dict_relus, history_list[0], split_relu_indices, indicator=0)
    right_relu_A, right_relu_b = calc_extra_A_b(A_b_dict_relus, history_list[1], split_relu_indices, indicator=1)
    return [left_relu_A, left_relu_b, right_relu_A, right_relu_b]
        #     # NOTE obtain the intermediate neuron constraints
        # split_relu_indices = []
        # for i, layer_split in enumerate(left_history):
        #     if len(layer_split[0]) != 0:
        #         split_relu_indices.append(i)
        # A_dict_relus = net.get_intermediate_constraints(split_relu_indices)
        # if len(domains) > max_domains:
        #     print("Maximum number of visited domains has reached.")
        #     del domains
        #     clean_net_mps_process(net)
        #     return global_lb, global_ub, glb_record, Visited, 'unknown'

        # if get_upper_bound or arguments.Config["bab"]["attack"]["enabled"]:
        #     if global_ub < decision_thresh:
        #         print("Attack success during branch and bound.")
        #         # Terminate MIP if it has been started.
        #         if arguments.Config["bab"]["attack"]["enabled"] and beam_mip_attack.started:
        #             print('Terminating MIP processes...')
        #             net.pool_termination_flag.value = 1
        #         del domains
        #         clean_net_mps_process(net)
        #         return global_lb, global_ub, glb_record, Visited, 'unsafe'

        # if record:
        #     glb_record.append([time.time() - start, global_lb])
        # print(f'Cumulative time: {time.time() - start}\n')

    

    # clean_net_mps_process(net)

    # if arguments.Config["bab"]["attack"]["enabled"]:
    #     # No domains left and no ub < 0 found.
    #     return global_lb, global_ub, glb_record, Visited, 'unknown'
    # else:
    #     # No domains left and not timed out.
    #     return global_lb, global_ub, glb_record, Visited, 'safe'
