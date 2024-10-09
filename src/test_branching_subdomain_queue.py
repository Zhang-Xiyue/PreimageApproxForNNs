#########################################################################
##                                                                     ##
##     This program is licenced under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
import torch
import numpy as np
import math
import random
from sortedcontainers import SortedList
from tensor_storage import TensorStorage
# import preimage_arguments
import arguments


class InputDomainList:
    """Abstract class that maintains a list of domains for input split."""

    def __init__(self):
        pass

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        # get lb, dm_l, dm_u, cs, threshold for idx; for convenience, slope and split_idx
        # are not returned for now
        raise NotImplementedError

    def add(
        self,
        lb,
        dm_l,
        dm_u,
        slope,
        cs,
        threshold=0,
        split_idx=None,
        remaining_index=None,
    ):
        raise NotImplementedError

    def pick_out_batch(self, batch, device="cuda"):
        raise NotImplementedError

    def get_topk_indices(self, k=1, largest=False):
        # get the topk indices, by default worst k
        raise NotImplementedError

class UnsortedInputDomainList(InputDomainList):
    """UnSorted domain list for input split."""

    def __init__(self):
        super(UnsortedInputDomainList, self).__init__()
        # self.domains = SortedList()
        self.domains = list()
        min_batch_size = (
            arguments.Config["solver"]["min_batch_size_ratio"]
            * arguments.Config["solver"]["batch_size"]
        )
        self.max_depth = max(int(math.log(max(min_batch_size, 1)) // math.log(2)), 1)

    def __len__(self):
        return len(self.domains)

    def __getitem__(self, idx):
        domain = self.domains[idx]
        return domain.cov_quota, domain.lA, domain.lbias, domain.dm_l, domain.dm_u, domain.c, domain.threshold, domain.total_sub_vol

    def add_multi(
        self,
        cov_quota_list,
        target_vol_list,
        lA_list,
        lbias_list,
        lb,
        dm_l,
        dm_u,
        slope,
        cs,
        threshold=0,
        split_idx=None
    ):
        # check shape consistency and correctness
        batch = len(lb)
        prop_num = len(cov_quota_list)
        spec_num = int(batch / prop_num)
        assert len(dm_l) == len(dm_u) == len(cs) == len(threshold) == batch
        # assert len(lA_list[0]) == len(lbias_list[0])
        # assert len(split_idx) == batch
        # assert split_idx.shape[1] == lA_list[0].shape[1]
        threshold = threshold.to(device=lb.device)
        dm_l = dm_l.to(device=lb.device)
        dm_u = dm_u.to(device=lb.device)
        cs = cs.to(device=lb.device)
        remaining_index = torch.arange(prop_num)
        # if remaining_index is None:
        #     # FIXME: this should check the criterion function.
        #     remaining_index = torch.where((lb <= threshold).all(1))[0]
        for i in remaining_index:
            if slope is not None and type(slope) != list:
                slope_dict = {}
                for key0 in slope.keys():
                    slope_dict[key0] = {}
                    for key1 in slope[key0].keys():
                        slope_dict[key0][key1] = slope[key0][key1][:, :, i*spec_num : (i + 1)*spec_num]
            dom = InputDomain(
                cov_quota=cov_quota_list[i],
                target_vol=target_vol_list[i],
                lA=lA_list[i],
                lbias=lbias_list[i], # these three comes from a batch of spec, therefore no need to *spec_num
                lb=lb[i*spec_num: (i + 1)*spec_num],
                slope=slope_dict if slope is not None and type(slope) != list else None,
                dm_l=dm_l[i*spec_num : (i + 1)*spec_num],
                dm_u=dm_u[i*spec_num  : (i + 1)*spec_num],
                c=cs[i*spec_num : (i + 1)*spec_num],
                threshold=threshold[i*spec_num : (i + 1)*spec_num],
                split_idx=(split_idx[i*spec_num : (i + 1)*spec_num] if split_idx is not None else None),
            )
            self.domains.append(dom)


    def pick_out_batch(self, batch, device="cuda"):
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # make sure GPU to CPU transfer is finished

        batch = min(len(self.domains), batch)
        # Lower and upper bounds of inputs.
        dm_l_all, dm_u_all = [], []
        # Specification matrices, and decision thresholds.
        c_all, thresholds_all = [], []
        slopes_all, split_idx = [], []
        assert len(self.domains) > 0, "The given domains list is empty."

        for i in range(batch):
            # Pop out domains from the list one by one (SLOW).
            # selected_candidate_domain = self.domains.pop()
            selected_candidate_domain = self.domains.pop(random.randrange(len(self.domains)))
            # We transfer only some of the tensors directly to GPU. Other tensors will be transfered in batch later.
            selected_candidate_domain.to_device(device, partial=True)
              
            dm_l_all.append(selected_candidate_domain.dm_l)
            dm_u_all.append(selected_candidate_domain.dm_u)
            c_all.append(selected_candidate_domain.c)
            thresholds_all.append(selected_candidate_domain.threshold)
            # slopes_all.append(selected_candidate_domain.slope)
            # NOTE deal with slope dimension issue
            slope_val = selected_candidate_domain.slope
            slope_batch = dm_l_all[0].shape[-2]
            for i in range(slope_batch):
                item = {}
                for key0 in slope_val.keys():
                    item[key0] = {}
                    for key1 in slope_val[key0].keys():
                        item[key0][key1] = slope_val[key0][key1][:, :, i : i + 1].to(
                            device=device, non_blocking=True
                        )                        
                slopes_all.append(item)              
            if (
                split_idx is not None
                and selected_candidate_domain.split_idx is not None
            ):
                split_idx.append(selected_candidate_domain.split_idx)
            else:
                split_idx = None

        thresholds = torch.stack(thresholds_all).to(device=device, non_blocking=True)
        split_idx = torch.cat(split_idx) if split_idx is not None else None

        # aggregate C to shape (batch, 1, num_outputs)
        cs = torch.cat(c_all).contiguous().to(device=device, non_blocking=True)

        # Input split domains.
        return (
            slopes_all,
            torch.cat(dm_l_all).to(device=device, non_blocking=True),
            torch.cat(dm_u_all).to(device=device, non_blocking=True),
            cs,
            thresholds,
            split_idx,
        )

    def get_topk_indices(self, k=1, largest=False):
        assert k <= self.__len__(), print("Asked indices more than domain length.")
        return -torch.arange(k) - 1 if largest else torch.arange(k)
class SortedInputDomainList(InputDomainList):
    """Sorted domain list for input split."""

    def __init__(self):
        super(SortedInputDomainList, self).__init__()
        self.domains = SortedList()
        min_batch_size = (
            arguments.Config["solver"]["min_batch_size_ratio"]
            * arguments.Config["solver"]["batch_size"]
        )
        self.max_depth = max(int(math.log(max(min_batch_size, 1)) // math.log(2)), 1)

    def __len__(self):
        return len(self.domains)

    def __getitem__(self, idx):
        domain = self.domains[idx]
        return domain.cov_quota, domain.lA, domain.lbias, domain.uA, domain.ubias, domain.dm_l, domain.dm_u, domain.c, domain.threshold, domain.total_sub_vol

    def add_multi(
        self,
        cov_quota_list,
        target_vol_list,
        dm_l,
        dm_u,
        slope,
        cs,
        threshold=0,
        split_idx=None,
        lA_list=None,
        lbias_list=None,
        lb=None,
        uA_list=None,
        ubias_list=None,
        ub=None
    ):
        # check shape consistency and correctness
        batch = len(lb) if lb is not None else len(ub)
        prop_num = len(cov_quota_list)
        spec_num = int(batch / prop_num)
        assert len(dm_l) == len(dm_u) == len(cs) == len(threshold) == batch
        # assert len(lA_list[0]) == len(lbias_list[0])
        # assert len(split_idx) == batch
        # assert split_idx.shape[1] == lA_list[0].shape[1]
        device = lb.device if lb is not None else ub.device
        threshold = threshold.to(device=device)
        dm_l = dm_l.to(device=device)
        dm_u = dm_u.to(device=device)
        cs = cs.to(device=device)
        remaining_index = torch.arange(prop_num)
        # if remaining_index is None:
        #     # FIXME: this should check the criterion function.
        #     remaining_index = torch.where((lb <= threshold).all(1))[0]
        if arguments.Config["preimage"]["under_approx"]:
            for i in remaining_index:
                if slope is not None and type(slope) != list:
                    slope_dict = {}
                    for key0 in slope.keys():
                        slope_dict[key0] = {}
                        for key1 in slope[key0].keys():
                            slope_dict[key0][key1] = slope[key0][key1][:, :, i*spec_num : (i + 1)*spec_num]
                dom = InputDomain(
                    cov_quota=cov_quota_list[i],
                    target_vol=target_vol_list[i],
                    lA=lA_list[i],
                    lbias=lbias_list[i], # these three comes from a batch of spec, therefore no need to *spec_num
                    lb=lb[i*spec_num: (i + 1)*spec_num],
                    slope=slope_dict if slope is not None and type(slope) != list else None,
                    dm_l=dm_l[i*spec_num : (i + 1)*spec_num],
                    dm_u=dm_u[i*spec_num  : (i + 1)*spec_num],
                    c=cs[i*spec_num : (i + 1)*spec_num],
                    threshold=threshold[i*spec_num : (i + 1)*spec_num],
                    split_idx=(split_idx[i*spec_num : (i + 1)*spec_num] if split_idx is not None else None),
                )
                self.domains.add(dom)
        elif arguments.Config["preimage"]["over_approx"]:
            for i in remaining_index:
                if slope is not None and type(slope) != list:
                    slope_dict = {}
                    for key0 in slope.keys():
                        slope_dict[key0] = {}
                        for key1 in slope[key0].keys():
                            slope_dict[key0][key1] = slope[key0][key1][:, :, i*spec_num : (i + 1)*spec_num]
                dom = InputDomain(
                    cov_quota=cov_quota_list[i],
                    target_vol=target_vol_list[i],
                    uA=uA_list[i],
                    ubias=ubias_list[i], # these three comes from a batch of spec, therefore no need to *spec_num
                    ub=ub[i*spec_num: (i + 1)*spec_num],
                    slope=slope_dict if slope is not None and type(slope) != list else None,
                    dm_l=dm_l[i*spec_num : (i + 1)*spec_num],
                    dm_u=dm_u[i*spec_num  : (i + 1)*spec_num],
                    c=cs[i*spec_num : (i + 1)*spec_num],
                    threshold=threshold[i*spec_num : (i + 1)*spec_num],
                    split_idx=(split_idx[i*spec_num : (i + 1)*spec_num] if split_idx is not None else None),
                )
                self.domains.add(dom)
                
    def pick_out_batch(self, batch, device="cuda"):
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # make sure GPU to CPU transfer is finished

        batch = min(len(self.domains), batch)
        # Lower and upper bounds of inputs.
        dm_l_all, dm_u_all = [], []
        # Specification matrices, and decision thresholds.
        c_all, thresholds_all = [], []
        slopes_all, split_idx = [], []
        assert len(self.domains) > 0, "The given domains list is empty."

        for i in range(batch):
            # Pop out domains from the list one by one (SLOW).
            selected_candidate_domain = self.domains.pop()
            # We transfer only some of the tensors directly to GPU. Other tensors will be transfered in batch later.
            selected_candidate_domain.to_device(device, partial=True)
              
            dm_l_all.append(selected_candidate_domain.dm_l)
            dm_u_all.append(selected_candidate_domain.dm_u)
            c_all.append(selected_candidate_domain.c)
            thresholds_all.append(selected_candidate_domain.threshold)
            # slopes_all.append(selected_candidate_domain.slope)
            # NOTE deal with slope dimension issue
            slope_val = selected_candidate_domain.slope
            if slope_val is not None:
                slope_batch = dm_l_all[0].shape[-2]
                for i in range(slope_batch):
                    item = {}
                    for key0 in slope_val.keys():
                        item[key0] = {}
                        for key1 in slope_val[key0].keys():
                            item[key0][key1] = slope_val[key0][key1][:, :, i : i + 1].to(
                                device=device, non_blocking=True
                            )                        
                    slopes_all.append(item) 
            else:
                slopes_all = None             
            if (
                split_idx is not None
                and selected_candidate_domain.split_idx is not None
            ):
                split_idx.append(selected_candidate_domain.split_idx)
            else:
                split_idx = None

        thresholds = torch.stack(thresholds_all).to(device=device, non_blocking=True)
        split_idx = torch.cat(split_idx) if split_idx is not None else None

        # aggregate C to shape (batch, 1, num_outputs)
        cs = torch.cat(c_all).contiguous().to(device=device, non_blocking=True)

        # Input split domains.
        return (
            slopes_all,
            torch.cat(dm_l_all).to(device=device, non_blocking=True),
            torch.cat(dm_u_all).to(device=device, non_blocking=True),
            cs,
            thresholds,
            split_idx,
        )

    def get_topk_indices(self, k=1, largest=False):
        assert k <= self.__len__(), print("Asked indices more than domain length.")
        return -torch.arange(k) - 1 if largest else torch.arange(k)



class InputDomain:
    """Singleton domain for input split, used by sorted domain list.
    xy: use cov_quota and dm_l, dm_u to calculate the uncovered volume"""

    def __init__(
        self,
        cov_quota,
        target_vol,
        lA=None,
        lbias=None,
        uA=None,
        ubias=None,
        lb=-float("inf"),
        ub=float("inf"),
        lb_all=None,
        up_all=None,
        slope=None,
        dm_l=None,
        dm_u=None,
        selected_dims=None,
        device="cpu",
        depth=None,
        c=None,
        threshold=0,
        split_idx=None,
    ):
        self.cov_quota = cov_quota
        self.lA = lA
        self.lbias = lbias
        self.uA = uA
        self.ubias = ubias
        self.lower_bound = lb
        self.upper_bound = ub
        self.dm_l = dm_l
        self.dm_u = dm_u
        self.lower_all = lb_all
        self.upper_all = up_all
        self.slope = slope
        self.selected_dims = selected_dims
        self.device = device
        self.split = False
        self.valid = True
        self.depth = depth
        self.beta = None
        self.intermediate_betas = None
        self.c = c
        self.threshold = threshold
        self.split_idx = split_idx
        self.total_sub_vol = target_vol
        if arguments.Config["preimage"]["under_approx"]:
            self.uncov_vol = self.total_sub_vol * (1 - self.cov_quota)
        elif arguments.Config["preimage"]["over_approx"]:
            self.uncov_vol = self.total_sub_vol * (self.cov_quota - 1)
    def __lt__(self, other):
        return self.uncov_vol < other.uncov_vol 

    def __le__(self, other):
        return self.uncov_vol <= other.uncov_vol

    def __eq__(self, other):
        return self.uncov_vol == other.uncov_vol

    def __repr__(self):
        rep = (
            f"CovQuota: {self.cov_quota}\n"
            if self.cov_quota is not None
            else ""
        )
        rep += (
            f"Input subdomain lower bound: {self.dm_l}\n"
        )
        rep += (
            f"Input subdomain upper bound: {self.dm_u}\n"
        )
        return rep
    
    # def calc_total_sub_vol(self):
    #     total_sub_vol = 1
    #     in_dim = self.dm_l.shape[-1]
    #     dm_shape = self.dm_l.shape
    #     print("check domain lower/upper shape", dm_shape)
    #     assert len(dm_shape) == 2 # or len(dm_shape) == 3
    #     for i in range(in_dim):
    #         total_sub_vol = total_sub_vol * (self.dm_u[0][i] - self.dm_l[0][i])

    #     return total_sub_vol
    

        
    def verify_criterion(self):
        return (self.lower_bound > self.threshold).any()

    def attack_criterion(self):
        return (self.upper_bound <= self.threshold).all()

    def to_cpu(self):
        if self.device == "cuda":
            return self
        # transfer the content of this domain to cpu memory (try to reduce memory consumption)
        self.dm_l = self.dm_l.to(device="cpu", non_blocking=True)
        self.dm_u = self.dm_u.to(device="cpu", non_blocking=True)
        self.lower_bound = self.lower_bound.to(device="cpu", non_blocking=True)
        self.upper_bound = self.upper_bound.to(device="cpu", non_blocking=True)
        if self.selected_dims is not None:
            self.selected_dims = self.selected_dims.to(device="cpu", non_blocking=True)

        if self.c is not None:
            self.c = self.c.to(device="cpu", non_blocking=True)

        self.threshold = self.threshold.to(device="cpu", non_blocking=True)

        if self.slope is not None:
            for layer in self.slope:
                for intermediate_layer in self.slope[layer]:
                    self.slope[layer][intermediate_layer] = (
                        self.slope[layer][intermediate_layer]
                        .half()
                        .to(device="cpu", non_blocking=True)
                    )

        return self

    def to_device(self, device, partial=False):
        if self.device == "cuda":
            return self
        self.dm_l = self.dm_l.to(device, non_blocking=True)
        self.dm_u = self.dm_u.to(device, non_blocking=True)
        # self.lower_bound = (
        #     self.lower_bound.to(device, non_blocking=True)
        #     if self.lower_bound is not None
        #     else None
        # )

        if self.c is not None:
            self.c = self.c.to(device, non_blocking=True)

        self.threshold = self.threshold.to(device, non_blocking=True)

        if self.slope is not None:
            for layer in self.slope:
                for intermediate_layer in self.slope[layer]:
                    self.slope[layer][intermediate_layer] = (
                        self.slope[layer][intermediate_layer]
                        .to(device, non_blocking=True)
                        .to(torch.get_default_dtype())
                    )

        return self
