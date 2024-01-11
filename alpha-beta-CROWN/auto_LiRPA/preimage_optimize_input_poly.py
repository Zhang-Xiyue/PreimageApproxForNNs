import torch
import torch.nn as nn

from .patches import Patches
from .utils import *
from .bound_ops import *


def concretize_poly_vol(lb, ub, node, root, batch_size, output_dim, bound_lower=True, bound_upper=True, average_A=False):

    for i in range(len(root)):
        if root[i].lA is None and root[i].uA is None: continue
        if average_A and isinstance(root[i], BoundParams):
            lA = root[i].lA.mean(node.batch_dim + 1, keepdim=True).expand(root[i].lA.shape) if bound_lower else None
            uA = root[i].uA.mean(node.batch_dim + 1, keepdim=True).expand(root[i].uA.shape) if bound_upper else None
        else:
            # print('lb is actually lbias')
            lA, uA = root[i].lA, root[i].uA
        if not isinstance(root[i].lA, eyeC) and not isinstance(root[i].lA, Patches):
            lA = root[i].lA.reshape(output_dim, batch_size, -1).transpose(0, 1) if bound_lower else None
        if not isinstance(root[i].uA, eyeC) and not isinstance(root[i].uA, Patches):
            uA = root[i].uA.reshape(output_dim, batch_size, -1).transpose(0, 1) if bound_upper else None
        if hasattr(root[i], 'perturbation') and root[i].perturbation is not None:
            if isinstance(root[i], BoundParams):
                # add batch_size dim for weights node
                lb = lb + root[i].perturbation.concretize(
                    root[i].center.unsqueeze(0), lA,
                    sign=-1, aux=root[i].aux) if bound_lower else None
                ub = ub + root[i].perturbation.concretize(
                    root[i].center.unsqueeze(0), uA,
                    sign=+1, aux=root[i].aux) if bound_upper else None
            else:
                # use this branch for polytope-quality guided optimization
                lb = root[i].perturbation.concretize_poly_vol(
                    root[i].center, lA, lb, sign=-1, aux=root[i].aux) if bound_lower else None
                ub = root[i].perturbation.concretize_poly_vol(
                    root[i].center, uA, ub, sign=+1, aux=root[i].aux) if bound_upper else None
                if lb is not None:
                    lb = torch.sigmoid(lb)
                    lb = torch.mean(lb)
                if ub is not None:
                    ub = torch.sigmoid(ub)
                    ub = torch.mean(ub)
        else:
            fv = root[i].forward_value
            if type(root[i]) == BoundInput:
                # Input node with a batch dimension
                batch_size_ = batch_size
            else:
                # Parameter node without a batch dimension
                batch_size_ = 1

            if bound_lower:
                if isinstance(lA, eyeC):
                    lb = lb + fv.view(batch_size_, -1)
                elif isinstance(lA, Patches):
                    lb = lb + lA.matmul(fv, input_shape=root[0].center.shape)
                elif type(root[i]) == BoundInput:
                    lb = lb + lA.matmul(fv.view(batch_size_, -1, 1)).squeeze(-1)
                else:
                    lb = lb + lA.matmul(fv.view(-1, 1)).squeeze(-1)
            else:
                lb = None

            if bound_upper:
                if isinstance(uA, eyeC):
                    ub = ub + fv.view(batch_size_, -1)
                elif isinstance(uA, Patches):
                    ub = ub + uA.matmul(fv, input_shape=root[0].center.shape)
                elif type(root[i]) == BoundInput:
                    ub = ub + uA.matmul(fv.view(batch_size_, -1, 1)).squeeze(-1)
                else:
                    ub = ub + uA.matmul(fv.view(-1, 1)).squeeze(-1)
            else:
                ub = None

    return lb, ub

def concretize_poly_vol_LSE(lb, ub, node, root, batch_size, output_dim, bound_lower=True, bound_upper=True, average_A=False):

    for i in range(len(root)):
        if root[i].lA is None and root[i].uA is None: continue
        if average_A and isinstance(root[i], BoundParams):
            lA = root[i].lA.mean(node.batch_dim + 1, keepdim=True).expand(root[i].lA.shape) if bound_lower else None
            uA = root[i].uA.mean(node.batch_dim + 1, keepdim=True).expand(root[i].uA.shape) if bound_upper else None
        else:
            # print('lb is actually lbias')
            lA, uA = root[i].lA, root[i].uA
        if not isinstance(root[i].lA, eyeC) and not isinstance(root[i].lA, Patches):
            lA = root[i].lA.reshape(output_dim, batch_size, -1).transpose(0, 1) if bound_lower else None
        if not isinstance(root[i].uA, eyeC) and not isinstance(root[i].uA, Patches):
            uA = root[i].uA.reshape(output_dim, batch_size, -1).transpose(0, 1) if bound_upper else None
        if hasattr(root[i], 'perturbation') and root[i].perturbation is not None:
            if isinstance(root[i], BoundParams):
                # add batch_size dim for weights node
                lb = lb + root[i].perturbation.concretize(
                    root[i].center.unsqueeze(0), lA,
                    sign=-1, aux=root[i].aux) if bound_lower else None
                ub = ub + root[i].perturbation.concretize(
                    root[i].center.unsqueeze(0), uA,
                    sign=+1, aux=root[i].aux) if bound_upper else None
            else:
                # use this branch for LSE-based optimization
                lb = root[i].perturbation.concretize_poly_vol_LSE(
                    root[i].center, lA, lb, sign=-1, aux=root[i].aux) if bound_lower else None
                ub = root[i].perturbation.concretize_poly_vol_LSE(
                    root[i].center, uA, ub, sign=+1, aux=root[i].aux) if bound_upper else None

        else:
            fv = root[i].forward_value
            if type(root[i]) == BoundInput:
                # Input node with a batch dimension
                batch_size_ = batch_size
            else:
                # Parameter node without a batch dimension
                batch_size_ = 1

            if bound_lower:
                if isinstance(lA, eyeC):
                    lb = lb + fv.view(batch_size_, -1)
                elif isinstance(lA, Patches):
                    lb = lb + lA.matmul(fv, input_shape=root[0].center.shape)
                elif type(root[i]) == BoundInput:
                    lb = lb + lA.matmul(fv.view(batch_size_, -1, 1)).squeeze(-1)
                else:
                    lb = lb + lA.matmul(fv.view(-1, 1)).squeeze(-1)
            else:
                lb = None

            if bound_upper:
                if isinstance(uA, eyeC):
                    ub = ub + fv.view(batch_size_, -1)
                elif isinstance(uA, Patches):
                    ub = ub + uA.matmul(fv, input_shape=root[0].center.shape)
                elif type(root[i]) == BoundInput:
                    ub = ub + uA.matmul(fv.view(batch_size_, -1, 1)).squeeze(-1)
                else:
                    ub = ub + uA.matmul(fv.view(-1, 1)).squeeze(-1)
            else:
                ub = None

    return lb, ub