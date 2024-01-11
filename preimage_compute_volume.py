#########################################################################
##   This file is for preimage volume computation                      ##
##                                                                     ##
## Copyright (C) 2022-2023                                             ##
##     This program is licenced under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################

import pickle
import os
from ppl import Variable, Constraint_System, C_Polyhedron, NNC_Polyhedron
from scipy.spatial import ConvexHull
import numpy as np

# Set a global multiplier
multiplier = 1000000000


def load_preimage(dataset, label, dm_num):
    result_dir = "./results/quant_analysis"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    save_file = os.path.join(result_dir,'{}_spec_{}_dm_{}'.format(dataset, label, dm_num))
    if not os.path.exists(save_file):
        raise FileNotFoundError('polytope constraints not found')
    with open(save_file, 'rb') as f:
        domain_preimage = pickle.load(f)
        preimage_dict, dm = domain_preimage
        # print(preimage_dict)
        # print(dm)
        build_polyhedra(preimage_dict, dm, dataset)
        
        
def test():
    x = Variable(0)
    y = Variable(1)
    vars = [x,y]
    cs = Constraint_System()
    cs.insert(10*x + 5*y>= 0)
    cs.insert(x <= 3)
    cs.insert(y >= 0)
    cs.insert(y <= 3)
    poly_from_constraints = C_Polyhedron(cs)
    generators = poly_from_constraints.minimized_generators()
    # print(generators)  
    for generator in generators:
        print(generator)
        for i in range(len(vars)):
            coeff=float(generator.coefficient(vars[i]))
            divisor=float(generator.divisor())
            print(coeff/divisor)
            # print(generator.coefficient(vars[i]) + generator.divisor())
            
def test_convex_hull():
    rng = np.random.default_rng()
    points = rng.random((5, 2))   # 5 random points in 2-D
    print(points)
    hull = ConvexHull(points)
    print(hull.volume)       
        
def build_polyhedra(pre_image, dm, dataset, rnd_num=9): # rnd_num corresponds to multiplier
    if dataset == 'auto_park':   
        # To build the polyhedra you have to declare a few variables:
        x = Variable(0) # position x value
        y = Variable(1) # position x value
        vars = [x,y] 
        poly_num = len(pre_image)
        # And you can construct the constraint system with:
        poly_vol = []
        for j in range(poly_num): 
            A_arr, b_arr = pre_image[j]
            dm_poly = dm[j]
            dm_lb = dm_poly[0][0]
            dm_ub = dm_poly[1][0]
            print(dm_lb, dm_ub)
            if len(A_arr) != 0:
                constraint_system = Constraint_System()
                x_lb, x_ub = int(dm_lb[0]*multiplier), int(dm_ub[0]*multiplier)
                y_lb, y_ub = int(dm_lb[1]*multiplier), int(dm_ub[1]*multiplier)
                constraint_system.insert(int(1*multiplier)* vars[0]>=x_lb)
                constraint_system.insert(int(1*multiplier)* vars[0]<=x_ub)
                constraint_system.insert(int(1*multiplier)* vars[1]>=y_lb)
                constraint_system.insert(int(1*multiplier)* vars[1]<=y_ub)
                for i, coeffi in enumerate(A_arr):
                    coeffi = [int(round(ele, rnd_num)*multiplier) for ele in coeffi]
                    bias = int(round(b_arr[i],rnd_num)* multiplier)
                    print('check coeffi, bias', coeffi, bias)
                    constraint_system.insert(coeffi[0] * vars[0] +
                                            coeffi[1] * vars[1] +
                                            bias >= 0)
            # And then the polyhedra:
            poly = NNC_Polyhedron(constraint_system) 
            # poly = C_Polyhedron(constraint_system) 
            # generators = poly.minimized_generators()
            # for generator in generators:
            #     print(generator)
            #     for i in range(len(vars)):
            #         coeff=float(generator.coefficient(vars[i]))
            #         divisor=float(generator.divisor())
            #         print(coeff/divisor)
            vol = compute_volume(poly, vars) 
            poly_vol.append(vol)
    elif dataset == 'vcas':   
        # To build the polyhedra you have to declare a few variables:
        h = Variable(0) # relative altitude
        hA = Variable(1) # vert climb rate ownship
        hB = Variable(2) # vert climb rate intruder
        t = Variable(3) # time until loss of the horizontal distance
        vars = [h,hA,hB,t] 
        poly_num = len(pre_image)
        # And you can construct the constraint system with:
        poly_vol = []
        for j in range(poly_num):  # j is the new advisory
            A_arr, b_arr = pre_image[j]
            dm_poly = dm[j]
            dm_lb = dm_poly[0][0]
            dm_ub = dm_poly[1][0]
            print(dm_lb, dm_ub)
            if len(A_arr) != 0:
                constraint_system = Constraint_System()
                h_lb, h_ub = int(dm_lb[0]*multiplier), int(dm_ub[0]*multiplier)
                hA_lb, hA_ub = int(dm_lb[1]*multiplier), int(dm_ub[1]*multiplier)
                hB_lb, hB_ub = int(dm_lb[2]*multiplier), int(dm_ub[2]*multiplier)
                t_lb, t_ub = int(dm_lb[3]*multiplier), int(dm_ub[3]*multiplier)
                constraint_system.insert(int(1*multiplier)* vars[0]>=h_lb)
                constraint_system.insert(int(1*multiplier)* vars[0]<=h_ub)
                constraint_system.insert(int(1*multiplier)* vars[1]>=hA_lb)
                constraint_system.insert(int(1*multiplier)* vars[1]<=hA_ub)
                constraint_system.insert(int(1*multiplier)* vars[2]>=hB_lb)
                constraint_system.insert(int(1*multiplier)* vars[2]<=hB_ub)
                constraint_system.insert(int(1*multiplier)* vars[3]>=t_lb)
                constraint_system.insert(int(1*multiplier)* vars[3]<=t_ub)                
                for i, coeffi in enumerate(A_arr):
                    coeffi = [int(round(ele, rnd_num)*multiplier) for ele in coeffi]
                    bias = int(round(b_arr[i],rnd_num)* multiplier)
                    print('check coeffi, bias', coeffi, bias)
                    constraint_system.insert(coeffi[0] * vars[0] +
                                            coeffi[1] * vars[1] +
                                            coeffi[2] * vars[2] +
                                            coeffi[3] * vars[3] +
                                            bias >= 0)
            # And then the polyhedra:
            poly = NNC_Polyhedron(constraint_system) 
            # poly = C_Polyhedron(constraint_system) 
            vol = compute_volume(poly, vars) 
            poly_vol.append(vol)            
    print(poly_vol)
    compute_proportion(poly_vol, dataset)
    
def compute_proportion(poly_vol, dataset):
    approx_vol = sum(poly_vol)
    if dataset == 'auto_park':
        p = approx_vol/(0.5*0.5)
    elif dataset == 'vcas':
        p = approx_vol/(0.5*0.5*0.0002*1)
    else:
        print('Currently the dataset not supported')
        return
    print(p, '\n {}: {:.3f}'.format(dataset, p))
    
def compute_volume(polyhedron, vars):
    points = []
    for generator in polyhedron.minimized_generators():
        point = []
        for i in range(0, len(vars)):
            point.append(float(generator.coefficient(vars[i])) / float(generator.divisor()))
            # point.append(float(generator.coefficient(vars[i]) / generator.divisor()))
        points.append(point)
    # print(points)
    try:
        return ConvexHull(points).volume
    except:
        print(" ** error in volume computation **")
        
def build_polyhedra_orig(pre_image, model):  
    ''' deprecated, only for reference'''      
    # To build the polyhedra you have to declare a few variables:
    h = Variable(0) # h
    h_1 = Variable(1) # climb rate for ownship
    h_2 = 30 # constant climb rate for intruder
    t = Variable(2) # t
    vars = [h, h_1, t] 

    # And you can construct the constraint system with:
    for j in range(0, len(pre_image)):  # j is the new advisory
        if len(pre_image[j]) != 0:
            for constraints in pre_image[j]:  # regions for the new advisory
                constraint_system = Constraint_System()
                for constraint in constraints:
                    constraint_system.insert(constraint[1] * model.multiplier * model.vars[0] +
                                            constraint[2] * model.multiplier * model.vars[1] +
                                            constraint[3] * model.multiplier * model.vars[2] +
                                            constraint[0] * model.multiplier > 0)
                    
                    
    # And then the polyhedra:
    poly = NNC_Polyhedron(constraint_system)
    
if __name__ == "__main__":
    # Specify the dataset, output label (spec), region number
    # load_preimage('auto_park', 0, 2)
    load_preimage('vcas', 0, 6)
    
    # FOR Testing
    # test()
    # test_convex_hull()
    