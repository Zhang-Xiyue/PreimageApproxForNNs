##################################################################################
##      This file is for argument configuration for preimage approximation      ##
##################################################################################

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Preimage Approx for neural networks")
    parser.add_argument('--dataset', type=str, default='dubinsrejoin', choices=["auto_park", "vcas", "cartpole", "lunarlander", "dubinsrejoin"])
    parser.add_argument('--vcas', type=int, default=1, help="(from 1-9) Index of models for VCAS classification problem. 0 indicates not vcas-model")  
    parser.add_argument('--label', type=int, default=0, help='which label to build input preimage for')
    # for ablation study of the proposed method
    parser.add_argument('--prioritize', type=bool, default=True, help='whether to use domain prioritization')
    parser.add_argument('--smart', type=bool, default=True, help='whether to use the proposed smart strategy')
    # for effect evaluation and quantitative analysis
    parser.add_argument('--effect', type=bool, default=False, help='whether to evaluate the effect of sampling size')
    parser.add_argument('--record', type=bool, default=False, help='whether to record the targeted coverage through iterations') 
    parser.add_argument('--save_process', type=bool, default=False, help='whether to record the detailed polytope info through iterations') 
    parser.add_argument('--quant', type=bool, default=False, help='whether to perform quantitative analysis')  
    parser.add_argument('--depth', type=bool, default=False, help='whether to eval depth')
    parser.add_argument('--width', type=bool, default=False, help='whether to eval width')
    
    # parameters
    parser.add_argument('--threshold', type=float, default=0.75, help='the polytope coverage threshold')
    parser.add_argument('--sample_num', type=int, default=10000, help='sample number to evaluate polytope coverage')

    # model name and model path 
    parser.add_argument('--model_name', type=str, default='auto_park_model')    
    parser.add_argument('--model', type=str, default="model_auto_park_auto_park_model_20.pt", help='The network model path, the extension can be .onnx, .pt') 
    
    # model config information for comparison with different size 
    parser.add_argument('--hidden_layer_num', type=int, default=1, help="Number of hidden layer for classification problem. For auto park") 
    parser.add_argument('--output_dim', type=int, default=4, help="Number of classes for classification problem.") 
    parser.add_argument('--hidden_dim', type=int, default=20, help="Number of neurons of hidden layer for classification problem. For auto park") 
    # output specification number
    parser.add_argument('--initial_max_domains', type=int, default=3,
                          help='Output specification number for multi-class problems',
                          )
    parser.add_argument('--result_dir', type=str, default='./results')
    parser.add_argument('--add_layer', type=bool, default=False)
    parser.add_argument('--base', type=str, default='alpha-crown')
    args = parser.parse_args()
    return args