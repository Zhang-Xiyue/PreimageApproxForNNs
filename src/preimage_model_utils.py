import torch
import torch.nn as nn
import numpy as np
# NOTE newly added ones
# import torch.nn.functional as F
import arguments

def load_model_simple(model_name, model_path, model_info=None, weights_loaded=True):
    """
    Load the pytorch model architectures and weights
    """

    model_ori = build_model(model_name, model_info) 
    model_ori.eval()
    # print(model_ori)

    if not weights_loaded:
        return model_ori

    if model_path is not None:
        # Load pytorch model
        sd = torch.load(model_path)
        if 'state_dict' in sd:
            sd = sd['state_dict']
        if isinstance(sd, list):
            sd = sd[0]
        if not isinstance(sd, dict):
            raise NotImplementedError("Unknown model format, please modify model loader yourself.")
        model_ori.load_state_dict(sd)

    # elif test_arguments.Config["model"]["onnx_path"] is not None:
    #     # Load onnx model
    #     model_ori, _ = load_model_onnx(test_arguments.Config["model"]["onnx_path"])
    else:
        print("Warning: pretrained model path is not given!")
    return model_ori



def load_input_info(dataset_tp, truth_label, quant=False, trans=False):
        # Returns: X, labels, runnerup, data_max, data_min, eps, target_label.
        # X is the data matrix in (batch, ...).
        # labels are the groud truth labels, a tensor of integers.
        # runnerup is the runnerup label used for quickly verify against the runnerup (second largest) label, can be set to None.
        # data_max is the per-example perturbation upper bound, shape (batch, ...) or (1, ...).
        # data_min is the per-example perturbation lower bound, shape (batch, ...) or (1, ...).
        # eps is the Lp norm perturbation epsilon. Can be set to None if element-wise perturbation (specified by data_max and data_min) is used.
        # Target label is the targeted attack label; can be set to None.
        data_config = load_input_bounds(dataset_tp, truth_label, quant, trans)
        if len(data_config) == 5:
            X, labels, data_max, data_min, eps_new = data_config
            runnerup, target_label = None, None
        elif len(data_config) == 6:
            X, labels, data_max, data_min, eps_new, runnerup = data_config
            target_label = None
        elif len(data_config) == 7:
            X, labels, data_max, data_min, eps_new, runnerup, target_label = data_config
        else:
            print("Data config types not correct!")
            exit()
        assert X.size(0) == labels.size(0), "batch size of X and labels should be the same!"
        assert (data_max - data_min).min()>=0, "data_max should always larger or equal to data_min!"
        return X, labels, runnerup, data_max, data_min, eps_new, target_label

def load_input_bounds(dataset, truth_label, quant, trans):
    if dataset == "demo":
        X = torch.tensor([[0, 0]]).float()
        labels = torch.tensor([truth_label]).long()
        data_max = torch.tensor([[1, 1]]).reshape(1, -1)
        data_min = torch.tensor([[-1, -1]]).reshape(1, -1)
        eps = None        
    elif dataset == "Customized_beta_demo_data":
        X = torch.tensor([[0.5, -0.5]]).float()
        labels = torch.tensor([truth_label]).long()
        data_max = torch.tensor([[2, 1]]).reshape(1, -1)
        data_min = torch.tensor([[-1, -2]]).reshape(1, -1)
        eps = None   
    elif dataset == "Customized_toy_model":
        X = torch.tensor([[0, 0.5]]).float()
        labels = torch.tensor([truth_label]).long()
        data_max = torch.tensor([[2, 1]]).reshape(1, -1)
        data_min = torch.tensor([[-1, 0]]).reshape(1, -1)
        eps = None         
    elif dataset == "box":
        X = torch.tensor([[1, 1]]).float()
        labels = torch.tensor([truth_label]).long()
        data_max = torch.tensor([[2, 2]]).reshape(1, -1)
        data_min = torch.tensor([[0, 0]]).reshape(1, -1)
        eps = None
    elif dataset == 'cartpole':
        if quant:
            X = torch.tensor([[0.5, 0.25, 0.05, -0.1]]).float()
            # X = torch.tensor([[0, 0, 0, 0]]).float()
            # X = torch.tensor([[-0.25, -0.5, 0.05, 0.05]]).float()
            # X = torch.tensor([[0.25, 0.5, -0.05, -0.5]]).float()
            labels = torch.tensor([truth_label]).long()
            data_max = torch.tensor([[1, 0.5, 0.1, 0]]).reshape(1, -1)
            data_min = torch.tensor([[0, 0, 0, -0.2]]).reshape(1, -1)
        else:
            X = torch.tensor([[0.5, 1, -0.1, -1.5]]).float()
            # X = torch.tensor([[0, 0, 0, 0]]).float()
            # X = torch.tensor([[-0.25, -0.5, 0.05, 0.05]]).float()
            # X = torch.tensor([[0.25, 0.5, -0.05, -0.5]]).float()
            labels = torch.tensor([truth_label]).long()
            data_max = torch.tensor([[1, 2, 0, 0]]).reshape(1, -1)
            data_min = torch.tensor([[-1, 0, -0.2, -2]]).reshape(1, -1)
            # data_max = torch.tensor([[0.5, 1, 0.1, 1]]).reshape(1, -1)
            # data_min = torch.tensor([[-0.5, -1, -0.1, -1]]).reshape(1, -1)
            # data_max = torch.tensor([[0, 0, 0.1, 0.1]]).reshape(1, -1)
            # data_min = torch.tensor([[-0.5, -1, 0, 0]]).reshape(1, -1)
            # data_max = torch.tensor([[0.5, 1, 0, 0]]).reshape(1, -1)
            # data_min = torch.tensor([[0, 0, -0.1, -1]]).reshape(1, -1)
            # add the transformation, define the transformations for the observed values
        
        # if trans:
        #     distur = torch.tensor([[0.1, 0, 0, 0]]).reshape(1, -1)
        #     data_max = data_max + distur
        #     data_min = data_min - distur 
        eps = None
    elif dataset == 'dubinsrejoin':
        X = torch.tensor([[-0.1, 0.25, -0.5, 0.1, 0.5, 0, 0.35, -0.75]]).float()
        # X = torch.tensor([[-0.25, 0.25, -0.75, 0.05, 0.55, -0.05, 0.25, 0.05]]).float()
        labels = torch.tensor([truth_label]).long()
        data_max = torch.tensor([[0.0, 0.5, 0.0, 0.2, 0.6, 0.1, 0.5, 0.5]]).reshape(1, -1)
        data_min = torch.tensor([[-0.2, 0.0, -1, 0.0, 0.4, -0.1, 0.2, -0.5]]).reshape(1, -1)
        # data_max = torch.tensor([[0.0, 0.5, 0.0, 0.2, 0.6, 0.3, 0.0, 0.5]]).reshape(1, -1)
        # data_min = torch.tensor([[-0.2, 0.0, -1, 0.0, 0.4, -0.3, -0.2, -0.5]]).reshape(1, -1)
        # data_max = torch.tensor([[0.0, 0.5, 0.0, 0.2, 0.6, 0.3, 0.5, 0]]).reshape(1, -1)
        # data_min = torch.tensor([[-0.2, 0.0, -1, 0.0, 0.4, -0.3, 0.2, -1.5]]).reshape(1, -1)
        # data_max = torch.tensor([[0.2, 0.2, 0.5, 0.1, 0.6, 0.1, 0.5, 0.5]]).reshape(1, -1)
        # data_min = torch.tensor([[-0.2, -0.2, -0.5, 0, 0.5, -0.1, 0, -0.5]]).reshape(1, -1)
        eps = None
    elif dataset == 'lunarlander':
        if quant:
            X = torch.tensor([[-0.5, 0.5, 1.5, -1.5, -0.5, 0.05, 1, 1]]).float()
            labels = torch.tensor([truth_label]).long()
            data_max = torch.tensor([[0, 1.0, 2.0, -1, 0, 0.1, 1.0, 1.0]]).reshape(1, -1)
            data_min = torch.tensor([[-1, 0.0, 1, -2, -1, 0, 0.9, 0.9]]).reshape(1, -1)
            # data_max = torch.tensor([[0, 1.0, 2.0, 0, 0, 0.1, 1.0, 1.0]]).reshape(1, -1)
            # data_min = torch.tensor([[-1, 0.0, 0, -2, -1, -0.1, 0.9, 0.9]]).reshape(1, -1)
            # data_max = torch.tensor([[0.5, 2.0, 0.1, -0.5, 0.1, 0.1, 1.0, 1.0]]).reshape(1, -1)
            # data_min = torch.tensor([[-0.5, 1, -0.1, -1, -0.1, -0.1, 0.9, 0.9]]).reshape(1, -1)
        else:
            X = torch.tensor([[-0.5, 0.5, 1, -1, -0.5, 0, 0.75, 0.75]]).float()
            # X = torch.tensor([[-0.05, 0.05, 0.5, -0.5, -0.05, 0.05, 0.75, 0.75]]).float()
            # X = torch.tensor([[-0.05, 0.05, 0.05, -0.05, -0.05, 0.05, 0.75, 0.75]]).float()
            labels = torch.tensor([truth_label]).long()
            # if arguments.Config["preimage"]["compare_split"]:
            #     data_max = torch.tensor([[0, 1.0, 2.0, 2, 0, 0.1, 1.0, 1.0]]).reshape(1, -1)
            #     data_min = torch.tensor([[-1, 0.0, 0, -2, -1, -0.1, 0.9, 0.9]]).reshape(1, -1)
            # else:
            data_max = torch.tensor([[0, 1.0, 2.0, 0, 0, 0.1, 1.0, 1.0]]).reshape(1, -1)
            data_min = torch.tensor([[-1, 0.0, 0, -2, -1, -0.1, 0.9, 0.9]]).reshape(1, -1)
            # data_max = torch.tensor([[0, 0.1, 1, 0, 0, 0.1, 1, 1]]).reshape(1, -1)
            # data_min = torch.tensor([[-0.1, 0, 0, -1, -0.1, 0, 0.5, 0.5]]).reshape(1, -1)
            # data_max = torch.tensor([[0, 0.1, 0.1, 0, 0, 0.1, 1, 1]]).reshape(1, -1)
            # data_min = torch.tensor([[-0.1, 0, 0, -0.1, -0.1, 0, 0.5, 0.5]]).reshape(1, -1)
            # data_max = torch.tensor([[0, 0.1, 0.1, 0, 0, 0.1, 1, 1]]).reshape(1, -1)
            # data_min = torch.tensor([[-0.1, 0, 0, -0.1, -0.1, 0, 0.9, 0.9]]).reshape(1, -1)
        eps = None
    elif dataset == "auto_park":
        if quant:
            X = torch.tensor([[0, 0]]).float()
            labels = torch.tensor([truth_label]).long()
            data_max = torch.tensor([[0, 0]]).reshape(1, -1)
            data_min = torch.tensor([[-0.5, -0.5]]).reshape(1, -1)
        else:
            X = torch.tensor([[0, 0]]).float()
            # X = torch.tensor([[-0.25, -0.25]]).float()
            labels = torch.tensor([truth_label]).long()
            data_max = torch.tensor([[0.5, 0.5]]).reshape(1, -1)
            # data_max = torch.tensor([[0, 0]]).reshape(1, -1)
            data_min = torch.tensor([[-0.5, -0.5]]).reshape(1, -1)
        eps = None
    elif dataset == "auto_park_part":
        # X = torch.tensor([[-0.25, -0.25]]).float()
        X = torch.tensor([[0, 0]]).float()
        labels = torch.tensor([truth_label]).long()
        # data_max = torch.tensor([[0, 0]]).reshape(1, -1)
        # data_max = torch.tensor([[0.15, 0.15]]).reshape(1, -1)
        data_max = torch.tensor([[0.5, 0.5]]).reshape(1, -1)
        data_min = torch.tensor([[-0.5, -0.5]]).reshape(1, -1)
        eps = None
    elif dataset == "auto_park_all":
        # X = torch.tensor([[-0.25, -0.25]]).float()
        X = torch.tensor([[0, 0]]).float()
        labels = torch.tensor([truth_label]).long()
        # data_max = torch.tensor([[0, 0]]).reshape(1, -1)
        data_max = torch.tensor([[0.5, 0.5]]).reshape(1, -1)
        data_min = torch.tensor([[-0.5, -0.5]]).reshape(1, -1)
        eps = None
    elif dataset == "auto_park_part2":
        # X = torch.tensor([[-0.25, -0.25]]).float()
        X = torch.tensor([[-0.25, 0.25]]).float()
        labels = torch.tensor([truth_label]).long()
        # data_max = torch.tensor([[0, 0]]).reshape(1, -1)
        data_max = torch.tensor([[0, 0.5]]).reshape(1, -1)
        data_min = torch.tensor([[-0.5, 0]]).reshape(1, -1)
        eps = None
    elif dataset == "auto_park_part_1_2":
        # X = torch.tensor([[-0.25, -0.25]]).float()
        X = torch.tensor([[-0.25, 0]]).float()
        labels = torch.tensor([truth_label]).long()
        # data_max = torch.tensor([[0, 0]]).reshape(1, -1)
        data_max = torch.tensor([[0, 0.5]]).reshape(1, -1)
        data_min = torch.tensor([[-0.5, -0.5]]).reshape(1, -1)
        eps = None
    elif dataset == 'pos_neg_box':
        X = torch.tensor([[0, 0]]).float()
        labels = torch.tensor([truth_label]).long()
        data_max = torch.tensor([[1, 1]]).reshape(1, -1)
        data_min = torch.tensor([[-1, -1]]).reshape(1, -1)
        eps = None
    elif dataset == "Customized(\"custom_model_data\", \"simple_box_data\")":
        X = torch.tensor([[0., 0.]]).float()
        labels = torch.tensor([1]).long()
        # customized element-wise upper bounds
        data_max = torch.tensor([[0.25, 0.25]]).reshape(1, -1)
        # customized element-wise lower bounds
        data_min = torch.tensor([[-0.25, -0.25]]).reshape(1, -1)
        eps = None
    elif dataset == "vcas":
        # NOTE use the input info after the normalization as the NN requires
        if quant:
            X = torch.tensor([[0, 0, 0, 0]]).float()
            labels = torch.tensor([truth_label]).long()
            data_max = torch.tensor([[0, 0.5, -0.1499, 0.5]]).reshape(1, -1)
            data_min = torch.tensor([[-0.5, 0, -0.1501, -0.5]]).reshape(1, -1)
        else:
            # if arguments.Config["preimage"]["over_approx"]:
            #     X = torch.tensor([[0, 0, 0, 0.25]]).float()
            #     labels = torch.tensor([truth_label]).long()
            #     data_max = torch.tensor([[0.5, 0.5, 0.5, 0.5]]).reshape(1, -1)
            #     data_min = torch.tensor([[-0.5, -0.5, -0.5, 0]]).reshape(1, -1)
            # else:
            X = torch.tensor([[0, 0, 0.15, 0.25]]).float()
            labels = torch.tensor([truth_label]).long()
            data_max = torch.tensor([[0.5, 0.5, 0.151, 0.5]]).reshape(1, -1)
            data_min = torch.tensor([[-0.5, -0.5, 0.149, -0.5]]).reshape(1, -1)
            # upper_time_loss = arguments.Config["preimage"]["upper_time_loss"]
            # X = torch.tensor([[0, 0, 0, 0.5-upper_time_loss/2]]).float()
            # labels = torch.tensor([truth_label]).long()
            # data_max = torch.tensor([[0.5, 0.5, 0.5, 0.5]]).reshape(1, -1)
            # data_min = torch.tensor([[-0.5, -0.5, -0.5, -0.5]]).reshape(1, -1)
        eps = None
    return X, labels, data_max, data_min, eps

def load_input_bounds_numpy(dataset, quant=False, trans=False):
    if dataset == "Customized(\"custom_model_data\", \"simple_box_data\")":
        data_ub = np.array([1, 1.5])
        data_lb = np.array([-1.5, -1])
        output_num = 2
    elif dataset == 'demo':
        data_ub = np.array([1, 1])
        data_lb = np.array([-1, -1])
        output_num = 2        
    elif dataset == "cartpole":
        data_ub = np.array([0.5, 1, 0.1, 1])
        data_lb = np.array([-0.5, -1, -0.1, -1])
        if trans:
            distur = np.array([0.1, 0, 0, 0])
            data_ub = data_ub + distur
            data_lb = data_lb - distur
        output_num = 2
    elif dataset == 'dubinsrejoin':
        # data_ub = np.array([0.4, 0.5, 0.8, 0.4, 0.5, 0.1, 0.5, 0.5])   
        # data_lb = np.array([-0.4, -0.5, -0.8, -0.4, -0.5, -0.1, -0.5, -0.5])
        # data_ub = np.array([0, 0.3, -0.4, 0.2, 0.6, 0.1, 0.5, 0])   
        # data_lb = np.array([-0.2, 0.1, -0.8, 0, 0.4, -0.1, 0.2, -0.5])   
        data_ub = np.array([0, 0.5, -0.5, 0.5, 0.6, 0, 0.5, 0])   
        data_lb = np.array([-0.5, 0, -1, 0, 0.4, -0.5, 0, -0.5])     
        output_num = 8
    elif dataset == 'lunarlander':
        # data_ub = np.array([-0.5, 0.5, 1.5, 0, 0, 0.5, 1.0, 1.0])   
        # data_lb = np.array([-1.0, 0, 1, -0.5, -0.5, -0.5, 0, 0]) 
        data_ub = np.array([0, 1.0, 2, 0, 0, 0, 1.0, 1.0])   
        data_lb = np.array([-1.0, 0, 0, -1, -1, -1, 0.5, 0.5])       
        output_num = 4
    elif dataset == "box":
        data_ub = np.array([2, 2])
        data_lb = np.array([0, 0])
        output_num = 2
    elif dataset == "auto_park":
        if quant:
            data_ub = np.array([0, 0])            
            data_lb = np.array([-0.5, -0.5])
        else:
            data_ub = np.array([0.5, 0.5])
            data_lb = np.array([-0.5, -0.5])
        output_num = 4    
    elif dataset == "auto_park_part":    
        data_lb = np.array([-0.5, -0.5])
        data_ub = np.array([0, 0])
        output_num = 4   
    elif dataset == 'pos_neg_box':
        data_lb = np.array([-1, -1])
        data_ub = np.array([1, 1])
        output_num = 2        
    elif dataset == 'vcas':
        # NOTE usethe vcas lb, ub to normalized range
        if quant:
            # print('quanti', quant)
            # data_lb = np.array([0, 0, -0.5, -0.5])
            # data_ub = np.array([0.5, 0.5, 0, 0.5])
            # data_lb = np.array([0, 0, -0.1501, -0.5])
            # data_ub = np.array([0.5, 0.5, -0.1499, 0.5])
            data_lb = np.array([-0.5, 0, -0.1501, -0.5])
            data_ub = np.array([0, 0.5, -0.1499, 0.5])
        else:
            data_lb = np.array([-0.5, -0.5, 0.1499, -0.5])
            data_ub = np.array([0.5, 0.5, 0.1501, 0.5])
        output_num = 9
    return data_lb, data_ub, output_num

def build_model_activation(model_tp):
    if "Customized" in model_tp:
        net = TwoReluModel()
    elif model_tp == "mnist_6_100":
        net = MnistSixFeedForward()
    elif model_tp == "mnist_3_128":
        net = MnistThreeFeedForward()
    return net
def build_model(model_tp, model_info=None):
    if model_tp == 'simple_fnn':
        net = SimpleFeedForward()
    elif model_tp == 'demo_fnn':
        net = DemoFeedForward()
    elif model_tp == 'two_layer':
        net = TwoLayerFeedForward()  
    elif model_tp == 'auto_park_model':
        if model_info is not None:
            if model_info['hidden_layer_num'] == 2:
                net = AutoParkModelTwoLayer(hidden_dim=model_info['hidden_dim'])
            else:
                net = AutoParkModel(hidden_dim=model_info['hidden_dim'])
        else:
            net = AutoParkModel()
    return net
class MnistThreeFeedForward(nn.Module):
    def __init__(self, in_dim=784, hidd_dim=128,out_dim=10):
        super().__init__()    
        self.fc1 = nn.Linear(in_dim,hidd_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidd_dim,hidd_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidd_dim,out_dim)
    def forward(self,x):
        x = torch.flatten(x, start_dim=1)
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(self.relu1(x_fc1))
        x = self.fc3(self.relu2(x_fc2))
        return x
class MnistSixFeedForward(nn.Module):
    def __init__(self, in_dim=784, hidd_dim=100,out_dim=10):
        super().__init__()
        # self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_dim,hidd_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidd_dim,hidd_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidd_dim,hidd_dim)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidd_dim,hidd_dim)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(hidd_dim,hidd_dim)
        self.relu5 = nn.ReLU()
        self.fc6 = nn.Linear(hidd_dim,out_dim)
    def forward(self,x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(self.relu1(x_fc1))
        x_fc3 = self.fc3(self.relu2(x_fc2))
        x_fc4 = self.fc4(self.relu3(x_fc3))
        x_fc5 = self.fc5(self.relu4(x_fc4))
        x = self.fc6(self.relu5(x_fc5))
        return x, [x_fc1, x_fc2, x_fc3, x_fc4, x_fc5]
class TwoReluModel(nn.Module):
    def __init__(self, in_dim=2, out_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 2)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(2, out_dim)
        self.fc1.weight.data = torch.tensor([[1., 2.], [2., 1.]])
        self.fc1.bias.data = torch.tensor([0., 0.])
        self.fc2.weight.data = torch.tensor([[1., -1.], [1., 0.]])
        self.fc2.bias.data = torch.tensor([2., 0.])
    def forward(self,x):
        x_fc1 = self.fc1(x)
        x_relu = self.relu1(x_fc1)
        x = self.fc2(x_relu)
        return x, x_fc1
class AutoParkModel(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=20, out_dim=4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, inputs):
        return self.model(inputs)
class AutoParkModelTwoLayer(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=10, out_dim=4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, inputs):
        return self.model(inputs)
class TwoLayerFeedForward(nn.Module):
    "A two hidden layer fnn, for toy example illustration."
    def __init__(self, in_dim=2, hidden_dim=3, out_dim=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, inputs):
        return self.model(inputs)
    
class SimpleFeedForward(nn.Module):
    """A very simple model, just for test."""
    def __init__(self, in_dim=2, hidden_dim=5, out_dim=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, inputs):
        return self.model(inputs)
    
class DemoFeedForward(nn.Module):
    """A very simple model, 2 inputs, 2 ReLUs, 2 ReLUs, 2 outputs"""
    def __init__(self, in_dim=2, hidden_dim=2, out_dim=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.model[0].weight.data = torch.tensor([[1., 1.], [1., -1.]])
        self.model[0].bias.data = torch.tensor([0., 0.])
        self.model[2].weight.data = torch.tensor([[1., 3.], [-1., 2.]])
        self.model[2].bias.data = torch.tensor([0., 0.])
        self.model[4].weight.data = torch.tensor([[1., 0.], [-2., -1.]])
        self.model[4].bias.data = torch.tensor([0., 0.])
        
    def forward(self, inputs):
        return self.model(inputs)
    
def Toy_model(in_dim=2, out_dim=2):
    
    model = nn.Sequential(
        nn.Linear(in_dim, 2),
        nn.ReLU(),
        nn.Linear(2, 2),
        nn.ReLU(),
        nn.Linear(2, out_dim)
    )
    """[relu(x+2y)-relu(2x+y)+2, 0*relu(2x-y)+0*relu(-x+y)]"""
    model[0].weight.data = torch.tensor([[1., 2.], [2., 1.]])
    model[0].bias.data = torch.tensor([0., 0.])
    model[2].weight.data = torch.tensor([[1., -1.], [0., 0.]])
    model[2].bias.data = torch.tensor([2., 0.])
    return model