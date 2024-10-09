#########################################################################
##                                                                     ##
##     This program is licenced under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################

import os
import gzip
import collections
import csv
import re
from ast import literal_eval
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import onnx2pytorch
import onnx
import onnxruntime as ort
import preimage_arguments
import warnings
from attack_pgd import attack_pgd

# Import all model architectures.
from model_defs import *
from read_vnnlib import read_vnnlib
from onnx_opt import compress_onnx

# Import pytorch model util functions
from preimage_model_utils import build_model

def reshape_bounds(lower_bounds, upper_bounds, y, global_lb=None):
    with torch.no_grad():
        last_lower_bounds = torch.zeros(size=(1, lower_bounds[-1].size(1)+1), dtype=lower_bounds[-1].dtype, device=lower_bounds[-1].device)
        last_upper_bounds = torch.zeros(size=(1, upper_bounds[-1].size(1)+1), dtype=upper_bounds[-1].dtype, device=upper_bounds[-1].device)
        last_lower_bounds[:, :y] = lower_bounds[-1][:, :y]
        last_lower_bounds[:, y+1:] = lower_bounds[-1][:, y:]
        last_upper_bounds[:, :y] = upper_bounds[-1][:, :y]
        last_upper_bounds[:, y+1:] = upper_bounds[-1][:, y:]
        lower_bounds[-1] = last_lower_bounds
        upper_bounds[-1] = last_upper_bounds
        if global_lb is not None:
            last_global_lb = torch.zeros(size=(1, global_lb.size(1)+1), dtype=global_lb.dtype, device=global_lb.device)
            last_global_lb[:, :y] = global_lb[:, :y]
            last_global_lb[:, y+1:] = global_lb[:, y:]
            global_lb = last_global_lb
    return lower_bounds, upper_bounds, global_lb


def convert_mlp_model(model, dummy_input):
    model.eval()
    feature_maps = {}

    def get_feature_map(name):
        def hook(model, input, output):
            feature_maps[name] = output.detach()

        return hook

    def conv_to_dense(conv, inputs):
        b, n, w, h = inputs.shape
        kernel = conv.weight
        bias = conv.bias
        I = torch.eye(n * w * h).view(n * w * h, n, w, h)
        W = F.conv2d(I, kernel, stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups)
        # input_flat = inputs.view(b, -1)
        b1, n1, w1, h1 = W.shape
        # out = torch.matmul(input_flat, W.view(b1, -1)).view(b, n1, w1, h1)
        new_bias = bias.view(1, n1, 1, 1).repeat(1, 1, w1, h1)

        dense_w = W.view(b1, -1).transpose(1, 0)
        dense_bias = new_bias.view(-1)

        new_m = nn.Linear(in_features=dense_w.shape[1], out_features=dense_w.shape[0], bias=m.bias is not None)
        new_m.weight.data.copy_(dense_w)
        new_m.bias.data.copy_(dense_bias)

        return new_m

    new_modules = []
    modules = list(model.named_modules())[1:]
    for mi, (name, m) in enumerate(modules):

        if mi+1 < len(modules) and isinstance(modules[mi+1][-1], nn.Conv2d):
            m.register_forward_hook(get_feature_map(name))
            model(dummy_input)
            pre_conv_input = feature_maps[name]
        elif mi == 0 and isinstance(m, nn.Conv2d):
            pre_conv_input = dummy_input

        if isinstance(m, nn.Linear):
            new_m = nn.Linear(in_features=m.in_features, out_features=m.out_features, bias=m.bias is not None)
            new_m.weight.data.copy_(m.weight.data)
            new_m.bias.data.copy_(m.bias)
            new_modules.append(new_m)
        elif isinstance(m, nn.ReLU):
            new_modules.append(nn.ReLU())
        elif isinstance(m, nn.Flatten):
            pass
            # will flatten at the first layer
            # new_modules.append(nn.Flatten())
        elif isinstance(m, nn.Conv2d):
            new_modules.append(conv_to_dense(m, pre_conv_input))
        else:
            print(m, 'not support in convert_mlp_model')
            raise NotImplementedError

    #  add flatten at the beginning
    new_modules.insert(0, nn.Flatten())
    seq_model = nn.Sequential(*new_modules)

    return seq_model

def deep_update(d, u):
    """Update a dictionary based another dictionary, recursively (https://stackoverflow.com/a/3233356)."""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def get_pgd_acc(model, X, labels, eps, data_min, data_max, batch_size):
    start = preimage_arguments.Config["data"]["start"]
    total = preimage_arguments.Config["data"]["end"]
    clean_correct = 0
    robust_correct = 0
    model = model.to(device=preimage_arguments.Config["general"]["device"])
    X = X.to(device=preimage_arguments.Config["general"]["device"])
    labels = labels.to(device=preimage_arguments.Config["general"]["device"])
    if isinstance(data_min, torch.Tensor):
        data_min = data_min.to(device=preimage_arguments.Config["general"]["device"])
    if isinstance(data_max, torch.Tensor):
        data_max = data_max.to(device=preimage_arguments.Config["general"]["device"])
    if isinstance(eps, torch.Tensor):
        eps = eps.to(device=preimage_arguments.Config["general"]["device"])
    if preimage_arguments.Config["attack"]["pgd_alpha"] == 'auto':
        alpha = eps.mean() / 4 if isinstance(eps, torch.Tensor) else eps / 4
    else:
        alpha = float(preimage_arguments.Config["attack"]["pgd_alpha"])
    while start < total:
        end = min(start + batch_size, total)
        batch_X = X[start:end]
        batch_labels = labels[start:end]
        if preimage_arguments.Config["specification"]["type"] == "lp":
            # Linf norm only so far.
            data_ub = torch.min(batch_X + eps, data_max)
            data_lb = torch.max(batch_X - eps, data_min)
        else:
            # Per-example, per-element lower and upper bounds.
            data_ub = data_max[start:end]
            data_lb = data_min[start:end]
        clean_output = model(batch_X)

        best_deltas, last_deltas = attack_pgd(model, X=batch_X, y=batch_labels, epsilon=float("inf"), alpha=alpha,
                num_classes=preimage_arguments.Config["data"]["num_outputs"],
                attack_iters=preimage_arguments.Config["attack"]["pgd_steps"], num_restarts=preimage_arguments.Config["attack"]["pgd_restarts"],
                upper_limit=data_ub, lower_limit=data_lb, multi_targeted=True, lr_decay=preimage_arguments.Config["attack"]["pgd_lr_decay"],
                target=None, early_stop=preimage_arguments.Config["attack"]["pgd_early_stop"])
        attack_images = torch.max(torch.min(batch_X + best_deltas, data_ub), data_lb)
        attack_output = model(attack_images)
        clean_labels = clean_output.argmax(1)
        attack_labels = attack_output.argmax(1)
        batch_clean_correct = (clean_labels == batch_labels).sum().item()
        batch_robust_correct = (attack_labels == batch_labels).sum().item()
        if start == 0:
            print("Clean prediction for first a few examples:")
            print(clean_output[:10].detach().cpu().numpy())
            print("PGD prediction for first a few examples:")
            print(attack_output[:10].detach().cpu().numpy())
        print(f'batch start {start}, batch size {end - start}, clean correct {batch_clean_correct}, robust correct {batch_robust_correct}')
        clean_correct += batch_clean_correct
        robust_correct += batch_robust_correct
        start += batch_size
        del clean_output, best_deltas, last_deltas, attack_images, attack_output
    print(f'data start {preimage_arguments.Config["data"]["start"]} end {total}, clean correct {clean_correct}, robust correct {robust_correct}')
    return clean_correct, robust_correct


def get_test_acc(model, input_shape=None, X=None, labels=None, batch_size=256):
    device = preimage_arguments.Config["general"]["device"]
    if X is None and labels is None:
        # Load MNIST or CIFAR, used for quickly debugging.
        database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
        mean = torch.tensor(preimage_arguments.Config["data"]["mean"])
        std = torch.tensor(preimage_arguments.Config["data"]["std"])
        normalize = transforms.Normalize(mean=mean, std=std)
        if input_shape == (3, 32, 32):
            testset = torchvision.datasets.CIFAR10(root=database_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), normalize]))
        elif input_shape == (1, 28, 28):
            testset = torchvision.datasets.MNIST(root=database_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), normalize]))
        else:
            raise RuntimeError("Unable to determine dataset for test accuracy.")
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    else:
        testloader = [(X, labels)]
    total = 0
    correct = 0
    if device != 'cpu':
        model = model.to(device)
    print_first_batch = True
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if device != 'cpu':
                images = images.to(device)
                labels = labels.to(device)
            if preimage_arguments.Config["model"]["convert_model_to_NCHW"]:
                images = images.permute(0, 2, 3, 1)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if print_first_batch:
                print_first_batch = False
                for i in range(min(outputs.size(0), 10)):
                    print(f"Image {i} norm {images[i].abs().sum().item()} label {labels[i].item()} correct {labels[i].item() == outputs[i].argmax().item()}\nprediction {outputs[i].cpu().numpy()}")
    print(f'correct {correct} of {total}')


def unzip_and_optimize_onnx(path, onnx_optimization_flags='none'):
    if onnx_optimization_flags == 'none':
        if path.endswith('.gz'):
            onnx_model = onnx.load(gzip.GzipFile(path))
        else:
            onnx_model = onnx.load(path)
        return onnx_model
    else:
        print(f"Onnx optimization with flag: {onnx_optimization_flags}")
        npath = path + ".optimized"
        if os.path.exists(npath):
            print(f"Found existed optimized onnx model at {npath}")
            return onnx.load(npath)
        else:
            print(f"Generate optimized onnx model to {npath}")
            if path.endswith('.gz'):
                onnx_model = onnx.load(gzip.GzipFile(path))
            else:
                onnx_model = onnx.load(path)
            return compress_onnx(onnx_model, path, npath, onnx_optimization_flags, debug=True)


def inference_onnx(path, *inputs):
    sess = ort.InferenceSession(unzip_and_optimize_onnx(path).SerializeToString())
    names = [i.name for i in sess.get_inputs()]
    inp = dict(zip(names, inputs))
    res = sess.run(None, inp)
    # sess = ort.InferenceSession(path)
    # names = [i.name for i in sess.get_inputs()]
    # inp = dict(zip(names, inputs))
    # sess_input = sess.get_inputs()
    # sess_output = sess.get_outputs()
    # print(f"No. of inputs : {len(sess_input)}, No. of outputs : {len(sess_output)}")    
    # # predict with ONNX Runtime
    # output_names = [ output.name for output in sess_output]
    # res = sess.run(output_names=output_names,\
    #     input_feed={sess_input[0].name: inputs})
    #     # input_feed=inp)
    return res
@torch.no_grad()
def load_model_onnx_simple(path):
    quirks = {}
    pt_save_path = path[:-4]+'pt'
    if os.path.exists(pt_save_path):
        pytorch_model = torch.load(pt_save_path)
        # if "VertCAS" in path:
        #     onnx_shape = tuple([4])
        # else:
        #     onnx_shape = None
        #     raise NotImplementedError('onnx shape needs to be specified')
    # print('check path', pt_save_path)
    else:
        onnx_model = onnx.load(path)
        # print(f"The model is:\n{onnx_model}")
        # if "VertCAS" in path:
        #     onnx_shape = tuple([4])
        # else:
        #     onnx_shape = None
        #     raise NotImplementedError('onnx shape needs to be specified')
        pytorch_model = onnx2pytorch.ConvertModel(onnx_model, experimental=True, quirks=quirks)
        pytorch_model.eval()
        pytorch_model.to(dtype=torch.get_default_dtype())
        torch.save(pytorch_model, pt_save_path)
    # print(f"The pytorch model is:\n{pytorch_model}")
    return pytorch_model
    # return pytorch_model, onnx_shape
    

@torch.no_grad()
def load_model_onnx(path, compute_test_acc=False, quirks=None, input_shape=None):
    onnx_optimization_flags = preimage_arguments.Config["model"]["onnx_optimization_flags"]
    if preimage_arguments.Config["model"]["cache_onnx_conversion"]:
        path_cache = f'{path}.cache'
        if os.path.exists(path_cache):
            print(f'Loading converted model from {path_cache}')
            return torch.load(path_cache)
    quirks = {} if quirks is None else quirks
    if preimage_arguments.Config["model"]["onnx_quirks"]:
        try:
            config_quirks = literal_eval(preimage_arguments.Config["model"]["onnx_quirks"])
        except ValueError as e:
            print(f'ERROR: onnx_quirks {preimage_arguments.Config["model"]["onnx_quirks"]} cannot be parsed!')
            raise
        assert isinstance(config_quirks, dict)
        deep_update(quirks, config_quirks)
    print(f'Loading onnx {path} wih quirks {quirks}')

    # pip install onnx2pytorch
    onnx_model = unzip_and_optimize_onnx(path, onnx_optimization_flags)
    print(f"The model is:\n{onnx_model}")
    if preimage_arguments.Config["model"]["input_shape"] is None:

        if "VertCAS" in path:
            onnx_shape = tuple([4])
        else:
            # find the input shape from onnx_model generally
            # https://github.com/onnx/onnx/issues/2657
            input_all = [node.name for node in onnx_model.graph.input]
            input_initializer = [node.name for node in onnx_model.graph.initializer]
            net_feed_input = list(set(input_all) - set(input_initializer))
            net_feed_input = [node for node in onnx_model.graph.input if node.name in net_feed_input]

            if len(net_feed_input) != 1:
                # in some rare case, we use the following way to find input shape but this is not always true (collins-rul-cnn)
                net_feed_input = [onnx_model.graph.input[0]]

            onnx_input_dims = net_feed_input[0].type.tensor_type.shape.dim
            onnx_shape = tuple(d.dim_value for d in onnx_input_dims[1:])
    else:
        # User specify input_shape
        onnx_shape = preimage_arguments.Config["model"]["input_shape"][1:]

    pytorch_model = onnx2pytorch.ConvertModel(onnx_model, experimental=True, quirks=quirks)
    pytorch_model.eval()
    pytorch_model.to(dtype=torch.get_default_dtype())
    # print(pytorch_model)

    conversion_check_result = True
    try:
        # check conversion correctness
        # FIXME dtype of dummy may not match the onnx model, which can cause runtime error
        dummy = torch.randn([1, *onnx_shape])
        output_pytorch = pytorch_model(dummy).numpy()
        output_onnx = inference_onnx(path, dummy.numpy())[0]
        if "remove_relu_in_last_layer" in onnx_optimization_flags:
            output_pytorch = output_pytorch.clip(min=0)
        conversion_check_result = np.allclose(
            output_pytorch, output_onnx, 1e-4, 1e-5)
    except:
        warnings.warn(f'Not able to check model\'s conversion correctness')
        print('\n*************Error traceback*************')
        import traceback; print(traceback.format_exc())
        print('*****************************************\n')
    if not conversion_check_result:
        print('\n**************************')
        print('Model might not be converted correctly. Please check onnx conversion carefully.')
        print('**************************\n')

    if compute_test_acc:
        get_test_acc(pytorch_model, onnx_shape)

    # if test_arguments.Config["model"]["cache_onnx_conversion"]:
        # torch.save((pytorch_model, onnx_shape), path_cache)
    pt_save_path = path[:-4]+'pt'
    torch.save(pytorch_model, pt_save_path)
    print(f"The pytorch model is:\n{pytorch_model}")
    return pytorch_model, onnx_shape

def load_model(weights_loaded=True):
    """
    Load the model architectures and weights
    """

    assert preimage_arguments.Config["model"]["name"] is None or preimage_arguments.Config["model"]["onnx_path"] is None, (
        "Conflict detected! User should specify model path by either --model or --onnx_path! "
        "The cannot be both specified.")

    assert preimage_arguments.Config["model"]["name"] is not None or preimage_arguments.Config["model"]["onnx_path"] is not None, (
        "No model is loaded, please set --model <modelname> for pytorch model or --onnx_path <filename> for onnx model.")

    if preimage_arguments.Config['model']['name'] is not None:
        # You can customize this function to load your own model based on model name.
        try:
            model_ori = eval(preimage_arguments.Config['model']['name'])()
        except Exception as e:
            print(f'Cannot load pytorch model definition "{preimage_arguments.Config["model"]["name"]}()". '
                  f'"{preimage_arguments.Config["model"]["name"]}()" must be a callable that returns a torch.nn.Module object.')
            import traceback
            traceback.print_exc()
            exit()
        model_ori.eval()
        print(model_ori)

        if not weights_loaded:
            return model_ori

        if preimage_arguments.Config["model"]["path"] is not None:
            # Load pytorch model
            # You can customize this function to load your own model based on model name.
            sd = torch.load(preimage_arguments.Config["model"]["path"], map_location=torch.device('cpu'))
            if 'state_dict' in sd:
                sd = sd['state_dict']
            if isinstance(sd, list):
                sd = sd[0]
            if not isinstance(sd, dict):
                raise NotImplementedError("Unknown model format, please modify model loader yourself.")
            model_ori.load_state_dict(sd)

    elif preimage_arguments.Config["model"]["onnx_path"] is not None:
        # Load onnx model
        model_ori, _ = load_model_onnx(preimage_arguments.Config["model"]["onnx_path"])

    else:
        print("Warning: pretrained model path is not given!")

    return model_ori



########################################
# Preprocess and load the datasets
########################################
def preprocess_cifar(image, inception_preprocess=False, perturbation=False):
    """
    Proprocess images and perturbations.Preprocessing used by the SDP paper.
    """
    MEANS = np.array([125.3, 123.0, 113.9], dtype=np.float32)/255
    STD = np.array([63.0, 62.1, 66.7], dtype=np.float32)/255
    if inception_preprocess:
        # Use 2x - 1 to get [-1, 1]-scaled images
        rescaled_devs = 0.5
        rescaled_means = 0.5
    else:
        rescaled_means = MEANS
        rescaled_devs = STD
    if perturbation:
        return image / rescaled_devs
    else:
        return (image - rescaled_means) / rescaled_devs


def load_cifar_sample_data(normalized=True, MODEL="a_mix"):
    """
    Load sampled cifar data: 100 images that are classified correctly by each MODEL
    """
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets/sample100_unnormalized')
    X = np.load(os.path.join(database_path, MODEL, "X.npy"))
    if normalized:
        X = preprocess_cifar(X)
    X = np.transpose(X, (0, 3, 1, 2))
    y = np.load(os.path.join(database_path, MODEL, "y.npy"))
    runnerup = np.load(os.path.join(database_path, MODEL, "runnerup.npy"))
    X = torch.from_numpy(X.astype(np.float32))
    y = torch.from_numpy(y.astype(int))
    runnerup = torch.from_numpy(runnerup.astype(int))
    print("############################")
    if normalized:
        print("Sampled data loaded. Data already preprocessed!")
    else:
        print("Sampled data loaded. Data not preprocessed yet!")
    print("Shape:", X.shape, y.shape, runnerup.shape)
    print("X range:", X.max(), X.min(), X.mean())
    print("############################")
    return X, y, runnerup


def load_mnist_sample_data(MODEL="mnist_a_adv"):
    """
    Load sampled mnist data: 100 images that are classified correctly by each MODEL
    """
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets/sample100_unnormalized')
    X = np.load(os.path.join(database_path, MODEL, "X.npy"))
    X = np.transpose(X, (0, 3, 1, 2))
    y = np.load(os.path.join(database_path, MODEL, "y.npy"))
    runnerup = np.load(os.path.join(database_path, MODEL, "runnerup.npy"))
    X = torch.from_numpy(X.astype(np.float32))
    y = torch.from_numpy(y.astype(int))
    runnerup = torch.from_numpy(runnerup.astype(int))
    print("############################")
    print("Shape:", X.shape, y.shape, runnerup.shape)
    print("X range:", X.max(), X.min(), X.mean())
    print("############################")
    return X, y, runnerup


def load_dataset():
    """
    Load regular datasets such as MNIST and CIFAR.
    """
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
    normalize = transforms.Normalize(mean=preimage_arguments.Config["data"]["mean"], std=preimage_arguments.Config["data"]["std"])
    if preimage_arguments.Config["data"]["dataset"] == 'MNIST':
        loader = datasets.MNIST
    elif preimage_arguments.Config["data"]["dataset"] == 'CIFAR':
        loader = datasets.CIFAR10
    elif preimage_arguments.Config["data"]["dataset"] == 'CIFAR100':
        loader = datasets.CIFAR100
    else:
        raise ValueError("Dataset {} not supported.".format(preimage_arguments.Config["data"]["dataset"]))
    test_data = loader(database_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), normalize]))
    test_data.mean = torch.tensor(preimage_arguments.Config["data"]["mean"])
    test_data.std = torch.tensor(preimage_arguments.Config["data"]["std"])
    # set data_max and data_min to be None if no clip
    data_max = torch.reshape((1. - test_data.mean) / test_data.std, (1, -1, 1, 1))
    data_min = torch.reshape((0. - test_data.mean) / test_data.std, (1, -1, 1, 1))
    return test_data, data_max, data_min


def load_sampled_dataset():
    """
    Load sampled data and define the robustness region
    """
    if preimage_arguments.Config["data"]["dataset"] == "CIFAR_SAMPLE":
        X, labels, runnerup = load_cifar_sample_data(normalized=True, MODEL=preimage_arguments.Config['model']['name'])
        data_max = torch.tensor(preprocess_cifar(1.)).reshape(1,-1,1,1)
        data_min = torch.tensor(preprocess_cifar(0.)).reshape(1,-1,1,1)
        eps_temp = 2./255.
        eps_temp = torch.tensor(preprocess_cifar(eps_temp, perturbation=True)).reshape(1,-1,1,1)
    elif preimage_arguments.Config["data"]["dataset"] == "MNIST_SAMPLE":
        X, labels, runnerup = load_mnist_sample_data(MODEL=preimage_arguments.Config['model']['name'])
        data_max = torch.tensor(1.).reshape(1,-1,1,1)
        data_min = torch.tensor(0.).reshape(1,-1,1,1)
        eps_temp = 0.3
        eps_temp = torch.tensor(eps_temp).reshape(1,-1,1,1)
    return X, labels, data_max, data_min, eps_temp, runnerup


def load_sdp_dataset(eps_temp=None):
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets/sdp')
    if preimage_arguments.Config["data"]["dataset"] == "CIFAR_SDP":
        X = np.load(os.path.join(database_path, "cifar/X_sdp.npy"))
        X = preprocess_cifar(X)
        X = np.transpose(X, (0,3,1,2))
        y = np.load(os.path.join(database_path, "cifar/y_sdp.npy"))
        runnerup = np.copy(y)
        X = torch.from_numpy(X.astype(np.float32))
        y = torch.from_numpy(y.astype(int))
        runnerup = torch.from_numpy(runnerup.astype(int))

        if eps_temp is None: eps_temp = 2./255.
        eps_temp = torch.tensor(preprocess_cifar(eps_temp, perturbation=True)).reshape(1,-1,1,1)

        data_max = torch.tensor(preprocess_cifar(1.)).reshape(1,-1,1,1)
        data_min = torch.tensor(preprocess_cifar(0.)).reshape(1,-1,1,1)

        print("############################")
        print("Sampled data loaded. Data already preprocessed!")
        print("Shape:", X.shape, y.shape, runnerup.shape)
        print("X range:", X.max(), X.min(), X.mean())
        print("############################")
    elif preimage_arguments.Config["data"]["dataset"] == "MNIST_SDP":
        X = np.load(os.path.join(database_path, "mnist/X_sdp.npy"))
        X = np.transpose(X, (0,3,1,2))
        y = np.load(os.path.join(database_path, "mnist/y_sdp.npy"))
        runnerup = np.copy(y)
        X = torch.from_numpy(X.astype(np.float32))
        y = torch.from_numpy(y.astype(int))
        runnerup = torch.from_numpy(runnerup.astype(int))

        if eps_temp is None: eps_temp = torch.tensor(0.3)

        data_max = torch.tensor(1.).reshape(1,-1,1,1)
        data_min = torch.tensor(0.).reshape(1,-1,1,1)

        print("############################")
        print("Shape:", X.shape, y.shape, runnerup.shape)
        print("X range:", X.max(), X.min(), X.mean())
        print("############################")
    else:
        exit("sdp dataset not supported!")

    return X, y, data_max, data_min, eps_temp, runnerup


def load_generic_dataset(eps_temp=None):
    """Load MNIST/CIFAR test set with normalization."""
    print("Trying generic MNIST/CIFAR data loader.")
    test_data, data_max, data_min = load_dataset()
    if eps_temp is None:
        raise ValueError('You must specify an epsilon')
    testloader = torch.utils.data.DataLoader(test_data, batch_size=10000, shuffle=False, num_workers=4)
    X, labels = next(iter(testloader))
    runnerup = None
    # Rescale epsilon.
    eps_temp = torch.reshape(eps_temp / torch.tensor(preimage_arguments.Config["data"]["std"], dtype=torch.get_default_dtype()), (1, -1, 1, 1))

    return X, labels, data_max, data_min, eps_temp, runnerup


def load_eran_dataset(eps_temp=None):
    """
    Load sampled data and define the robustness region
    """
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets/eran')

    if preimage_arguments.Config["data"]["dataset"] == "CIFAR_ERAN":
        X = np.load(os.path.join(database_path, "cifar_eran/X_eran.npy"))
        mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, -1, 1, 1).astype(np.float32)
        std = np.array([0.2023, 0.1994, 0.201]).reshape(1, -1, 1, 1).astype(np.float32)
        X = (X - mean) / std

        labels = np.load(os.path.join(database_path, "cifar_eran/y_eran.npy"))
        runnerup = np.copy(labels)
        X = torch.from_numpy(X.astype(np.float32))
        labels = torch.from_numpy(labels.astype(int))
        runnerup = torch.from_numpy(runnerup.astype(int))
        if eps_temp is None: eps_temp = 2. / 255.

        eps_temp = torch.tensor(eps_temp / std).reshape(1, -1, 1, 1)
        data_max = torch.tensor((1. - mean) / std).reshape(1, -1, 1, 1)
        data_min = torch.tensor((0. - mean) / std).reshape(1, -1, 1, 1)

        print("############################")
        print("Sampled data loaded. Data already preprocessed!")
        print("Shape:", X.shape, labels.shape, runnerup.shape)
        print("X range:", X.max(), X.min(), X.mean())
        # print("epsilon:", eps_temp)
        # print("max, min:", data_max, data_min)
        print("Note runnerup label is empty here!")
        print("############################")

    elif preimage_arguments.Config["data"]["dataset"] == "MNIST_ERAN":
        X = np.load(os.path.join(database_path, "mnist_eran/X_eran.npy"))
        mean = 0.1307
        std = 0.3081
        X = (X - mean) / std

        labels = np.load(os.path.join(database_path, "mnist_eran/y_eran.npy"))
        runnerup = np.copy(labels)
        X = torch.from_numpy(X.astype(np.float32))
        labels = torch.from_numpy(labels.astype(int))
        runnerup = torch.from_numpy(runnerup.astype(int))
        if eps_temp is None: eps_temp = 0.3

        eps_temp = torch.tensor(eps_temp / std).reshape(1, -1, 1, 1)
        data_max = torch.tensor((1. - mean) / std).reshape(1, -1, 1, 1)
        data_min = torch.tensor((0. - mean) / std).reshape(1, -1, 1, 1)

        print("############################")
        print("Sampled data loaded. Data already preprocessed!")
        print("Shape:", X.shape, labels.shape, runnerup.shape)
        print("X range:", X.max(), X.min(), X.mean())
        # print("epsilon:", eps_temp)
        # print("max, min:", data_max, data_min)
        print("Note runnerup label is empty here!")
        print("############################")

    elif preimage_arguments.Config["data"]["dataset"] == "MNIST_ERAN_UN":
        X = np.load(os.path.join(database_path, "mnist_eran/X_eran.npy"))

        labels = np.load(os.path.join(database_path, "mnist_eran/y_eran.npy"))
        runnerup = np.copy(labels)
        X = torch.from_numpy(X.astype(np.float32))
        labels = torch.from_numpy(labels.astype(int))
        runnerup = torch.from_numpy(runnerup.astype(int))
        if eps_temp is None: eps_temp = 0.3

        eps_temp = torch.tensor(eps_temp).reshape(1, -1, 1, 1)
        data_max = torch.tensor(1.).reshape(1, -1, 1, 1)
        data_min = torch.tensor(0.).reshape(1, -1, 1, 1)

        print("############################")
        print("Sampled data loaded. No normalization used!")
        print("Shape:", X.shape, labels.shape, runnerup.shape)
        print("X range:", X.max(), X.min(), X.mean())
        # print("epsilon:", eps_temp)
        # print("max, min:", data_max, data_min)
        print("Note runnerup label is empty here!")
        print("############################")

    elif preimage_arguments.Config["data"]["dataset"] == "MNIST_MADRY_UN":
        X = np.load(os.path.join(database_path, "mnist_madry/X.npy")).reshape(-1, 1, 28, 28)
        labels = np.load(os.path.join(database_path, "mnist_madry/y.npy"))
        runnerup = np.copy(labels)
        X = torch.from_numpy(X.astype(np.float32))
        labels = torch.from_numpy(labels.astype(int))
        runnerup = torch.from_numpy(runnerup.astype(int))
        if eps_temp is None: eps_temp = 0.3

        eps_temp = torch.tensor(eps_temp).reshape(1, -1, 1, 1)
        data_max = torch.tensor(1.).reshape(1, -1, 1, 1)
        data_min = torch.tensor(0.).reshape(1, -1, 1, 1)

        print("############################")
        print("Sampled data loaded. No normalization used!")
        print("Shape:", X.shape, labels.shape, runnerup.shape)
        print("X range:", X.max(), X.min(), X.mean())
        # print("epsilon:", eps_temp)
        # print("max, min:", data_max, data_min)
        print("Note runnerup label is empty here!")
        print("############################")

    else:
        raise(f'Unsupported dataset {preimage_arguments.Config["data"]["dataset"]}')

    return X, labels, data_max, data_min, eps_temp, runnerup


def Customized(def_file, callable_name, *args, **kwargs):
    """Fully customized model or dataloader."""
    if def_file.endswith('.py'):
        spec = importlib.util.spec_from_file_location("customized", def_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(def_file)
    # Load model from a specified file.
    model_func = getattr(module, callable_name)
    customized_func = partial(model_func, *args, **kwargs)
    # We need to return a Callable which returns the model.
    return customized_func


def load_verification_dataset(eps_before_normalization):
    if preimage_arguments.Config["data"]["dataset"].startswith("Customized("):
        # FIXME (01/10/22): fully document customized data loader.
        # Returns: X, labels, runnerup, data_max, data_min, eps, target_label.
        # X is the data matrix in (batch, ...).
        # labels are the groud truth labels, a tensor of integers.
        # runnerup is the runnerup label used for quickly verify against the runnerup (second largest) label, can be set to None.
        # data_max is the per-example perturbation upper bound, shape (batch, ...) or (1, ...).
        # data_min is the per-example perturbation lower bound, shape (batch, ...) or (1, ...).
        # eps is the Lp norm perturbation epsilon. Can be set to None if element-wise perturbation (specified by data_max and data_min) is used.
        # Target label is the targeted attack label; can be set to None.
        data_config = eval(preimage_arguments.Config["data"]["dataset"])(eps=eps_before_normalization)
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
    target_label = None
    # Add your customized dataset here.
    if preimage_arguments.Config["data"]["pkl_path"] is not None:
        # FIXME (01/10/22): "pkl_path" should not exist in public code!
        # for oval20 base, wide, deep or other datasets saved in .pkl file, we load the pkl file here.
        assert preimage_arguments.Config["specification"]["epsilon"] is None, 'will use epsilon saved in .pkl file'
        gt_results = pd.read_pickle(preimage_arguments.Config["data"]["pkl_path"])
        test_data, data_max, data_min = load_dataset()
        X, labels = zip(*test_data)
        X = torch.stack(X, dim=0)
        labels = torch.tensor(labels)
        runnerup = None
        idx = gt_results["Idx"].to_list()
        X, labels = X[idx], labels[idx]
        target_label = gt_results['prop'].to_list()
        eps_new = gt_results['Eps'].to_list()
        print('Overwrite epsilon that saved in .pkl file, they should be after normalized!')
        eps_new = [torch.reshape(torch.tensor(i, dtype=torch.get_default_dtype()), (1, -1, 1, 1)) for i in eps_new]
        data_config = (X, labels, data_max, data_min, eps_new, runnerup, target_label)
    # Some special model loaders.
    elif "ERAN" in preimage_arguments.Config["data"]["dataset"] or "MADRY" in preimage_arguments.Config["data"]["dataset"]:
        data_config = load_eran_dataset(eps_temp=eps_before_normalization)
    elif "SDP" in preimage_arguments.Config["data"]["dataset"]:
        data_config = load_sdp_dataset(eps_temp=eps_before_normalization)
    elif "SAMPLE" in preimage_arguments.Config["data"]["dataset"]:
        # Sampled datapoints (a small subset of MNIST/CIFAR), only for reproducing some paper results.
        data_config = load_sampled_dataset()
    elif "CIFAR" in preimage_arguments.Config["data"]["dataset"] or "MNIST" in preimage_arguments.Config["data"]["dataset"]:
        # general MNIST and CIFAR dataset with mean/std defined in config file.
        data_config = load_generic_dataset(eps_temp=eps_before_normalization)
    else:
        exit("Dataset not supported in this file! Please customize load_verification_dataset() function in utils.py.")

    if len(data_config) == 5:
        (X, labels, data_max, data_min, eps_new) = data_config
        runnerup = None
    elif len(data_config) == 6:
        (X, labels, data_max, data_min, eps_new, runnerup) = data_config
    elif len(data_config) == 7:
        (X, labels, data_max, data_min, eps_new, runnerup, target_label) = data_config

    if preimage_arguments.Config["specification"]["norm"] != np.inf:
        if isinstance(preimage_arguments.Config["data"]["std"], (list, tuple)):
            assert preimage_arguments.Config["data"]["std"].count(preimage_arguments.Config["data"]["std"][0]) == len(
                preimage_arguments.Config["data"]["std"]), ('For non-Linf norm, we support only 1-d eps (all channels with the same perturbation). '
                'If you have more complex, per-channel eps (e.g., an ellipsoid L2 perturbation, you can '
                'add the data normalization into part of the model.')
            preimage_arguments.Config["data"]["std"] = preimage_arguments.Config["data"]["std"][0]
        else:
             preimage_arguments.Config["data"]["std"] = float(preimage_arguments.Config["data"]["std"])
        eps_new = eps_new[0, 0, 0, 0]  # only support eps as a scalar for non-Linf norm

    # FIXME (01/10/22): we should have a common interface for dataloader.
    return X, labels, runnerup, data_max, data_min, eps_new, target_label
def load_bounded_dataset(data_config):
    # FIXME (01/10/22): fully document customized data loader.
    # Returns: X, labels, runnerup, data_max, data_min, eps, target_label.
    # X is the data matrix in (batch, ...).
    # labels are the groud truth labels, a tensor of integers.
    # runnerup is the runnerup label used for quickly verify against the runnerup (second largest) label, can be set to None.
    # data_max is the per-example perturbation upper bound, shape (batch, ...) or (1, ...).
    # data_min is the per-example perturbation lower bound, shape (batch, ...) or (1, ...).
    # eps is the Lp norm perturbation epsilon. Can be set to None if element-wise perturbation (specified by data_max and data_min) is used.
    # Target label is the targeted attack label; can be set to None.
    # data_config = eval(test_arguments.Config["data"]["dataset"])(eps=None)
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

class Normalization(nn.Module):
    def __init__(self, mean, std, model):
        super(Normalization, self).__init__()
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)
        self.model = model

    def forward(self, x):
        return self.model((x - self.mean)/self.std)


def default_onnx_and_vnnlib_loader(file_root, onnx_path, vnnlib_path):
    vnnlib = read_vnnlib(os.path.join(file_root, vnnlib_path))

    model_ori, onnx_shape = load_model_onnx(os.path.join(file_root, onnx_path))
    shape = (-1, *onnx_shape)  # add the batch dim to onnx_shape

    return model_ori, shape, vnnlib


def construct_vnnlib(X, labels, runnerups, data_max, data_min, perturb_epsilon, target_labels, example_idx_list, dataset, init_run=False):
    vnnlib = []
    num_outputs = preimage_arguments.Config["data"]["num_outputs"]
    # Lower/Upper bound take the else branch
    if type(perturb_epsilon) == list:
        # Each example has different perturbations.
        perturb_epsilon = torch.cat(perturb_epsilon)
        perturb_epsilon = perturb_epsilon[example_idx_list]
    elif type(perturb_epsilon) == torch.Tensor:
        # Same perturbation for all examples.
        pass
    else:
        # No perturbation, use lower and upper bounds directly. 
        assert preimage_arguments.Config["specification"]["type"] == 'bound'

    if preimage_arguments.Config["specification"]["type"] == 'bound':
        assert preimage_arguments.Config["specification"]["norm"] == float("inf")
        x_lower = data_min.flatten(1)
        # print('check x_lower type', x_lower.type())
        x_upper = data_max.flatten(1)
    elif preimage_arguments.Config["specification"]["type"] == 'lp':
        if preimage_arguments.Config["specification"]["norm"] == float("inf"):
            if data_max is None:
                # perturb_eps is already normalized.
                x_lower = (X[example_idx_list] - perturb_epsilon).flatten(1)
                x_upper = (X[example_idx_list] + perturb_epsilon).flatten(1)
            else:
                x_lower = (X[example_idx_list] - perturb_epsilon).clamp(min=data_min).flatten(1)
                x_upper = (X[example_idx_list] + perturb_epsilon).clamp(max=data_max).flatten(1)
        else:
            x_lower = X[example_idx_list].flatten(1)
            x_upper = X[example_idx_list].flatten(1)
            # Save the actual perturbation epsilon to global variable dictionary.
            preimage_arguments.Globals["lp_perturbation_eps"] = perturb_epsilon
    else:
        raise ValueError(f'Unsupported perturbation type {preimage_arguments.Config["specification"]["type"]}')


    x_range = torch.stack([x_lower, x_upper], -1).numpy()

    for i in range(len(example_idx_list)):
        label = labels[example_idx_list[i]].view(1, 1)
        this_x_range = x_range[i]

        if preimage_arguments.Config["data"]["num_outputs"] > 1:
            # Multi-class.
            if preimage_arguments.Config["specification"]["robustness_type"] == "verified-acc":
                if init_run:
                    c = None
                    new_c = [(c, np.array([preimage_arguments.Config["bab"]["decision_thresh"]]))]
                else:
                    c = torch.eye(num_outputs).type_as(x_lower)[label].unsqueeze(1) - torch.eye(num_outputs).type_as(
                        x_lower).unsqueeze(0)
                    I = (~(label.unsqueeze(1) == torch.arange(num_outputs).type_as(label.data).unsqueeze(0)))
                    c = (c[I].view(1, num_outputs - 1, num_outputs)).numpy()
                    new_c = []
                    if dataset == 'dubinsrejoin': # dubinsrejoin is special, it uses two actions for two actualtors
                        for ii in range(num_outputs - 2):
                            # FIXME: we do not need the decision threshold, oh perhaps for some case studies
                            if ii >= 3:
                                spec_idx = ii - 2
                                c_var = c[:, ii]
                                c_var[0][0] = 0
                                c_var[0][ii+1] = 0
                                c_var[0][4] = 1
                                c_var[0][4+spec_idx] = -1
                                new_c.append((c_var, np.array([preimage_arguments.Config["bab"]["decision_thresh"]])))
                            # if ii >= 3:
                            #     spec_idx = ii - 3
                            #     c_var = c[:, ii]
                            #     c_var[0][3] = 0
                            #     c_var[0][ii+1] = 0
                            #     c_var[0][7] = 1
                            #     c_var[0][4+spec_idx] = -1
                            #     new_c.append((c_var, np.array([test_arguments.Config["bab"]["decision_thresh"]])))                            
                            else:
                                new_c.append((c[:, ii], np.array([preimage_arguments.Config["bab"]["decision_thresh"]])))
                            
                    else:                    
                        for ii in range(num_outputs - 1):
                            # FIXME: we do not need the decision threshold, oh perhaps for some case studies
                            new_c.append((c[:, ii], np.array([preimage_arguments.Config["bab"]["decision_thresh"]])))

            elif preimage_arguments.Config["specification"]["robustness_type"] == "specify-target":
                if target_labels is None:
                    # FIXME: have not thought through
                    c = np.zeros([1, num_outputs])
                    c[0, label] = 1
                    c[0, target_label] = -1
                    new_c = [(c, np.array([preimage_arguments.Config["bab"]["decision_thresh"]]))]
                else:
                    target_label = target_labels[example_idx_list[i]]
                    c = np.zeros([1, num_outputs])
                    c[0, label] = 1
                    c[0, target_label] = -1
                    new_c = [(c, np.array([preimage_arguments.Config["bab"]["decision_thresh"]]))]

            elif preimage_arguments.Config["specification"]["robustness_type"] == "runnerup":
                runnerup = runnerups[example_idx_list[i]]
                c = np.zeros([1, num_outputs])
                c[0, label] = 1
                c[0, runnerup] = -1
                new_c = [(c, np.array([preimage_arguments.Config["bab"]["decision_thresh"]]))]
        else:
            # Binary class, no target label.
            c = np.ones([1, 1])
            new_c = [(c, np.array([preimage_arguments.Config["bab"]["decision_thresh"]]))]

        print("check c", new_c)
        vnnlib.append([(this_x_range, new_c)])

    return vnnlib

def update_vnnlib(data_min_ori, label, vnnlib):
    new_vnnlib = []
    num_outputs = preimage_arguments.Config["data"]["num_outputs"]
    # No perturbation, instead we use lower and upper bounds directly.
    assert preimage_arguments.Config["specification"]["type"] == 'bound'
    x_lower = data_min_ori.flatten(1)   
    # x_range = torch.stack([x_lower, x_upper], -1).numpy()
    print('Old vnnlib', vnnlib)
    x_range = vnnlib[0][0]
    thres = vnnlib[0][1][0][1]


    # this_x_range = x_range[i]

    if preimage_arguments.Config["data"]["num_outputs"] > 1:
        # Multi-class.
        if preimage_arguments.Config["specification"]["robustness_type"] == "verified-acc":
            c = torch.eye(num_outputs).type_as(x_lower)[label].unsqueeze(1) - torch.eye(num_outputs).type_as(
                x_lower).unsqueeze(0)
            I = (~(label.unsqueeze(1) == torch.arange(num_outputs).type_as(label.data).unsqueeze(0)))
            c = (c[I].view(1, num_outputs - 1, num_outputs)).numpy()
            new_c = []
            for ii in range(num_outputs - 1):
                new_c.append((c[:, ii], thres))
    else:
        # Binary class, no target label.
        c = np.ones([1, 1])
        new_c = [(c, np.array([preimage_arguments.Config["bab"]["decision_thresh"]]))]

    new_vnnlib = [(x_range, new_c)]
    print('new vnnlib', new_vnnlib)

    return new_vnnlib
def parse_run_mode():
    """ parse running by vnnlib or customized data
     if using customized data, we convert them to vnnlib format
     """
    file_root = model_ori = vnnlib_all = shape = None

    if preimage_arguments.Config["general"]["csv_name"] is not None and preimage_arguments.Config["specification"]["vnnlib_path"] is None:
        # A CSV filename is specified, and we will go over all models and specs in this csv file.
        # Used for running VNN-COMP benchmarks in batch mode.
        # In this case, vnnlib_path should not be specified, otherwise we will run only a single model/spec.
        run_mode = 'csv_file'
        file_root = preimage_arguments.Config["general"]["root_path"]

        with open(os.path.join(file_root, preimage_arguments.Config["general"]["csv_name"]), newline='') as csv_f:
            reader = csv.reader(csv_f, delimiter=',')

            csv_file = []
            for row in reader:
                # In VNN-COMP each line of the csv containts 3 elements: model, vnnlib, timeout
                csv_file.append(row)

        if len(csv_file[0]) == 1:
            # Each row contains only one item, which is the vnnlib spec. So we load and return the model only once here.
            # This case is used when we have a batch of vnnlib specs to verify with one model either pytorch or ONNX.
            model_ori = load_model()

        save_path = 'a-b-crown_[{}]_start={}_end={}_iter={}_b={}_timeout={}_branching={}-{}-{}_lra-init={}_lra={}_lrb={}_PGD={}_cplex_cuts={}_initial_max_domains={}.npz'.format(
                   os.path.splitext(preimage_arguments.Config["general"]["csv_name"])[0], preimage_arguments.Config["data"]["start"],
                   preimage_arguments.Config["data"]["end"], preimage_arguments.Config["solver"]["beta-crown"]["iteration"],
                   preimage_arguments.Config["solver"]["batch_size"],
                   preimage_arguments.Config["bab"]["timeout"], preimage_arguments.Config["bab"]["branching"]["method"],
                   preimage_arguments.Config["bab"]["branching"]["reduceop"],
                   preimage_arguments.Config["bab"]["branching"]["candidates"],
                   preimage_arguments.Config["solver"]["alpha-crown"]["lr_alpha"],
                   preimage_arguments.Config["solver"]["beta-crown"]["lr_alpha"],
                   preimage_arguments.Config["solver"]["beta-crown"]["lr_beta"], preimage_arguments.Config["attack"]["pgd_order"],
                   preimage_arguments.Config["bab"]["cut"]["cplex_cuts"],
                   preimage_arguments.Config["bab"]["initial_max_domains"])

        preimage_arguments.Config["data"]["end"] = min(preimage_arguments.Config["data"]["end"], reader.line_num)
        if preimage_arguments.Config["data"]["start"] != 0 or preimage_arguments.Config["data"]["end"] != reader.line_num:
            assert 0 <= preimage_arguments.Config["data"]["start"] <= reader.line_num and preimage_arguments.Config["data"]["end"] > preimage_arguments.Config["data"]["start"], \
                    "specified --start or --end out of range: start={}, end={}, total_in_csv={}".format(preimage_arguments.Config["data"]["end"], preimage_arguments.Config["data"]["start"], reader.line_num)
            print("customized start/end sample from instance {} to {} in {}".format(preimage_arguments.Config["data"]["start"], preimage_arguments.Config["data"]["end"], preimage_arguments.Config["general"]["csv_name"]))
        else:
            print("no customized start/end sample, testing all samples in {}".format(preimage_arguments.Config["general"]["csv_name"]))
            preimage_arguments.Config["data"]["start"], preimage_arguments.Config["data"]["end"] = 0, reader.line_num
        example_idx_list = csv_file[preimage_arguments.Config["data"]["start"]:preimage_arguments.Config["data"]["end"]]
    elif preimage_arguments.Config["model"]["onnx_path"] is not None and preimage_arguments.Config["specification"]["vnnlib_path"] is not None:
        # A onnx file and a vnnlib file is specified, run this onnx file with vnnlib, ignore csv file.
        # Used for VNN-COMP in single instance mode, will be used in run_instance.sh
        run_mode = 'single_vnnlib'
        preimage_arguments.Config["data"]["start"], preimage_arguments.Config["data"]["end"] = 0, 1
        csv_file = [(preimage_arguments.Config["model"]["onnx_path"], preimage_arguments.Config["specification"]["vnnlib_path"],
                     preimage_arguments.Config["bab"]["timeout"])]
        save_path = preimage_arguments.Config["general"]["results_file"]
        file_root = ''
        example_idx_list = csv_file[preimage_arguments.Config["data"]["start"]:preimage_arguments.Config["data"]["end"]]
    elif preimage_arguments.Config["general"]["csv_name"] is None:
        # No CSV specified, we will use specifications defined in yaml file.
        # This part replaces the old robustness_verifier.py interface.
        run_mode = 'customized_data'
        # Load Pytorch or ONNX model depends on the model path or onnx_path is given.
        model_ori = load_model(weights_loaded=True)
        if preimage_arguments.Config["specification"]["vnnlib_path"] is None:
            # Lp norm perturbation, replacing robustness_verifier.py
            if preimage_arguments.Config["specification"]["epsilon"] is not None:
                perturb_epsilon = torch.tensor(preimage_arguments.Config["specification"]["epsilon"], dtype=torch.get_default_dtype())
            else:
                print('No epsilon defined!')
                perturb_epsilon = None
            X, labels, runnerup, data_max, data_min, perturb_epsilon, target_label = load_verification_dataset(perturb_epsilon)

            if preimage_arguments.Config["data"]["data_idx_file"] is not None:
                # Go over a list of data indices.
                with open(preimage_arguments.Config["data"]["data_idx_file"]) as f:
                    example_idx_list = re.split(r'[;|,|\n|\s]+', f.read().strip())
                    example_idx_list = [int(b_id) for b_id in example_idx_list]
                    print(f'Example indices (total {len(example_idx_list)}): {example_idx_list}')
            else:
                # By default, we go over all data.
                example_idx_list = list(range(X.shape[0]))
            example_idx_list = example_idx_list[preimage_arguments.Config["data"]["start"]:  preimage_arguments.Config["data"]["end"]]
            vnnlib_all = construct_vnnlib(X, labels, runnerup, data_max, data_min, perturb_epsilon, target_label, example_idx_list)
            shape = [-1] + list(X.shape[1:])
        else:
            # Using vnnlib specification (e.g., loading a pytorch model and use vnnlib to define general specification).
            example_idx_list = [0]
            vnnlib = read_vnnlib(preimage_arguments.Config["specification"]["vnnlib_path"])
            assert preimage_arguments.Config["model"]["input_shape"] is not None, 'vnnlib does not have shape information, please specify by --input_shape'
            shape = preimage_arguments.Config["model"]["input_shape"]
            vnnlib_all = [vnnlib]  # Only 1 vnnlib file.

        if preimage_arguments.Config['model']['name'] is None:
            # use onnx model prefix as model_name
            model_name = preimage_arguments.Config["model"]["onnx_path"].split('.onnx')[-2].split('/')[-1]
        elif "Customized" in preimage_arguments.Config['model']['name']:
            model_name = "Customized_model"
        else:
            model_name = preimage_arguments.Config['model']['name']
        save_path = '{}_{}_{}'.format(model_name, preimage_arguments.Config["data"]["dataset"],preimage_arguments.Config["solver"]["bound_prop_method"])
        # save_path = 'Verified_ret_[{}]_start={}_end={}_iter={}_b={}_timeout={}_branching={}-{}-{}_lra-init={}_lra={}_lrb={}_PGD={}_cplex_cuts={}_multiclass={}.npy'.format(
        #            model_name, test_arguments.Config["data"]["start"], test_arguments.Config["data"]["end"],
        #            test_arguments.Config["solver"]["beta-crown"]["iteration"],
        #            test_arguments.Config["solver"]["batch_size"],
        #            test_arguments.Config["bab"]["timeout"], test_arguments.Config["bab"]["branching"]["method"],
        #            test_arguments.Config["bab"]["branching"]["reduceop"],
        #            test_arguments.Config["bab"]["branching"]["candidates"],
        #            test_arguments.Config["solver"]["alpha-crown"]["lr_alpha"],
        #            test_arguments.Config["solver"]["beta-crown"]["lr_alpha"],
        #            test_arguments.Config["solver"]["beta-crown"]["lr_beta"],
        #            test_arguments.Config["attack"]["pgd_order"], test_arguments.Config["bab"]["cut"]["cplex_cuts"],
        #            test_arguments.Config["solver"]["multi_class"]["multi_class_method"])

    else:
        raise NotImplementedError

    print(f'Internal results will be saved to {save_path}.')
    # FIXME_NOW: model_ori should not be handled in this function! Do it in the utility function that loads models for all cases.
    return run_mode, save_path, file_root, example_idx_list, model_ori, vnnlib_all, shape


def parse_run_mode_simple():
    """ parse running by vnnlib or customized data
     if using customized data, we convert them to vnnlib format
     """
    file_root = model_ori = vnnlib_all = shape = None

    # No CSV specified, we will use specifications defined in yaml file.
    # This part replaces the old robustness_verifier.py interface.
    run_mode = 'customized_data'
    # Load Pytorch or ONNX model depends on the model path or onnx_path is given.
    model_ori = load_model_simple(weights_loaded=True)
    X, labels, runnerup, data_max, data_min, perturb_epsilon, target_label = load_bounded_dataset()
    example_idx_list = list(range(X.shape[0]))

    vnnlib_all = construct_vnnlib(X, labels, runnerup, data_max, data_min, perturb_epsilon, target_label, example_idx_list)
    shape = [-1] + list(X.shape[1:])


    if preimage_arguments.Config['model']['name'] is None:
        # use onnx model prefix as model_name
        model_name = preimage_arguments.Config["model"]["onnx_path"].split('.onnx')[-2].split('/')[-1]
    elif "Customized" in preimage_arguments.Config['model']['name']:
        model_name = "Customized_model"
    else:
        model_name = preimage_arguments.Config['model']['name']

    save_path = 'Verified_ret_[{}]_start={}_end={}_iter={}_b={}_timeout={}_branching={}-{}-{}_lra-init={}_lra={}_lrb={}_PGD={}_cplex_cuts={}_multiclass={}.npy'.format(
                model_name, preimage_arguments.Config["data"]["start"], preimage_arguments.Config["data"]["end"],
                preimage_arguments.Config["solver"]["beta-crown"]["iteration"],
                preimage_arguments.Config["solver"]["batch_size"],
                preimage_arguments.Config["bab"]["timeout"], preimage_arguments.Config["bab"]["branching"]["method"],
                preimage_arguments.Config["bab"]["branching"]["reduceop"],
                preimage_arguments.Config["bab"]["branching"]["candidates"],
                preimage_arguments.Config["solver"]["alpha-crown"]["lr_alpha"],
                preimage_arguments.Config["solver"]["beta-crown"]["lr_alpha"],
                preimage_arguments.Config["solver"]["beta-crown"]["lr_beta"],
                preimage_arguments.Config["attack"]["pgd_order"], preimage_arguments.Config["bab"]["cut"]["cplex_cuts"],
                preimage_arguments.Config["solver"]["multi_class"]["multi_class_method"])



    print(f'Internal results will be saved to {save_path}.')
    # FIXME_NOW: model_ori should not be handled in this function! Do it in the utility function that loads models for all cases.
    return run_mode, save_path, file_root, example_idx_list, model_ori, vnnlib_all, shape
