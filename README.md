# Provable Preimage Under-Approximation for Neural Networks

This is the repository for preimage under-approximation approach ([PreimgApprox](https://arxiv.org/abs/2305.03686)). It includes
* the implementation of the preimage approximation algorithm
* the benchmark data
* the required environment configuration
* instructions on how to reproduce the results
* github repository of alpha-beta-crown for bound propagation
(This project is licensed under the terms of the MIT license.)


## Components
* `preimage_main.py`: main interface for evaluation
* `preimage_approx_batch_input_split.py`: module for global branching and refinement
* `preimage_arguments.py`: module for setting up alpha-beta-crown arguments
* `preimage_branching_heuristics.py`: module for feature-splitting parallelization
* `preimage_compute_volume.py`: module for computing the volume of preimage polytopes
* `preimage_CROWN_solver.py`: module for linear bound computation
* `preimage_model_utils.py`: utility module for model wrapper and loading
* `preimage_parse_args.py`: module for setting up preimage approximation arguments
* `preimage_polyhedron_util.py`: utility module for polytope (under-approximation) evaluation
* `preimage_read_vnnlib.py`: module for vnnlib wrapper
* `preimage_subdomain_queue.py`: module for subdomain class and management
* `preimage_utils.py`: utility module for calling upon crown
* `preimage_optimize_input_poly.py`: module for computing polytope-volume guided objective function (contained in `alpha-beta-CROWN/auto_LiRPA`) 

Below we introduce a list of core modules for preimage abstraction:

1. auto_LiPRA
* optimize_input_poly: this file contains the core function to compute the approximated preimage polytope volume under three different designs, including: `concretize_poly_vol` and  `concretize_poly_vol_LSE` for preimage polytope volume without and with LSE function for K-specification for input splitting planes.

## Environment requirements
We recommend installing the required dependencies with conda using the following command:
```
conda env create -f environment.yml
```

## Benchmark tasks and evaluation
### Benchmark tasks
The models of the benchmark tasks are stored in the `model_dir` folder, including models for vehicle parking, VCAS, and reinforcement learning tasks from [VNN-COMP 2022](https://github.com/ChristopherBrix/vnncomp2022\_benchmarks). 

### Evaluation
To run the preimage approximation, use the following command:
``` bash
python preimage_main.py --dataset 'vcas'
```
* use the `dataset` argument to specify the benchmark task.
* use the `result_dir` argument to specify the directory for saving evaluation results.
* usage of other arguments is detailed in the arguments parsing file.

 