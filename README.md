# Preimage Approximation for Neural Networks
This repository contains the code and data used in the paper:

"PREMAP: A Unifying PREiMage APproximation Framework for Neural Networks"

Authors: Xiyue Zhang, Benjie Wang, Marta Kwiatkowska, Huan Zhang

## Table of Contents
- Introduction
- Installation
- Usage
- Data
- Citation
- License
- Contact

## Introduction
This repository provides the implementation of a general and flexible preimage approximation framework designed to generate inputs that satisfy specific target properties.
It supports both under-approximation and over-approximation, with an anytime algorithm design that balances approximation precision and computational cost for user-specified trade-offs.

## Installation
Here summarises the general steps to take for setting up the project on your local machine.
1. Clone the repository
```
git clone https://github.com/Zhang-Xiyue/PreimageApproxForNNs.git
cd PreimageApproxForNNs
```
2. Set up the virtual environment
You can select any environment manager depending on your habit. 
Set up the environment using the *requirements* file 
```
pip install -r requirements.txt
```
Or if you have the conda environment manager, you can set up using the *environment.yml* file
```
conda env create -f environment.yml
```
If you runinto installation issue on a specific package, e.g., pplpy, you can modify the *environment.yml* file and specify a different version that works for your system, or you can try creating the environment without pplpy first then install it manually.

## Usage
### Example scripts
You can find the parameter configuration file in `src/preimg_configs` and fully-specified explanation for each argument in `arguments.py`. 
More complete YAML files and a separate document to explain the configurable parameters are coming ...

To generate preimage for the Cartpole task, run
```
python preimage_main.py --config preimg_configs/cartpole.yaml
```
You can find the detailed explanation of the most useful parameters in this [configuration file](https://github.com/Zhang-Xiyue/PreimageApproxForNNs/blob/main/src/preimg_configs/cartpole.yaml).


**Find more example scripts in this [example-script](https://github.com/Zhang-Xiyue/PreimageApproxForNNs/blob/main/example_script.md) file.**

## Data 
The dataset used for this project is available at [VNN-COMP2022](https://github.com/ChristopherBrix/vnncomp2022).

## Citation
- The **Preimage Under-Approximation** paper provides an abstraction and refinement framework to computer preimage under-approximations (inputs) for targeted output specifications.
- **PREMAP** enables both preimage under-approximations and over-approximations with additioanl features of Lagrangian optimisation.

We provide bibtex entries below.
```
@inproceedings{zhang2024provable,
  title={Provable preimage under-approximation for neural networks},
  author={Zhang, Xiyue and Wang, Benjie and Kwiatkowska, Marta},
  booktitle={International Conference on Tools and Algorithms for the Construction and Analysis of Systems},
  pages={3--23},
  year={2024},
  organization={Springer}
}

@article{zhang2024premap,
  title={PREMAP: A Unifying PREiMage APproximation Framework for Neural Networks},
  author={Zhang, Xiyue and Wang, Benjie and Kwiatkowska, Marta and Zhang, Huan},
  journal={arXiv preprint arXiv:2408.09262},
  year={2024}
}
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.
Auto_LiRPA and alpha-beta-CROWN are released under the BSD 3-Clause license. See the LICENSE file in LICENSE-abCROWN.

## Contact
For inquiries or further information, please contact Xiyue Zhang.

## Acknowledgements
PREMAP has been partially developed with the support of European Union’s ELSA – European Lighthouse on Secure and Safe AI, Horizon Europe, grant agreement No. 101070617 under UK guarantee.

<img src="logo/elsa.jpg" alt="elsa" style="width:70px;"/>