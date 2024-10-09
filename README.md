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
## Usage
Example scripts
## Data 
The dataset used for this project is available at [VNN-COMP2022](https://github.com/ChristopherBrix/vnncomp2022 benchmarks).

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
Auto_LiRPA and alpha-beta-CROWN are released under the BSD 3-Clause license. See the LICENSE file in LICENSE-abCROWN

## Contact
For inquiries or further information, please contact Xiyue Zhang.
