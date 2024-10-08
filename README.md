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
'''
git clone https://github.com/Zhang-Xiyue/PreimageApproxForNNs.git
cd PreimageApproxForNNs
'''
2. Set up the virtual environment
You can select any environment manager depending on your habit. 
Set up the environment using the *requirements* file 
'''
pip install -r requirements.txt
'''
## Usage
Example scripts
## Data 
The dataset used for this project is available at [VNN-COMP2022](https://github.com/ChristopherBrix/vnncomp2022 benchmarks).