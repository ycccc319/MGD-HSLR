# Multi-Granularity Decomposition and Hierarchical Semantics Learning for Domain Adaptation Retrieval (MGD-HSLR)
This repository contains the codes of the MGD-HSLR.

## Code files (matlab implementation)
├─demo.m: experimental demonstration on Office31 (resnet50) dataset.<br>
├─MGDHSLR.m: the implementation of MGD-HSLR.<br>
├─construct_dataset.m: the construction of the dataset.<br>
└─utils: some auxiliary functions.

## Example usage
- Use addpath(genpath('./utils/')) to add the required auxiliary functions, after which you can use ```MGDHSLR(exp_data,options)``` to call MGD-HSLR wherever you need to, dataset can be constructed by using "construct_dataset.m".
- You can launch the program by executing "demo.m" in the root directory. The codes will be run directly without errors.
