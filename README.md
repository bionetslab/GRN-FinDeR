# GRN-FinDeR: Efficient P-value Computation for Gene-Regulatory Networks

## Overview

We here provide the software implementing our novel strategy for reducing the computational cost of tree-based GRN inference while maintaining reasonable runtime. The method clusters genes to create a fixed number of background distributions, enabling the computation of significance values for multiple genes simultaneously and reducing computational cost linearly.

Our software extends the popular GRNBoost2/arboreto package and can be used as an add-on. It outputs a list of regulatory interactions, their importance scores, and empirical p-values, computed up to a user-defined significance threshold.



![Schematic of the workflow](/img/flowchart_grn_finder.svg-1.pdf)

## Installation
## Usage examples

TODO

## Parameter Details & Input Format

TODO
