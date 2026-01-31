# Orb-ECG

We provide a minimal example script implementing a training run of Orb-ECG, as described in our paper "Frontier-Orbital Predictions from Coarse-Grained Geometries with Physics-Constrained Neural Hamiltonians". It instantiates a network and performs a 100-epoch training run. The final per-orbital R^2 values are printed at the end of training. An example output is given in results/train.log.

The provided dataset contains 10000 conformations of the BT molecule, genereated as described in the text. Each conformation also has the 8 orbital energy values from HOMO-3 to LUMO+3, as well as the per-atom pooled wavefunction densities.