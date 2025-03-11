# Transfer of Distributions between External Conditions with Normalizing Flows

Code for the paper TRADE: Transfer of Distributions between External Conditions with Normalizing Flows

### Installation
First go to https://github.com/noegroup/bgmol and follow the instructions to install the bgmol package, including openMM and mdtraj.
Then run the following commands in this github repository:
```bash
pip install -r requirements.txt
pip install -e .
```
Fixes for broken packages: 
- Currently the package `bgmol` uses `versioneer`, which only works for `python<3.12`.
- The package `lightning_trainable` is missing the requirments `einops` and `requests`, which need to be manually installed.
- bgmol relies on an old version of `mdtraj`, which is broken with current `cython`. A dirty fix is to change the import statements in `bgmol/tpl/hdf5.py` from
  ```python
  from mdtraj.utils.six import PY3
  ...
  from mdtraj.utils.six import string_types
  ```
  to
  ```python
  from six import PY3
  ...
  from six import string_types
  ```
- Temperature steerable splines are not provided in the public implementations of bgflow and nflows, we provide forks in requirements.txt with our implementation.

- Currently there exist several issues in bgflow, which lead to  
  1. Ignoring the temperature parameter passed to the Boltzmann generator for sampling
  2. Only being able to evaluate the nll of the Boltzmann generator at temperature 1.0
  3. Incorrect computation of the energy for augmented normalizing flows

  When trying to evaluate models trained with bgflow we therefore advise you to implement metrics and sampling functions yourself or directly using the BGFlowFlow class

### Usage
Define your configurations in a `config.yaml` file. You can find various default configs in the `configs` folder.
Then start the experiment using lightning trainable:
```bash
python3 -m lightning_trainable.launcher.fit configs/config.yaml
```
You can also use multiple config files at once. Later configs overwrite the values specified in earlier ones.

To run the experiments for the lattice model and the two-dimensional Gaussian mixture model, use 

```bash
python3 train_INN.py --tag <your experiment tag> --config_path ./configs/<data set name>/config_<experiment name>.json
```

### Datasets
You can download the toy datasets from google drive. Alanine dipeptide is downloaded on-demand.
```bash
curl -L -o data.zip "https://drive.google.com/uc?export=download&id=1cPHvXGPR2MyzPQNeqF_Q77MV0kQeGkfY"
unzip data.zip
```
Currently five datasets are implemented and can be used via specifying
Double well 2d:  
```yaml
dataset:
  name: double_well_2d
```
Gaussian Mixture Model 2d:  
```yaml
dataset:
  name: gmm_2d
```
Multi well 5d:
```yaml
dataset:
  name: multi_well_5d
```
Two Moons:
```yaml
dataset:
  name: two_moons
```
Alanine Dipeptide 66(/60)d:  
```yaml
dataset:
  name: ala2
```

### Citation

You can cite our work using the following bibtex entry:
```
@article{wahl2024trade,
  title={TRADE: Transfer of Distributions between External Conditions with Normalizing Flows},
  author={Wahl, Stefan and Rousselot, Armand and Draxler, Felix and K{\"o}the, Ullrich},
  journal={arXiv preprint arXiv:2410.19492},
  year={2024}
}
```
