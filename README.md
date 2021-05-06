# Warning 

**Quandarium is a work in process.** It doesn't reach its first stable version yet and its source code is changing every week. **Use at your own risk.**

# Quandarium 

This package present several tools to deal with Quantum Chemistry (QC) data of molecular systems (non-periodic): 
* Extraction from QC calculations;
* Molecular analysis, seuch as effective coordination bonds and exposed area;
* AtoMF (Atomic to Molecular Featurization);
* Correlation analysis and correlation significance analysis with bootstrap.

## Install:

First, download the package:
```bash
cd 
git clone https://github.com/johnatanmucelini/Quandarium.git
```

Then, add the following lines at the top of your python code:
```python
import sys
sys.path.append('~/Quandarium/')
import quandarium as qd
``` 

## Usage:

I will improve this section further.

## Requirements:

The following python packages are prerequisites:
- **numpy**
- **scipy**
- **pandas**
- **atomic simulation environment**
- **scikit-learn**
- **matplotlib**
- **seaborn**

If you employ Anaconda package management, you can install the packages with the following commands:
```bash 
conda install pandas numpy scipy matplotlib seaborn scikit-learn
conda install -c conda-forge ase
```

## References:

The following papers have employed this package:

1. Methane dehydrogenation on 3d 13-atom transition-metal clusters: A density functional theory investigation combined with Spearman rank correlation analysis. *Fuel*, **2020**, 275, 117790. DOI: [10.1016/j.fuel.2020.117790](https://www.sciencedirect.com/science/article/pii/S0016236120307857?via%3Dihub)

2. Ab Initio Insights Into the Formation Mechanisms of 55-Atom Pt-Based Core-Shell Nanoalloys. *The Journal of Physical Chemistry*, **2020**, 124, 1, 1158-1164. DOI: [10.1021/acs.jpcc.9b09561](https://pubs.acs.org/doi/abs/10.1021/acs.jpcc.9b09561)

3. Ab initio insights into the structural, energetic, electronic, and stability properties of mixed Ce<sub>*n*</sub>Zr<sub>15-*n*</sub>O<sub>30</sub> nanoclusters, *Physical Chemistry Chemical Physics*, **2019**, 21, 26637-26646. DOI: [10.1039/C9CP04762J](https://pubs.rsc.org/en/content/articlelanding/2019/CP/C9CP04762J)

## Cite us:

Please, if you employed the Quandarium package, cite us:

Correlation-Based Framework for Extraction of Insights from Quantum Chemistry Databases: Applications for Nanoclusters. *Journal of Chemical Information and Modeling*, **2021**, 61 (3), 1125-1135. DOI: [10.1021/acs.jcim.0c01267](https://doi.org/10.1021/acs.jcim.0c01267)

## Contact

Fell free to contact me by email: 

* johnatan.mucelini@usp.br
* johnatan.mucelini@gmail.com
