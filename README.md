# GOLDFISH
A Python library for **G**radient-based **O**ptimization, **L**arge-scale **D**esign **F**ramework for **I**sogeometric **SH**ells. This framework currently supports shape optimization for shell structures consisting of non-matching spline patches.

## Dependencies
1. [PENGoLINS](https://github.com/hanzhao2020/PENGoLINS) and its dependencies for structural analysis of isogeometric shells.
2. [CSDL](https://github.com/LSDOlab/csdl) and related packages: [csdl_om](https://github.com/LSDOlab/csdl_om), [python_csdl_backend](https://github.com/LSDOlab/python_csdl_backend), [ModOpt](https://github.com/LSDOlab/modopt), [array_manager
](https://github.com/anugrahjo/array_manager) for multidisciplinary optimization ([OpenMDAO](https://github.com/OpenMDAO/OpenMDAO) is also supported to perform optimization).