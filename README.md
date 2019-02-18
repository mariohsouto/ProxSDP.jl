# ProxSDP

| **Build Status** |
|:-----------------:|
| [![Build Status][build-img]][build-url] [![Codecov branch][codecov-img]][codecov-url] |

[build-img]: https://travis-ci.org/mariohsouto/ProxSDP.jl.svg?branch=master
[build-url]: https://travis-ci.org/mariohsouto/ProxSDP.jl
[codecov-img]: http://codecov.io/github/mariohsouto/ProxSDP.jl/coverage.svg?branch=master
[codecov-url]: http://codecov.io/github/mariohsouto/ProxSDP.jl?branch=master

ProxSDP is a semidefinite programming ([SDP](https://en.wikipedia.org/wiki/Semidefinite_programming)) solver based on the paper ["Exploiting Low-Rank Structure in Semidefinite Programming by Approximate Operator Splitting"](https://arxiv.org/abs/1810.05231). ProxSDP solves general SDP problems by means of a first order proximal algorithm based on the [primal-dual hybrid gradient](http://www.cmapx.polytechnique.fr/preprint/repository/685.pdf), also known as Chambolle-Pock method. The main advantage of ProxSDP over other state-of-the-art solvers is the ability of exploit the low-rank property inherent to several SDP problems.

### Overview of problems ProxSDP can solve

* Any semidefinite programming problem in [standard form](http://web.stanford.edu/~boyd/papers/pdf/semidef_prog.pdf);
* Semidefinite relaxations of nonconvex problems, e.g. [max-cut](http://www-math.mit.edu/~goemans/PAPERS/maxcut-jacm.pdf), [binary MIMO](https://arxiv.org/pdf/cs/0606083.pdf), [optimal power flow](http://authorstest.library.caltech.edu/141/1/TPS_OPF_2_tech.pdf), [sensor localization](https://web.stanford.edu/~boyd/papers/pdf/sensor_selection.pdf);
* Nuclear norm minimization problems, e.g. [matrix completion](https://statweb.stanford.edu/~candes/papers/MatrixCompletion.pdf).

### Referencing

The first version of the paper can be found [here](https://arxiv.org/abs/1810.05231).

```
@article{souto2018exploiting,
  title={Exploiting Low-Rank Structure in Semidefinite Programming by Approximate Operator Splitting},
  author={Souto, Mario and Garcia, Joaquim D and Veiga, {\'A}lvaro},
  journal={arXiv preprint arXiv:1810.05231},
  year={2018}
}
```

### TODO

- Add support for exponential and power cones
