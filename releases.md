# 2019.2

Numpy requirements updated to be >= 1.14

#### New features

#### Bug fixes

#### Enhancements

- Removed support for numpy < 1.14, and fixed formatting from earlier numpy versions

# 2019.1

Numba requirements changed to be >= 0.41 due to Windows segfaults

#### New features

#### Bug fixes

- Fixed the numba lowering error from the scaling check for `ddof not in (0, 1)`

#### Enhancements

- Now testing all PRs on python 3.5, 3.6 and 3.7 on Linux and Windows, and 3.7-dev on linux
- Travis CI using Xenial instead of Trusty to get python 3.7 support

# 2018.1

#### New features

- Exponentially Weighted Moving Average

#### Bug fixes


#### Enhancements

# 2017.1

#### New features

- newton_raphson : Root finding using the Newton-Raphson iteration method
- erf : error function using Abramowitz and Stegun method (maximum error: 1.5e-7) 
- correlation : Spearman and Pearson correlation coefficient functions
- pca : Principal Component Analysis, returning the transformed data, not eigenvalues
or eigenvectors.
- Added Windows CI builds: #23
- LU Decomposition
- binary_search : Root finding using the bisection method
- Documentation added using sphinx + numpydoc.
- OLS: f_statistic
- QR: classical and modified Gram Schmidt methods
- Matrix inverse using adjoint method
- Matrix determinant
- Matrix minor (sub-matrix with one row and one column eliminated)
- Scaling functions (standard, min_max, rank, demean, shrink off diagonals)
- Lasso regression for orthonormal covariates (features)
- drop_missing : helper function analogous to statsmodels missing='drop' mechanism which allows the user to evict 
features and observations where one or more data points is not finite such that OLS may then be performed on dense / 
complete data.
- EWMA functions for 1D and 2D data sets (dense data / no NaNs)

#### Bug fixes


#### Enhancements
