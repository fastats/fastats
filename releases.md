
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
- OLS: f_statistic
- drop_missing : helper function analogous to statsmodels missing='drop' mechanism which allows the user to evict 
features and observations where one or more data points is NaN such that OLS may then be performed on dense / complete 
data.

#### Bug fixes


#### Enhancements
