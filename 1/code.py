import numdifftools as nd
import numpy as np
import pandas as pd
import scipy as sc
import scipy.optimize
from scipy.stats import chi2, norm
import warnings


def MVT_MLE_approach2(X):
    n, p = X.shape
    # i n i t i a l i s a t i o n
    m = np.zeros(p)  # initial mu
    v = np.log(5)  # initial nu ( note : take log because we take the exp later !)
    S = np.eye(p)  # initial Sigma
    s_vech = S[np.tril_indices(S.shape[0])]  # p(p+1)/2 uniques params - lower matrix
    theta_initial = np.append(np.array(m), np.append(s_vech, v))
    # solver
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mle = sc.optimize.minimize(
            LL_mvt_appr,
            theta_initial,
            args=X,
            method="L-BFGS-B",
            options={'disp': False, 'maxiter': 2500}
        )
    converged = mle['success']
    Theta_tilde = mle.x
    # get the variance of theta_tilde
    Hessfun = nd.Hessian(LL_mvt_appr)
    H = Hessfun(Theta_tilde, X)
    Hinv = np.linalg.pinv(H)
    V_asy = n * Hinv  # asymptotic variance
    V_sam = Hinv  # sample variance

    # translate theta_tilde to theta (g (.) ) and apply delta   method
    Theta = g_fun_mvt(Theta_tilde, p).reshape(-1, 1)
    Jfun = nd.Jacobian(g_fun_mvt)
    D = Jfun(Theta_tilde, p)  # jacobian of g()
    VCV_asy = np.squeeze(D @ V_asy @ D.T)  # asy variance theta ( delta method )
    VCV_sam = np.squeeze(D @ V_sam @ D.T)  # sample variance theta (delta method )

    return Theta, np.squeeze(VCV_sam), np.squeeze(VCV_asy), converged


def g_fun_mvt(pars, p):
    m = pars[0:p]
    v = np.exp(pars[-1])
    # put scale elements in lower triangular  matrix  L
    a = pars[p:-1]
    L = np.zeros([p, p])
    idx = np.tril_indices(p)
    L[idx] = a
    S = L @ L.T  # calculate Sigma
    b = S[np.tril_indices(S.shape[0])]  # extract unique elements from Sigma
    Theta = np.append(np.array(m), np.append(b, v))
    return Theta


def LL_mvt_appr(pars, X):
    n, p = X.shape
    # g() : translate theta_tilde to theta
    m = np.reshape(pars[0:p], (-1, 1))
    v = np.exp(pars[-1])
    a = pars[p:-1]
    L = np.zeros([p, p])
    idx = np.tril_indices(p)
    L[idx] = a
    S = L @ L.T
    # fill into and evaluate the LL
    nl = negloglik_mvt(X, n, p, m, S, v)  # same as approach 1
    return np.array([nl])


def negloglik_mvt(X, n, p, m, S, v):
    Si = np.linalg.pinv(S)
    ll1 = n * (sc.special.gammaln((v + p) / 2) - sc.special.gammaln(v / 2) - (p / 2) * np.log(v * np.pi) - 0.5 * np.log(
        np.linalg.det(S)))
    # Use broadcasting and Einstein summation for efficient computation
    xd = X - np.ones([n, 1]) @ m.T  # n x p matrix
    ll2 = -0.5 * (v + p) * np.log(1 + np.einsum('ij,ij->i', xd @ Si, xd) / v).sum()
    nll = -(ll1 + ll2)  # we use a minimizer, so get the negative LL
    return nll

def generate_mvt(n, nu, mu, Sigma):
    p = mu.shape[0]

    # Generate multivariate normal
    N = np.random.randn(n, p)

    # Generate Chi-squared
    rng = np.random.default_rng()
    Chi = rng.chisquare(df=nu, size=n)

    # Combine to get multivariate T
    T_tilde = np.sqrt(nu) * N / np.sqrt(Chi)[:, None]

    # Apply transformation
    A = np.linalg.cholesky(Sigma)
    T = mu + T_tilde @ A.T

    return T


def reject_h0_nu(theta, omega, n, critical_value):
    # Create A (1x15) matrix full of zeros and only 1 at the last position.
    # 1 is on the last position because we are testing nu, the parameter that
    # is the last in our Theta
    A = np.zeros(15).T
    A[14] = 1

    # Create b, it's (1x1) because we have only one estimator to check
    b = 4

    # Difference (A @ theta - b) has size (1x1)
    diff = A @ theta - b

    # A @ omega @ A.T has also size (1) as well as its inverse
    AOA = A @ omega @ A.T
    AOA_inv = 1 / AOA

    # Calculate the test statistic
    statistic = n * diff.T * AOA_inv * diff

    # We compare test statistic with critical value of Chi-squared
    # with 1 degree of freedom and alpha=0.05.
    # We reject H0 if statistic is higher than critical value.
    return statistic > critical_value


def extract_estimated_results(p, theta_hat):
    # First p elements is mu
    mu = theta_hat[0:p]

    # Last element is nu
    nu = theta_hat[-1]

    # Everything in between is lower triangle of Sigma
    a = theta_hat[p:-1].flatten()
    sigma = np.zeros((p, p))
    idx = np.tril_indices(p)
    sigma[idx] = a

    # Combine lower triangle of Sigma to get full Sigma
    sigma = sigma + sigma.T - np.diag(np.diag(sigma))

    return mu, sigma, nu


def calculations_for_n(n,p, nu, mu, Sigma, critical_value):
    print(f"\n### Start for n = {n}")
    reject = 0

    mu_sum = 0
    sigma_sum = np.zeros((p, p))
    nu_sum = 0

    mu1_list = []
    se_mu1_list = []

    sigma32_list = []
    se_sigma32_list = []

    nu_list = []
    se_nu_list = []

    m = 1000
    for i in range(0, m):
        # Generate MVT with given parameters
        T = generate_mvt(n, nu, mu, Sigma)

        # Estimate parameters of MVT
        theta, VCV_sam, VCV_asy, _ = MVT_MLE_approach2(T)
        theta = np.asarray(theta).ravel()
        mu_hat, Sigma_hat, nu_hat = extract_estimated_results(p, theta)

        # Update sum of estimated params for calculating average
        mu_sum += mu_hat
        sigma_sum += Sigma_hat
        nu_sum += nu_hat

        # Update lists with estimated parameters and their variance
        se = np.sqrt(np.diag(VCV_sam))
        mu1_list.append(mu_hat[0])
        se_mu1_list.append(se[0])
        sigma32_list.append(Sigma_hat[2][1])
        se_sigma32_list.append(se[8])
        nu_list.append(nu_hat)
        se_nu_list.append(se[-1])

        # Test H0: nu=4
        if reject_h0_nu(theta, VCV_asy, n, critical_value):
            reject += 1

    # Calculate rejection rate for the H0
    rejection_rate = reject / m
    print(f"For {m} iteration for sample size of {n} the rejection rate is {rejection_rate}\n")

    # Calculate average per estimated parameter across all m runs
    mu_average = mu_sum / m
    sigma_average = sigma_sum / m
    nu_average = nu_sum / m

    print(f"Mu average: \n{mu_average}\n")
    print(f"Sigma average: \n{sigma_average}\n")
    print(f"Nu average: {nu_average}\n")

    # Calculate bias per estimated parameter
    mu1_bias = mu_average[0] - mu[0]
    sigma32_bias = sigma_average[2][1]-Sigma[2][1]
    nu_bias = nu_average - nu

    print(f"Mu_1 bias: {mu1_bias}\n")
    print(f"Sigma32 bias: {sigma32_bias}\n")
    print(f"Nu bias: {nu_bias}\n")

    emp_sd_mu1 = np.std(mu1_list, ddof=1)
    avg_se_mu1 = np.mean(se_mu1_list)
    emp_sd_sigma32 = np.std(sigma32_list, ddof=1)
    avg_se_sigma32 = np.mean(se_sigma32_list)
    emp_sd_nu = np.std(nu_list, ddof=1)
    avg_se_nu = np.mean(se_nu_list)

    print(f"Empirical SD mu: {emp_sd_mu1:.2f} Average SE: {avg_se_mu1:.2f} Ratio: {(avg_se_mu1 / emp_sd_mu1):.2f}")

    print(f"Empirical SD Sigma 32: {emp_sd_sigma32:.2f} Average SE: {avg_se_sigma32:.2f} Ratio: {(avg_se_sigma32 / emp_sd_sigma32):.2f}")

    print(f"Empirical SD nu: {emp_sd_nu:.2f} Average SE: {avg_se_nu:.2f} Ratio: {(avg_se_nu / emp_sd_nu):.2f}")


def defining_theta(theta, p):
    theta = np.asarray(theta).ravel()
    mu = theta[:p]
    nu = theta[-1]
    a = theta[p:-1]

    Sigma = np.zeros((p, p))
    idx = np.tril_indices(p)
    Sigma[idx] = a
    Sigma = Sigma + Sigma.T - np.diag(np.diag(Sigma))
    return mu, Sigma, nu


def report_mvt_results(theta_hat, VCV_asy, X, names=None):
    n, p = X.shape
    if names is None:
        names = [f"X{j+1}" for j in range(p)]

    se = np.sqrt(np.diag(VCV_asy))

    mu_hat, Sigma_hat, nu_hat = defining_theta(theta_hat, p)

    se_mu = se[:p]
    se_nu = se[-1]

    cov_implied = (nu_hat / (nu_hat - 2.0)) * Sigma_hat
    vol = np.sqrt(np.diag(cov_implied))

    corr_implied = cov_implied / np.outer(vol, vol)

    C = corr_implied.copy()
    np.fill_diagonal(C, -np.inf)
    i, j = np.unravel_index(np.argmax(C), C.shape)

    print("\nMVT MLE estimates (with asymptotic SE)")
    for nm, m, s in zip(names, mu_hat, se_mu):
        print(f"{nm:5s}: mu_hat = {m: .6f}   SE = {s: .6f}   t = {m/s: .3f}")

    print(f"\nnu_hat = {nu_hat:.6f}   SE = {se_nu:.6f}   t = {nu_hat/se_nu: .3f}")

    print("\nImplied daily volatility (from Cov = nu/(nu-2)*Sigma)")
    for nm, v in zip(names, vol):
        print(f"{nm:5s}: {v:.4f}%")

    print("\nImplied correlation matrix (rounded)")
    print(np.round(corr_implied, 3))

    print(f"\nHighest correlation pair: {names[i]} - {names[j]} = {corr_implied[i, j]:.3f}")

    se_sig_vech = se[p:-1]
    print("\nSigma_hat (scale matrix in MVT)")
    print(Sigma_hat)

    print("\nSEs for vech(Sigma) (lower triangle incl diag)")
    k = 0
    for r in range(p):
        for c in range(r + 1):
            print(f"Sigma[{r+1},{c+1}] = {Sigma_hat[r,c]: .6f}   SE = {se_sig_vech[k]: .6f}")
            k += 1

    return {
        "mu_hat": mu_hat,
        "Sigma_hat": Sigma_hat,
        "nu_hat": nu_hat,
        "VCV_asy": VCV_asy,
        "se": se,
        "cov_implied": cov_implied,
        "corr_implied": corr_implied,
        "vol": vol,
    }


def main():
    print("########### PART A ###################")
    # Parameters given in the assigment
    n_options = [50, 100, 500, 750]
    nu = 4
    mu = np.array([1, 2, -1, 3])
    Sigma = np.array([[2, 0, 1, 1],
                      [0, 3, 2, 1],
                      [1, 2, 5, 2],
                      [1, 1, 2, 6]])

    # We use rank of A matrix to find degrees of freedom for Chi squared
    # distribution. In our case df=1 as we are testing only one condition
    df_chi = 1
    alpha = 0.05
    critical_value = chi2.ppf(1 - alpha, df=1)

    for n in n_options:
        calculations_for_n(n=n, p=4, nu=nu, mu=mu, Sigma=Sigma, critical_value=critical_value)

    print("########### PART B ###################")
    data = pd.read_excel("data/techstocks.xlsx")
    df = data[['AAPL', 'MSFT', 'AMZN', 'GOOG', 'TSLA']].to_numpy()
    theta, VCV_sam, VCV_asy, _ = MVT_MLE_approach2(df)
    names = ["AAPL", "MSFT", "AMZN", "GOOG", "TSLA"]
    out = report_mvt_results(theta, VCV_sam, df, names=names)

    print("########### PART C ###################")
    # Removes column with dates
    clean_data = data[["AAPL", "MSFT", "AMZN", "GOOG", "TSLA"]]
    X = clean_data.to_numpy()

    # MLE of the data
    Theta, VCV_sam, VCV_asy, converged = MVT_MLE_approach2(X)
    print(f"Theta {Theta}")

    A_2 = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    b = 2
    n = len(data['AMZN'])
    test_stat_2 = (A_2 @ VCV_asy @ A_2.T) ** (-0.5) * np.sqrt(n) * (A_2 @ Theta - b)
    crit_val = norm.ppf(0.975)
    print("Test statistic c", test_stat_2)
    print("Critical value c", crit_val)

    print("########### PART D ###################")
    # Restriction matrix
    A = np.array([[1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    # Test statistic and critical value of chi squared distribution
    n = len(data['AMZN'])
    test_stat = n * (A @ Theta).T @ np.linalg.inv((A @ (VCV_asy) @ A.T)) @ (A @ Theta)
    crit_val_chi_square = chi2.ppf(0.95, 4)
    print("Test statistic d", test_stat)
    print("Critical value d", crit_val_chi_square)


if __name__ == "__main__":
    main()