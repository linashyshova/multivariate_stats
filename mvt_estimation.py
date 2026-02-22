import numdifftools as nd
import numpy as np
import scipy as sc
import scipy.optimize


def MVT_MLE_approach2(X):
    n, p = X.shape
    # i n i t i a l i s a t i o n
    m = np.zeros(p)  # initial mu
    v = np.log(5)  # initial nu ( note : take log because we take the exp later !)
    S = np.eye(p)  # initial Sigma
    s_vech = S[np.tril_indices(S.shape[0])]  # p(p+1)/2 uniques params - lower matrix
    theta_initial = np.append(np.array(m), np.append(s_vech, v))
    # solver
    mle = sc.optimize.minimize(LL_mvt_appr, theta_initial, args=X, method="L-BFGS-B",
                               options={'disp': False, 'maxiter': 2500})
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
