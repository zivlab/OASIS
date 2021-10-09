"""Extract neural activity from a fluorescence trace using OASIS,
an active set method for sparse nonnegative deconvolution
Created on Mon Apr 4 18:21:13 2016
@author: Johannes Friedrich
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, log, exp, fmax, fabs
from scipy.optimize import fminbound, minimize
from cpython cimport bool
from libcpp.vector cimport vector

ctypedef fused FLOAT:
    np.float32_t
    np.float_t

cdef cppclass Pool[T]:
    T v
    T w
    Py_ssize_t t
    Py_ssize_t l


@cython.cdivision(True)
def oasisAR1(np.ndarray[FLOAT, ndim=1] y, FLOAT g, FLOAT lam=0, FLOAT s_min=0):
    """ Infer the most likely discretized spike train underlying an AR(1) fluorescence trace

    Solves the sparse non-negative deconvolution problem
    min 1/2|c-y|^2 + lam |s|_1 subject to s_t = c_t-g c_{t-1} >=s_min or =0

    Parameters
    ----------
    y : array of float
        One dimensional array containing the fluorescence intensities with
        one entry per time-bin.
    g : float
        Parameter of the AR(1) process that models the fluorescence impulse response.
    lam : float, optional, default 0
        Sparsity penalty parameter lambda.
    s_min : float, optional, default 0
        Minimal non-zero activity within each bin (minimal 'spike size').

    Returns
    -------
    c : array of float
        The inferred denoised fluorescence signal at each time-bin.
    s : array of float
        Discretized deconvolved neural activity (spikes)

    References
    ----------
    * Friedrich J and Paninski L, NIPS 2016
    * Friedrich J, Zhou P, and Paninski L, PLOS Computational Biology 2017
    """

    cdef:
        Py_ssize_t i, j, k, t, T
        FLOAT tmp, lg
        np.ndarray[FLOAT, ndim = 1] c, s
        vector[Pool[FLOAT]] P
        Pool[FLOAT] newpool

    lg = log(g)
    T = len(y)
    # [value, weight, start time, length] of pool
    newpool.v, newpool.w, newpool.t, newpool.l = y[0] - lam * (1 - g), 1, 0, 1
    P.push_back(newpool)
    i = 0  # index of last pool
    t = 1  # number of time points added = index of next data point
    while t < T:
        # add next data point as pool
        newpool.v = y[t] - lam * (1 if t == T - 1 else (1 - g))
        newpool.w, newpool.t, newpool.l = 1, t, 1
        P.push_back(newpool)
        t += 1
        i += 1
        while (i > 0 and  # backtrack until violations fixed
               (P[i-1].v / P[i-1].w * exp(lg*P[i-1].l) + s_min > P[i].v / P[i].w)):
            i -= 1
            # merge two pools
            P[i].v += P[i+1].v * exp(lg*P[i].l)
            P[i].w += P[i+1].w * exp(lg*2*P[i].l)
            P[i].l += P[i+1].l
            P.pop_back()
    # construct c
    c = np.empty(T, dtype=y.dtype)
    for j in range(i + 1):
        tmp = P[j].v / P[j].w
        if (j == 0 and tmp < 0) or (j > 0 and tmp < s_min):
            tmp = 0
        for k in range(P[j].l):
            c[k + P[j].t] = tmp
            tmp *= g
    # construct s
    s = c.copy()
    s[0] = 0
    s[1:] -= g * c[:-1]
    return c, s


def _oasis1strun(np.ndarray[FLOAT, ndim=1] y, FLOAT g, np.ndarray[FLOAT, ndim=1] c):

    cdef:
        Py_ssize_t i, j, k, t, T
        FLOAT tmp, lg
        vector[Pool[FLOAT]] P
        Pool[FLOAT] newpool

    lg = log(g)
    T = len(y)
    # [value, weight, start time, length] of pool
    newpool.v, newpool.w, newpool.t, newpool.l = y[0], 1, 0, 1
    P.push_back(newpool)
    i = 0  # index of last pool
    t = 1  # number of time points added = index of next data point
    while t < T:
        # add next data point as pool
        newpool.v, newpool.w, newpool.t, newpool.l = y[t], 1, t, 1
        P.push_back(newpool)
        t += 1
        i += 1
        while (i > 0 and  # backtrack until violations fixed
                (P[i-1].v / P[i-1].w * exp(lg*P[i-1].l) > P[i].v / P[i].w)):
            i -= 1
            # merge two pools
            P[i].v += P[i+1].v * exp(lg*P[i].l)
            P[i].w += P[i+1].w * exp(lg*2*P[i].l)
            P[i].l += P[i+1].l
            P.pop_back()
    # construct c
    c = np.empty(T, dtype=y.dtype)
    for j in range(i + 1):
        tmp = fmax(P[j].v, 0) / P[j].w
        for k in range(P[j].l):
            c[k + P[j].t] = tmp
            tmp *= g
    return c, P


def _oasis(vector[Pool[FLOAT]] P, FLOAT g, np.ndarray[FLOAT, ndim=1] c):

    cdef:
        Py_ssize_t i, j, k
        FLOAT tmp, lg

    lg = log(g)
    i = 0
    while i < P.size() - 1:
        i += 1
        while (i > 0 and  # backtrack until violations fixed
                (P[i-1].v / P[i-1].w * exp(lg*P[i-1].l) > P[i].v / P[i].w)):
            i -= 1
            # merge two pools
            P[i].v += P[i+1].v * exp(lg*P[i].l)
            P[i].w += P[i+1].w * exp(lg*2*P[i].l)
            P[i].l += P[i+1].l
            P.erase(P.begin() + i + 1)
    # construct c
    c = np.empty(P[P.size() - 1].t + P[P.size() - 1].l, dtype=y.dtype)
    for j in range(i + 1):
        tmp = fmax(P[j].v, 0) / P[j].w
        for k in range(P[j].l):
            c[k + P[j].t] = tmp
            tmp *= g
    return c, P


@cython.cdivision(True)
def constrained_oasisAR1(np.ndarray[FLOAT, ndim=1] y, FLOAT g, FLOAT sn,
                         bool optimize_b=False, bool b_nonneg=True, int optimize_g=0,
                         int decimate=1, int max_iter=5, int penalty=1):
    """ Infer the most likely discretized spike train underlying an AR(1) fluorescence trace

    Solves the noise constrained sparse non-negative deconvolution problem
    min |s|_1 subject to |c-y|^2 = sn^2 T and s_t = c_t-g c_{t-1} >= 0

    Parameters
    ----------
    y : array of float
        One dimensional array containing the fluorescence intensities (with baseline
        already subtracted, if known, see optimize_b) with one entry per time-bin.
    g : float
        Parameter of the AR(1) process that models the fluorescence impulse response.
    sn : float
        Standard deviation of the noise distribution.
    optimize_b : bool, optional, default False
        Optimize baseline if True else it is set to 0, see y.
    b_nonneg: bool, optional, default True
        Enforce strictly non-negative baseline if True.
    optimize_g : int, optional, default 0
        Number of large, isolated events to consider for optimizing g.
        No optimization if optimize_g=0.
    decimate : int, optional, default 1
        Decimation factor for estimating hyper-parameters faster on decimated data.
    max_iter : int, optional, default 5
        Maximal number of iterations.
    penalty : int, optional, default 1
        Sparsity penalty. 1: min |s|_1  0: min |s|_0

    Returns
    -------
    c : array of float
        The inferred denoised fluorescence signal at each time-bin.
    s : array of float
        Discretized deconvolved neural activity (spikes).
    b : float
        Fluorescence baseline value.
    g : float
        Parameter of the AR(1) process that models the fluorescence impulse response.
    lam : float
        Sparsity penalty parameter lambda of dual problem.

    References
    ----------
    * Friedrich J and Paninski L, NIPS 2016
    * Friedrich J, Zhou P, and Paninski L, PLOS Computational Biology 2017
    """

    cdef:
        Py_ssize_t i, j, k, t, l
        unsigned int ma, count, T
        FLOAT thresh, v, w, RSS, aa, bb, cc, lam, dlam, b, db, dphi, lg
        bool g_converged
        np.ndarray[FLOAT, ndim = 1] c, res, tmp, fluor, h
        np.ndarray[long, ndim = 1] ff, ll
        vector[Pool[FLOAT]] P
        Pool[FLOAT] newpool

    lg = log(g)
    T = len(y)
    thresh = sn * sn * T
    if decimate > 1:  # parameter changes due to downsampling
        fluor = y.copy()
        y = y.reshape(-1, decimate).mean(1)
        lg *= decimate
        g = exp(lg)
        thresh = thresh / decimate / decimate
        T = len(y)
    # explicit kernel, useful for constructing solution 
    h = np.exp(lg * np.arange(T, dtype=y.dtype))  
    c = np.empty(T, dtype=y.dtype)
    lam = 0


    g_converged = False
    count = 0
    if not optimize_b:  # don't optimize b, just the dual variable lambda and g if optimize_g>0
        c, P = _oasis1strun(y, g, c)
        tmp = np.empty(len(c), dtype=y.dtype)
        res = y - c
        RSS = (res).dot(res)
        b = 0
        # until noise constraint is tight or spike train is empty
        while RSS < thresh * (1 - 1e-4) and c.sum() > 1e-9:
            # update lam
            for i in range(P.size()):
                if i == P.size() - 1:  # for |s|_1 instead |c|_1 sparsity
                    # faster than tmp[P[i].t:P[i].t + P[i].l] = 1 / P[i].w * h[:P[i].l]
                    aa = 1 / P[i].w
                    for j in range(P[i].l):
                        tmp[P[i].t + j] = aa
                        aa *= g
                else:
                    aa = (1 - exp(lg*P[i].l)) / P[i].w
                    for j in range(P[i].l):
                        tmp[P[i].t + j] = aa
                        aa *= g
            aa = tmp.dot(tmp)
            bb = res.dot(tmp)
            cc = RSS - thresh
            dlam = (-bb + sqrt(bb * bb - aa * cc)) / aa
            lam += dlam
            for i in range(P.size() - 1):  # perform shift
                P[i].v -= dlam * (1 - exp(lg*P[i].l))
            P[P.size() - 1].v -= dlam  # correct last pool; |s|_1 instead |c|_1
            c, P = _oasis(P, g, c)

            # update g
            if optimize_g and count < max_iter - 1 and (not g_converged):
                ma = max([P[i].l for i in range(P.size())])
                idx = np.argsort([P[i].v for i in range(P.size())])
                Pt = [P[i].t for i in idx[-optimize_g:]]
                Pl = [P[i].l for i in idx[-optimize_g:]]

                def bar(y, g, Pt, Pl):
                    lg = log(g)
                    h = np.exp(lg * np.arange(ma, dtype=y.dtype))

                    def foo(y, t, l, q, g, lg, lam=lam):
                        yy = y[t:t + l]
                        if t + l == T:  # |s|_1 instead |c|_1
                            tmp = ((q.dot(yy) - lam) * (1 - g * g) /
                                   (1 - exp(lg*2 * l))) * q - yy
                        else:
                            tmp = ((q.dot(yy) - lam * (1 - exp(lg*l))) * (1 - g * g) /
                                   (1 - exp(lg*2 * l))) * q - yy
                        return tmp.dot(tmp)
                    return sum([foo(y, Pt[i], Pl[i], h[:Pl[i]], g, lg)
                                for i in range(optimize_g)])

                def baz(y, Pt, Pl):
                    # minimizes residual
                    return fminbound(lambda x: bar(y, x, Pt, Pl), 0, 1, xtol=1e-4, maxfun=50)
                aa = baz(y, Pt, Pl)
                if abs(aa - g) < 1e-4:
                    g_converged = True
                g = aa
                lg = log(g)
                # explicit kernel, useful for constructing c
                h = np.exp(lg * np.arange(T, dtype=y.dtype))
                for i in range(P.size()):
                    q = h[:P[i].l]
                    P[i].v = q.dot(y[P[i].t:P[i].t + P[i].l]) - lam * (1 - exp(lg*P[i].l))
                    P[i].w = q.dot(q)
                P[P.size() - 1].v -= lam * exp(lg*P[P.size() - 1].l)  # |s|_1 instead |c|_1
                c, P = _oasis(P, g, c)
            # calc RSS
            res = y - c
            RSS = res.dot(res)

    else:  # optimize b and dependent on optimize_g g too
        b = np.percentile(y, 15)  # initial estimate of baseline
        if b_nonneg:
            b = fmax(b, 0)
        c, P = _oasis1strun(y - b, g, c)
        # update b and lam
        db = fmax(np.mean(y - c), 0 if b_nonneg else -np.inf) - b
        b += db
        lam -= db / (1 - g)
        # correct last pool
        i = P.size() - 1
        P[i].v -= lam * exp(lg*P[i].l)  # |s|_1 instead |c|_1
        c[P[i].t:P[i].t + P[i].l] = fmax(0, P[i].v) / P[i].w * h[:P[i].l]
        # calc RSS
        res = y - b - c
        RSS = res.dot(res)
        tmp = np.empty(len(c), dtype=y.dtype)
        # until noise constraint is tight or spike train is empty or max_iter reached
        while fabs(RSS - thresh) > thresh * 1e-4 and c.sum() > 1e-9 and count < max_iter:
            count += 1
            # update lam and b
            # calc total shift dphi due to contribution of baseline and lambda
            for i in range(P.size()):
                if i == P.size() - 1:  # for |s|_1 instead |c|_1 sparsity
                    aa = 1 / P[i].w
                    for j in range(P[i].l):
                        tmp[P[i].t + j] = aa
                        aa *= g
                else:
                    aa = (1 - exp(lg*P[i].l)) / P[i].w
                    for j in range(P[i].l):
                        tmp[P[i].t + j] = aa
                        aa *= g
            tmp -= 1. / T / (1 - g) * np.sum([(1 - exp(lg*P[i].l)) ** 2 / P[i].w
                                              for i in range(P.size())])
            aa = tmp.dot(tmp)
            bb = res.dot(tmp)
            cc = RSS - thresh
            if bb * bb - aa * cc > 0:
                dphi = (-bb + sqrt(bb * bb - aa * cc)) / aa
            else:
                dphi = -bb / aa
            if b_nonneg:
                dphi = fmax(dphi, -b / (1 - g))
            b += dphi * (1 - g)
            for i in range(P.size()):  # perform shift
                P[i].v -= dphi * (1 - exp(lg*P[i].l))
            c, P = _oasis(P, g, c)
            # update b and lam
            db = fmax(np.mean(y - c), 0 if b_nonneg else -np.inf) - b
            b += db
            dlam = -db / (1 - g)
            lam += dlam
            # correct last pool
            i = P.size() - 1
            P[i].v -= dlam * exp(lg*P[i].l)  # |s|_1 instead |c|_1
            c[P[i].t:P[i].t + P[i].l] = fmax(0, P[i].v) / P[i].w * h[:P[i].l]

            # update g and b
            if optimize_g and count < max_iter - 1 and (not g_converged):
                ma = max([P[i].l for i in range(P.size())])
                idx = np.argsort([P[i].v for i in range(P.size())])
                Pt = [P[i].t for i in idx[-optimize_g:]]
                Pl = [P[i].l for i in idx[-optimize_g:]]

                def bar(y, opt, Pt, Pl):
                    b, g = opt
                    lg = log(g)
                    h = np.exp(lg * np.arange(ma, dtype=y.dtype))

                    def foo(y, t, l, q, b, g, lg, lam=lam):
                        yy = y[t:t + l] - b
                        if t + l == T:  # |s|_1 instead |c|_1
                            tmp = ((q.dot(yy) - lam) * (1 - g * g) /
                                   (1 - exp(lg*2 * l))) * q - yy
                        else:
                            tmp = ((q.dot(yy) - lam * (1 - exp(lg*l))) * (1 - g * g) /
                                   (1 - exp(lg*2 * l))) * q - yy
                        return tmp.dot(tmp)
                    return sum([foo(y, Pt[i], Pl[i], h[:Pl[i]], b, g, lg)
                                for i in range(P.size() if P.size() < optimize_g else optimize_g)])

                def baz(y, Pt, Pl):
                    return minimize(lambda x: bar(y, x, Pt, Pl), (b, g),
                                    bounds=((0 if b_nonneg else None, None), (.001, .999)),
                                    method='L-BFGS-B',
                                    options={'gtol': 1e-04, 'maxiter': 3, 'ftol': 1e-05})
                result = baz(y, Pt, Pl)
                if fabs(result['x'][1] - g) < 1e-3:
                    g_converged = True
                b, g = result['x']
                lg = log(g)
                # explicit kernel, useful for constructing c
                h = np.exp(lg * np.arange(T, dtype=y.dtype))
                for i in range(P.size()):
                    q = h[:P[i].l]
                    P[i].v = q.dot(y[P[i].t:P[i].t + P[i].l]) - \
                        (b / (1 - g) + lam) * (1 - exp(lg*P[i].l))
                    P[i].w = q.dot(q)
                P[P.size() - 1].v -= lam * exp(lg*P[P.size() - 1].l)  # |s|_1 instead |c|_1
                c, P = _oasis(P, g, c)
                # update b and lam
                db = fmax(np.mean(y - c), 0 if b_nonneg else -np.inf) - b
                b += db
                dlam = -db / (1 - g)
                lam += dlam
                # correct last pool
                i = P.size() - 1
                P[i].v -= dlam * exp(lg*P[i].l)  # |s|_1 instead |c|_1
                c[P[i].t:P[i].t + P[i].l] = fmax(0, P[i].v) / P[i].w * h[:P[i].l]

            # calc RSS
            res = y - c - b
            RSS = res.dot(res)

    if decimate > 1:  # deal with full data
        y = fluor
        lam *= (1 - g)
        lg /= decimate
        g = exp(lg)
        lam /= (1 - g)
        thresh = thresh * decimate * decimate
        T = len(fluor)
        # warm-start active set
        ff = np.ravel([P[i].t * decimate + np.arange(-decimate, 3 * decimate / 2)
                       for i in range(P.size())])  # this window size seems necessary and sufficient
        ff = np.unique(ff[(ff >= 0) * (ff < T)])
        ll = np.append(ff[1:] - ff[:-1], T - ff[-1])
        h = np.exp(log(g) * np.arange(T, dtype=y.dtype))
        P.resize(0)
        for i in range(len(ff)):
            q = h[:ll[i]]
            newpool.v = q.dot(fluor[ff[i]:ff[i] + ll[i]]) - \
                (b / (1 - g) + lam) * (1 - exp(lg*ll[i]))
            newpool.w = q.dot(q)
            newpool.t = ff[i]
            newpool.l = ll[i]
            P.push_back(newpool)
        P[P.size() - 1].v -= lam * exp(lg*P[P.size() - 1].l)  # |s|_1 instead |c|_1
        c = np.empty(T, dtype=y.dtype)

        c, P = _oasis(P, g, c)

    if penalty == 0:  # get (locally optimal) L0 solution
        lls = [(P[i+1].v / P[i+1].w - P[i].v / P[i].w * exp(lg*P[i].l))
               for i in range(P.size() - 1)]
        pos = [P[i+1].t for i in np.argsort(lls)[::-1]]
        y = y - b
        res = -y
        RSS = y.dot(y)
        c = np.zeros_like(y)
        P.resize(0)
        newpool.v, newpool.w, newpool.t, newpool.l = 0, 1, 0, len(y)
        P.push_back(newpool)
        for p in pos:
            i = 0
            while P[i].t + P[i].l <= p:
                i += 1
            # split current pool at pos
            j, k = P[i].t, P[i].l
            q = h[:j - p + k]
            newpool.v = q.dot(y[p:j + k])
            newpool.w, newpool.t, newpool.l = q.dot(q), p, j - p + k
            P.insert(P.begin() + i + 1, newpool)
            q = h[:p - j]
            P[i].v, P[i].w, P[i].t, P[i].l = q.dot(y[j:p]), q.dot(q), j, p - j
            for t in [i, i + 1]:
                c[P[t].t:P[t].t + P[t].l] = fmax(0, P[t].v) / P[t].w * h[:P[t].l]
            # calc RSS
            RSS -= res[j:j + k].dot(res[j:j + k])
            res[P[i].t:j + k] = c[P[i].t:j + k] - y[P[i].t:j + k]
            RSS += res[P[i].t:j + k].dot(res[P[i].t:j + k])
            if RSS < thresh:
                break
    # construct s
    s = c.copy()
    s[0] = 0
    s[1:] -= g * c[:-1]
    return c, s, b, g, lam