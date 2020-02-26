"""
Implements the SELVAR (Selective auto-regressive model) algorithm.

Based on an implementation that is originally due to Gherardo Varando
(gherardovarando).
"""
from scipy.stats import chi2
from .utils import common_pre_post_processing
###
# compile selvarF.f with:
#    f2py -llapack -c -m selvarF selvarF.f
###
try:
    from .selvarF import slvar, gtstat, gtcoef
except ImportError:
    slvar = None
    gtstat = None
    gtcoef = None


@common_pre_post_processing
def selvar(data,
           maxlags=1,
           batchsize=-1,
           mxitr=-1,
           trace=0,
           ):
    """
    SELVAR (Selective auto-regressive model).

    Parameters
    ----------
    data : ndarray
        T (timepoints) x N (variables) input data

    maxlags : int
        Maximum number of lags to include in the model.
        If maxlags < 0 then the maximum lag will be iteratively
        adjusted for each variable until no decrease in PRSS.

    batchsize : int
        Number of consecutive time points to use in each training batch.
        If batchsize < 0 then batchsize is set to the maximum available
        time boints given maxlags.

    mxitr : int
        Maximum number of iterations (each variable) for the
        hill-climbing search. If mxitr < 0 then the
        search will stop only when no decrease in PRSS is possible.

    trace : int
        If positive messages will be printed out during the search.

    Arguments for the common pre-processing steps of the data and the common
    post-processing steps of the scores are documented in
    utils.common_pre_post_processing

    Returns
    ----------
    scores : ndarray
        Array with scores for each link i -> j
    pvalues : ndarray
        Array with naively adjuste p-values of likelihood-ratio test,
        should not be used for testing presence of edges
    lags : ndarray
        Array with estimated positive lags, 0 if no link
    """

    if slvar is None:
        raise RuntimeError("selvarF.f is not yet compiled")

    scores, lags, info = slvar(data, bs=int(batchsize), ml=int(maxlags),
                               mxitr=int(mxitr), trc=int(trace))
    stat, df = gtstat(data, a=lags, bs=int(batchsize),
                      ml=int(maxlags), job="LR")

    pvalues = 1 - chi2.cdf(stat, df[:, 0] - df[:, 1])
    pvalues = pvalues * (data.shape[1] - 1) * data.shape[1]
    pvalues[pvalues > 1] = 1

    return scores, pvalues, lags
