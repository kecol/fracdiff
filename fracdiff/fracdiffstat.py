import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted

from fracdiff import Fracdiff
from fracdiff.stat import StatTester, StrictStatTester


class FracdiffStat(TransformerMixin, BaseEstimator):
    """
    Carry out fractional derivative with the minumum order
    with which the differentiation becomes stationary.

    Parameters
    ----------
    - window : int > 0 or None, default 10
        Number of observations to compute each element in the output.
    - mode : {"full", "valid"}, default "full"
        "full" (default) :
            Return elements where at least one coefficient is used.
            Shape of a transformed array is the same with the original array.
            At the beginning of a transformed array, boundary effects may be seen.
        "valid" :
            Return elements where all coefficients are used.
            Output size along axis 1 is `n_features - window_`.
            At the beginning of a time-series, boundary effects is not seen.
    - window_policy : {"fixed"}, default "fixed"
        If "fixed" :
            Fixed window method.
            Every term in the output is evaluated using `window_` observations.
            In other words, a fracdiff operator, which is a polynominal of a backshift
            operator, is truncated up to the `window_`-th term.
            The beginning `window_ - 1` elements in output are filled with `numpy.nan`.
        If "expanding" (not available yet) :
            Expanding window method.
            Every term in fracdiff time-series is evaluated using at least `window_`
            observations.
            The beginning `window_ - 1` elements in output are filled with `numpy.nan`.
    - stattest : {"ADF"}, default "ADF"
        Method of stationarity test.
    - pvalue : float, default 0.05
        Threshold of p-value to judge stationarity.
    - precision : float, default .01
        Precision for the order of differentiation.
    - upper : float, default 1.0
        Upper limit of the range to search the order.
    - lower : float, default 0.0
        Lower limit of the range to search the order.

    Attributes
    ----------
    - d_ : numpy.array, shape (n_features,)
        Minimum order of fractional differentiation
        that makes time-series stationary.

    Note
    ----
    If `upper`th differentiation of series is still non-stationary,
    order_ is set to `np.nan`.
    If `lower`th differentiation of series is already stationary,
    order_ is set to `lower`, but the true value may be smaller.

    Examples
    --------
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 4).cumsum(0)
    >>> f = FracdiffStat().fit(X)
    >>> f.d_
    array([0.140625 , 0.5078125, 0.3984375, 0.140625 ])
    >>> X = f.transform(X)
    """

    def __init__(
        self,
        window=10,
        mode="full",
        window_policy="fixed",
        stattest="ADF-KPSS",
        pvalue=0.05,
        precision=0.01,
        upper=1.0,
        lower=0.0,
    ):
        self.window = window
        self.mode = mode
        self.window_policy = window_policy
        self.stattest = stattest
        self.pvalue = pvalue
        self.precision = precision
        self.upper = upper
        self.lower = lower

    def fit(self, X, y=None):
        check_array(X)

        self.d_ = np.array([self._find_d(X[:, i]) for i in range(X.shape[1])])

        return self

    def transform(self, X, y=None) -> np.array:
        check_is_fitted(self, ["d_"])
        check_array(X)

        prototype = Fracdiff(0.5, window=self.window, mode=self.mode).fit_transform(X)
        out = np.empty_like(prototype[:, :0])

        for i in range(X.shape[1]):
            f = Fracdiff(self.d_[i], window=self.window, mode=self.mode)
            d = f.fit_transform(X[:, [i]])[-out.shape[0] :]
            out = np.concatenate((out, d), 1)

        return out

    def _is_stat(self, x) -> bool:
        """
        Parameters
        ----------
        x : array, shape (n,)

        Returns
        -------
        is_stat : bool
        """
        if self.stattest in ["ADF", "KPSS"]:
            return StatTester(method=self.stattest).is_stat(x, pvalue=self.pvalue)
        elif self.stattest == 'ADF-KPSS':
            return StrictStatTester().is_stat(x, pvalue=self.pvalue)
        raise Exception(f'Unknown stattest {self.stattest}')

    def _find_d(self, x) -> float:
        """
        Carry out binary search of minimum order of fractional
        differentiation to make the time-series stationary.

        Parameters
        ----------
        x : array, shape (n,)

        Returns
        -------
        d : float
        """

        def diff(d):
            return (
                Fracdiff(d, window=self.window, mode=self.mode)
                .fit_transform(x.reshape(-1, 1))
                .reshape(-1)
            )

        if not self._is_stat(diff(self.upper)):
            return np.nan
        if self._is_stat(diff(self.lower)):
            return self.lower

        upper, lower = self.upper, self.lower
        while upper - lower > self.precision:
            m = (upper + lower) / 2
            if self._is_stat(diff(m)):
                upper = m
            else:
                lower = m

        return upper
