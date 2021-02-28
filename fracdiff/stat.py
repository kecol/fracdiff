import statsmodels.tsa.stattools as stattools


class StatTester:
    """
    Carry out stationarity test of time-series.
    Parameters
    ----------
    - method : {"ADF", "KPSS"}, default "ADF"
        If "ADF":
            Augmented Dickey-Fuller unit-root test.
        if "KPSS":
            Kwiatkowski–Phillips–Schmidt–Shin (KPSS) test.
            
    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    Stationary time-series:
    >>> x = np.random.randn(100)
    >>> tester = StatTester(method='ADF')
    >>> tester.pvalue(x)
    1.1655044784188669e-17
    >>> tester.is_stat(x)
    True
    Non-stationary time-series:
    >>> x = np.cumsum(x)
    >>> tester.pvalue(x)
    0.6020814791099098
    >>> tester.is_stat(x)
    False
    """

    def __init__(self, method="ADF"):
        self.method = method

    @property
    def null_hypothesis(self) -> str:
        if self.method == "ADF":
            return "unit-root"
        elif self.method == "KPSS":
            return "trend-stationary"
        raise Exception(f'Unknown method {self.method}')

    def pvalue(self, x) -> float:
        """
        Return p-value of the stationarity test.
        Parameters
        ----------
        - x : array, shape (n_samples,)
            Time-series to evaluate p-value.
        Returns
        -------
        pvalue : float
            p-value of the stationarity test.
        """
        if self.method == "ADF":
            _, pvalue, _, _, _, _ = stattools.adfuller(x)
        elif self.method == "KPSS":
            pvalue = stattools.kpss(x, nlags="auto")[1]
        else:
            raise Exception(f'Unknown method {self.method}')
        return pvalue

    def is_stat(self, x, pvalue=0.05) -> bool:
        """
        Return whether stationarity test implies stationarity.
        Parameters
        ----------
        - x : array, shape (n_samples,)
            Time-series to evaluate p-value.
        - pvalue : float, default 0.05
            Threshold of p-value.
        Note
        ----
        The name 'is_stat' may be misleading.
        Strictly speaking, `is_stat = True` implies that the null-hypothesis of
        the presence of a unit-root has been rejected (ADF test) or the null-hypothesis
        of the absence of a unit-root has not been rejected (KPSS test).
        Returns
        -------
        is_stat : bool
            True may imply the stationarity.
        """
        if self.null_hypothesis == "unit-root":
            return self.pvalue(x) < pvalue
        elif self.null_hypothesis == "trend-stationary":
            return self.pvalue(x) >= pvalue
        raise Exception(f"Unknown method {self.method}")
        
class StrictStatTester:
    
    def __init__(self):
        self.adf_test = StatTester('ADF')
        self.kpss_test = StatTester('KPSS')
    
    @property
    def null_hypothesis(self) -> str:
        return "unit-root and not trend-stationary"

    def is_stat(self, x, pvalue=0.05, verbose=1) -> bool:
        """
        Return whether stationarity test implies stationarity.
        Parameters
        ----------
        - x : array, shape (n_samples,)
            Time-series to evaluate p-value.
        - pvalue : float, default 0.05
            Threshold of p-value.
        Note
        ----
        The name 'is_stat' may be misleading.
        Strictly speaking, `is_stat = True` implies that the null-hypothesis of
        the presence of a unit-root has been rejected (ADF test) or the null-hypothesis
        of the absence of a unit-root has not been rejected (KPSS test).
        Returns
        -------
        is_stat : bool
            True may imply strict stationarity.
        """
        t1 = self.adf_test.is_stat(x, pvalue)
        t2 = self.kpss_test.is_stat(x, pvalue)
        
        # Case 1: Both tests conclude that the series is not stationary
        if not t1 and not t2:
            # the series is not stationary
            pass
        # Case 2: Both tests conclude that the series is stationary
        elif t1 and t2:
            # the series is stationary
            pass
        # Case 3: KPSS indicates stationarity and ADF indicates non-stationarity - The series is trend stationary.
        elif not t1 and t2:
            # Trend needs to be removed to make series strict stationary. The detrended series is checked for stationarity.
            if verbose > 0:
                print('Trend needs to be removed to make series strict stationary')
            pass
        # Case 4: KPSS indicates non-stationarity and ADF indicates stationarity
        elif t1 and not t2:
            # The series is difference stationary.
            # Differencing is to be used to make series stationary.
            # The differenced series is checked for stationarity.
            if verbose > 0:
                print('Series is difference stationary')
            pass
        return t1 and t2
