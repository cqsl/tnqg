from netket.operator import AbstractOperator


class L2Loss(AbstractOperator):
    def __init__(self, hit, ham, T, n_times, window=1):
        super().__init__(hit)
        self._ham = ham
        self._T = T
        self._window = window
        self._n_times = n_times

    @property
    def ham(self):
        return self._ham

    @property
    def T(self):
        return self._T

    @property
    def window(self):
        return self._window

    @property
    def n_times(self):
        return self._n_times

    @property
    def dtype(self):
        return float

    def __eq__(self, o):
        if isinstance(o, L2Loss):
            return o.ham == self.ham
        return False

    @property
    def is_hermitian(self):
        return True

    def __repr__(self):
        return f"L2Loss(op={self.op})"
