class BaseStrategy:

    def generate_signals(self, data):
        """
        Accepts a Polars DataFrame and must return
        a DataFrame with a 'signal' column.
        """
        raise NotImplementedError("Strategy must implement generate_signals")