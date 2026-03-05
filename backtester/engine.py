from backtester.portfolio import Portfolio


class BacktestEngine:

    def __init__(self, data, strategy):

        self.strategy = strategy
        self.data = strategy.generate_signals(data)

        self.portfolio = Portfolio()

    def run(self):

        for row in self.data.iter_rows(named=True):

            price = row["Close"]
            signal = row["signal"]

            self.portfolio.update(price, signal)

        return self.portfolio.values