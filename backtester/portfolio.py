class Portfolio:

    def __init__(self, initial_cash=10000):

        self.cash = initial_cash
        self.position = 0
        self.values = []

    def update(self, price, signal):

        if signal == 1 and self.position == 0:
            self.position = self.cash / price
            self.cash = 0

        elif signal == -1 and self.position > 0:
            self.cash = self.position * price
            self.position = 0

        value = self.cash + self.position * price

        self.values.append(value)