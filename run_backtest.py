from data.loader import load_data
from strategies.ma_crossover import MovingAverageCrossStrategy
from backtester.engine import BacktestEngine


def main():

    symbol = "SPY"

    data = load_data(symbol)
    print("Data Loaded: {}".format(data))

    strategy = MovingAverageCrossStrategy(
        fast=20,
        slow=50
    )

    engine = BacktestEngine(data, strategy)

    portfolio = engine.run()

    initial_cash = 10000
    final_value = portfolio[-1]

    # get start and end date
    start_date = data.select("Date").min().item()
    end_date = data.select("Date").max().item()

    years = (end_date - start_date).days / 365

    cagr = (final_value / initial_cash) ** (1 / years) - 1

    print("Initial Value:", initial_cash)
    print("Final Value:", round(final_value, 2))
    print("Backtest Years:", round(years, 2))
    print("Annual Return (CAGR):", round(cagr * 100, 2), "%")


if __name__ == "__main__":
    main()