import pandas as pd
import math


def get_tx_amounts(transactions):
    amounts = transactions['amount']
    buys = amounts[amounts > 0]
    sells = amounts[amounts < 0]
    return amounts, buys, sells


def get_tx_prices(transactions):
    amounts = transactions['amount']
    buy_prices = transactions['price'][amounts > 0]
    sell_prices = transactions['price'][amounts < 0]
    return transactions['price'], buy_prices, sell_prices


def total_trades(transactions):
    amounts, buys, sells, = get_tx_amounts(transactions)
    return min(len(buys), len(sells))


def calculate_trade_profit(ts_b, buy_price, ts_s, sell_price, amounts):
    if ts_b > ts_s:
        raise ValueError("error: sell transaction preceeds buy transaction: {} <-> {}".format(ts_s, ts_b))
    #if amounts[ts_b] != -amounts[ts_s]:
    #    raise ValueError("error:", "sell & buy amounts do not match: {} != {}".format(amounts[ts_b], amounts[ts_s]))
    buy_value = buy_price * amounts[ts_b]
    sell_value = sell_price * -amounts[ts_s]
    return sell_value - buy_value


def get_trades(transactions):
    amounts, buys_amounts, sells_amounts, = get_tx_amounts(transactions)
    prices, b_prices, s_prices, = get_tx_prices(transactions)
    trade_gains = []

    for (ts_b, buy_price), (ts_s, sell_price) in zip(b_prices.iteritems(), s_prices.iteritems()):
        gain = calculate_trade_profit(ts_b, buy_price, ts_s, sell_price, amounts)
        trade_gains.append( [ts_b, ts_s, gain] )

    return pd.DataFrame(trade_gains, columns=['buy_time', 'sell_time', 'gain'])


def get_losing_trades(trades):
    return trades.loc[trades['gain'] < 0]


def get_winning_trades(trades):
    return trades.loc[trades['gain'] > 0]


def get_avg_win_trade(trades):
    raw = get_winning_trades(trades)['gain'].mean()
    return 0 if math.isnan(raw) else raw


def get_avg_lose_trade(trades):
    raw = get_losing_trades(trades)['gain'].mean()
    return 0 if math.isnan(raw) else raw


def get_max_win_trade(trades):
    wins = get_winning_trades(trades)
    try:
        return wins.iloc[wins['gain'].idxmax()] if len(wins) > 0 else None
    except:
        return None


def get_max_lose_trade(trades):
    losses = get_losing_trades(trades)
    return losses.loc[losses['gain'].idxmin()] if len(losses) > 0 else None


def get_max_trades(trades):
    extrema = trades.loc[[trades['gain'].idxmin(), trades['gain'].idxmax()]]
    extrema.index = ['Min', 'Max']
    return extrema


def get_gross_loss(trades):
    losses = get_losing_trades(trades)
    return losses['gain'].sum()


def get_gross_profit(trades):
    wins = get_winning_trades(trades)
    return wins['gain'].sum()


def get_average_hold_time(trades):
    hold_times = []
    for idx, (b_time, s_time, gain) in trades.iterrows():
        hold_times.append(s_time - b_time)
    return pd.DataFrame(hold_times).mean()


def monthly_returns(daily_returns):
    return [(((x + 1) ** 30.44) - 1) * 100 for x in daily_returns]


def annual_returns(daily_returns):
    return [(((x + 1) ** 365) - 1) * 100 for x in daily_returns]


def get_max_drawdown(series):
    peak, mdd, trough = 0, 0, 0
    for i, value in series.iteritems():
        dif = series[peak] - value
        peak = i if dif < 0 else peak
        if mdd < dif:
            mdd = dif
            trough = i

    return mdd/series[peak], peak, trough


def total_commission(trades):
    return trades['commission'].sum()


def sharpe_ratio(daily_returns):
    return (365**0.5) * daily_returns.mean() / daily_returns.std()


def percent_profitable(perf):
    return 100 * (perf['cash'][-1] - perf['cash'][0]) / perf['cash'][0]


def net_profit_cash(perf):
    return perf['cash'][-1] - perf['cash'][0]


def net_profit_portfolio(perf):
    return perf['portfolio_value'][-1] - perf['portfolio_value'][0]