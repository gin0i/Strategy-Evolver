import numpy as np
from numba import cuda
import time
import sys
import dask.multiprocessing
from dask import compute, delayed


MAVG = 0
SHOULD_BUY = 42
SHOULD_SELL = 666
OPEN = 1
CLOSE = 2
HIGH = 3
TIMESTAMP = 0
LOW = 4
VOLUME = 5

CURRENT = 0
STARTING = 1

BUYS = 2
SELLS = 3
CASH = 0
POSITION = 1

RSI_PERIOD = 1
MAVG_PERIOD = 0
MAVG_TYPE = 2
RSI_OVERSOLD = 3
RSI_OVERBOUGHT = 4
OSCL_TYPE = 5
SIGNAL_A = 0
SIGNAL_B = 1

RSI = 1
SMA = 0
EMA = 1
DEMA = 2
T3 = 3
CMO = 5


indicators_to_index = {
    'SMA': 0,
    'RSI': 1,
    'DEMA': 2,
    'T3': 3,
    'EMA': 4,
    'CMO': 5,
}

# Detach this in other file, non-kernel
# Common utils file
def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))
        return ret
    return wrap

# Detach this in other file, non-kernel
# GPU backtest utils file
def reset_buffer(buffers, name, shape):
    if name not in buffers or buffers[name].shape != shape:
        buffers[name] = np.ascontiguousarray(np.zeros(shape, dtype=np.float64))
    buffers[name].fill(0)

# Detach this in other file, non-kernel
# GPU backtest utils file
def prepare_buffers(buffers, data_length, strat_count, starting_cash, starting_position):
    reset_buffer(buffers, 'output_data', (strat_count, data_length, 8) )
    reset_buffer(buffers, 'strategies_parameters', (strat_count, 6) )
    reset_buffer(buffers, 'indicators_data', (strat_count, 2, data_length))
    buffers['cash_holders'] = np.array(strat_count * [[starting_cash, starting_cash]], dtype=np.float64)
    buffers['position_holders'] = np.array(strat_count * [[0, starting_position]], dtype=np.float64)

#@timing
# Detach this in other file, non-kernel
# GPU backtest utils file
def initialize_all_strategies(individuals, candle_data, indicators_data, strategies_parameters, backtest_initialization, name):
    def initialize(context):
        backtest_initialization(*context)

    init_contexts = []
    for strategy_id, indy in enumerate(individuals):
        init_contexts.append((strategy_id, indy, candle_data, indicators_data, strategies_parameters, name))
        initialize(init_contexts[-1])

# Detach this in other file, non-kernel
# GPU backtest utils file
def simulate_all_strategies(buffers, candle_data, individuals, starting_cash, starting_position, backtest_initialization, run_kernel, name):
    prepare_buffers(buffers, len(candle_data), len(individuals), starting_cash, starting_position)
    initialize_all_strategies(individuals, candle_data, buffers['indicators_data'], buffers['strategies_parameters'], backtest_initialization, name)
    comm_fee = 0.0025
    return run_kernel(buffers['strategies_parameters'], candle_data, buffers['indicators_data'], buffers['output_data'], len(candle_data), buffers['cash_holders'], buffers['position_holders'], comm_fee, np.int32(6), np.int32(4))


@cuda.jit(device=True)
def record_current_portfolio(cash_holder, position_holder, current_progress, output, close_price, timestamp):
    RSI, MAVG, OPEN, CLOSE, HIGH, TIMESTAMP, LOW, VOLUME, CURRENT, STARTING, BUYS, SELLS = 1, 0, 1, 2, 3, 0, 4, 5, 0, 1, 2, 3
    CASH, POSITION, RSI_PERIOD, MAVG_PERIOD, MAVG_TYPE, RSI_OVERSOLD, RSI_OVERBOUGHT = 0, 1, 1, 0, 2, 3, 4
    OUT_PRICE, OUT_TIMESTAMP = 4, 6

    output[current_progress][CASH] = cash_holder[CURRENT]
    output[current_progress][POSITION] = position_holder[CURRENT]
    output[current_progress][OUT_PRICE] = close_price
    output[current_progress][OUT_TIMESTAMP] = timestamp

@cuda.jit(device=True)
def buy_standalone(close_price, quantity, current_progress, current_ts, cash_holder, position_holder, fee, output):
    RSI, MAVG, OPEN, CLOSE, HIGH, TIMESTAMP, LOW, VOLUME, CURRENT, STARTING, BUYS, SELLS = 1, 0, 1, 2, 3, 0, 4, 5, 0, 1, 2, 3
    price = close_price * quantity
    cost = price * (1.0 + fee)
    if cost < cash_holder[CURRENT] and quantity >= 0.1:
        cash_holder[CURRENT] -= cost
        position_holder[CURRENT] += quantity
        output[current_progress][BUYS] = quantity
        return quantity
    return 0


@cuda.jit(device=True)
def sell_standalone(close_price, quantity, current_progress, current_ts, cash_holder, position_holder, fee, output):
    RSI, MAVG, OPEN, CLOSE, HIGH, TIMESTAMP, LOW, VOLUME, CURRENT, STARTING, BUYS, SELLS = 1, 0, 1, 2, 3, 0, 4, 5, 0, 1, 2, 3
    price = close_price * quantity
    revenue = price / (1.0 + fee)
    if quantity <= position_holder[CURRENT]:
        cash_holder[CURRENT] += revenue
        position_holder[CURRENT] -= quantity
        output[current_progress][SELLS] = quantity
        return quantity
    return 0

@cuda.jit
def run_strategy(candle_data, indicators, outputs, data_length, cash_holders, position_holders, comm_fee, strategies_parameters):
    RSI, MAVG, OPEN, CLOSE, HIGH, TIMESTAMP, LOW, VOLUME, CURRENT, STARTING, BUYS, SELLS = 1, 0, 1, 2, 3, 0, 4, 5, 0, 1, 2, 3

    i = cuda.threadIdx.x
    parameters = strategies_parameters[i]
    output = outputs[i]
    cash_holder = cash_holders[i]
    position_holder = position_holders[i]

    for current_progress in range(data_length):
        candle_handler(candle_data, current_progress, cash_holder, position_holder, indicators, comm_fee,
                       output, parameters, i)
        record_current_portfolio(cash_holder, position_holder, current_progress, output, candle_data[current_progress][CLOSE], candle_data[current_progress][TIMESTAMP])

    sell_standalone(candle_data[-1][CLOSE], position_holder[CURRENT], -1, candle_data[-1][TIMESTAMP], cash_holder,
                    position_holder, comm_fee, output)
    record_current_portfolio(cash_holder, position_holder, -1, output, candle_data[current_progress][CLOSE], candle_data[current_progress][TIMESTAMP])


@cuda.jit(device=True)
def candle_handler(candle_data, current_progress, cash_holder, position_holder, indicators, comm_fee, output, parameters, i):
    RSI, MAVG, OPEN, CLOSE, HIGH, TIMESTAMP, LOW, VOLUME, CURRENT, STARTING, BUYS, SELLS = 1, 0, 1, 2, 3, 0, 4, 5, 0, 1, 2, 3
    CASH, POSITION, RSI_PERIOD, MAVG_PERIOD, MAVG_TYPE, RSI_OVERSOLD, RSI_OVERBOUGHT = 0, 1, 1, 0, 2, 3, 4
    SIGNAL_A, SIGNAL_B = 0, 1
    SHOULD_BUY = 42
    SHOULD_SELL = 666
    SIGNAL_O = 7

    rsi_period = int(parameters[RSI_PERIOD])
    mavg_period = int(parameters[MAVG_PERIOD])
    wait_period = int(max(rsi_period, mavg_period))
    close_price = candle_data[current_progress][CLOSE]
    current_ts = candle_data[current_progress][TIMESTAMP]
    if current_progress < wait_period:
        return


    if indicators[i][SIGNAL_B][current_progress] == SHOULD_BUY and indicators[i][SIGNAL_A][current_progress] == SHOULD_BUY:
        buy_price = close_price * 1.001
        output[current_progress][SIGNAL_O] = 42
        qty = min(cash_holder[CURRENT], cash_holder[CURRENT])
        qty = cash_holder[CURRENT] / (buy_price * (1.003))
        if buy_standalone(buy_price, qty, current_progress, current_ts, cash_holder, position_holder, comm_fee, output) > 0:
            pass
    elif indicators[i][SIGNAL_B][current_progress] == SHOULD_SELL and indicators[i][SIGNAL_A][current_progress] == SHOULD_SELL:
        output[current_progress][SIGNAL_O] = 666
        good_sell_qty = min(position_holder[CURRENT], position_holder[CURRENT])
        sell_price = close_price / 1.001
        if sell_standalone(sell_price, good_sell_qty, current_progress, current_ts, cash_holder, position_holder, comm_fee, output) > 0:
            pass

# Detach in backtest GPU utils file
#@timing
def cuda_run_kernel(genomes, candle_data, indicators_data, output_data, data_length, cash_holders, positions_holders, comm_fee, candle_width, output_width):
    threadsperblock = len(genomes)
    blockspergrid = (len(genomes) + (threadsperblock - 1)) // threadsperblock

    run_strategy[blockspergrid, threadsperblock](candle_data, indicators_data, output_data, np.int32(data_length), cash_holders, positions_holders, np.float64(comm_fee), genomes)
    return output_data