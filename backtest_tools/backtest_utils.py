from datetime import datetime
import pandas as pd
import numpy as np
import json
import requests
import time


def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))

        return ret
    return wrap


def convert_candles_step(minute_candles, step_size):
    new_index = pd.DatetimeIndex(minute_candles.index[step_size-1::step_size])
    result = pd.DataFrame(index=new_index, columns=minute_candles.columns)
    for i in range(0, len(minute_candles), step_size):
        extract = minute_candles[i:i+step_size]
        first_timestamp, latest_timestamp = extract.index[0], extract.index[-1]
        result.at[pd.DatetimeIndex([latest_timestamp]), 'high'] = max(extract['high'])
        result.at[pd.DatetimeIndex([latest_timestamp]), 'open'] = extract.loc[first_timestamp]['open']
        result.at[pd.DatetimeIndex([latest_timestamp]), 'close'] = extract.loc[latest_timestamp]['close']
        result.at[pd.DatetimeIndex([latest_timestamp]), 'low'] = min(extract['low'])
        result.at[pd.DatetimeIndex([latest_timestamp]), 'volume'] = sum(extract['volume'])
    return result


def load_candles_pandas(path):
    candles = pd.read_json(path)
    candles.index = candles[0]
    del candles[0]
    candles.index = [datetime.fromtimestamp(x / 1000) for x in candles.index]
    candles['open'] = candles[1]
    candles['close'] = candles[2]
    candles['high'] = candles[3]
    candles['low'] = candles[4]
    candles['volume'] = candles[5]
    del candles[1]
    del candles[2]
    del candles[3]
    del candles[4]
    del candles[5]
    return candles[:10000]


def hitbtc_candles_to_numpy(path):
    pdf = pd.read_json(path)
    pdf['ts'] = pdf.index
    pdf['ts'] = pdf['ts'].apply(lambda ts: ts.timestamp())
    purged = pdf[['ts', 'open', 'close', 'min', 'max', 'volume']]
    purged = purged.rename(columns={"min": "low", "max": "high"})
    return np.ascontiguousarray(purged.values.astype(np.float64))

def cryptodatadownload_to_numpy(path):
    pdf = pd.read_csv(path)
    pdf['Date'] = pdf['Date'].apply(lambda ts: datetime.strptime(ts, '%Y-%m-%d %H-%p').timestamp())
    purged = pdf[['Date', 'Open', 'Close', 'Low', 'High', 'Volume BTC']]
    return np.ascontiguousarray(purged.values[::-1].astype(np.float64))


def catalyst_candles_to_numpy(path):
    pdf = pd.read_json(path)
    pdf['ts'] = pdf.index
    pdf['ts'] = pdf['ts'].apply(lambda ts: ts.timestamp())
    purged = pdf[['ts', 'open', 'close', 'low', 'high', 'volume']]
    return purged.values


def heikin_ashi(candles):
    result = np.zeros(candles.shape)
    for i, (ts, open, close, high, low, volume) in enumerate(candles):
        h_close = (open + close + low + high)/4
        h_open = open if i == 0 else (result[i-1][1] + result[i-1][2])/2
        h_high = max(high, h_open, h_close)
        h_low = min(low, h_open, h_close)
        result[i] = [ts, h_open, h_close, h_high, h_low, volume]
    return np.ascontiguousarray(result)


def save_cexio_month_candles(instrument, year, month, savepath):
    for day in range(1, 32):
        time.sleep(3.0)
        url = 'https://cex.io/api/ohlcv/hd/{}{:02d}{:02d}/{}'.format(year, month, day, instrument)
        print("Hitting", url)
        try:
            r = requests.get(url)
            candles = np.array(json.loads(r.json()['data1m']))
            pdf = pd.DataFrame(candles)
            pdf = pdf.set_index(0)
        except Exception as e:
            print("Exception occured:", e)
            continue
        filename = '{}/cexio_minutes_{}_{}{:02d}{:02d}.json'.format(savepath, instrument.replace('/',''), year, month, day)
        pdf.to_json(filename)
        print('Saved', filename)


def save_hitbtc_candles(instrument, start, end, savepath):
    for ts in range(start, end, 60000):
        time.sleep(0.1)
        url = 'https://api.hitbtc.com/api/2/public/candles/{}?period=M1&from={}&limit=1000'.format(instrument, ts)
        print("Hitting", url)
        try:
            r = requests.get(url)
            pdf = pd.DataFrame(r.json()).set_index('timestamp')
        except Exception as e:
            print("Exception occured:", e)
            continue
        filename = '{}/hitbtc_minutes_{}_{}.json'.format(savepath, instrument, ts)
        pdf.to_json(filename)
        print('Saved', filename)


def load_custom(path):
    with open(path, 'r') as f:
        raw = json.load(f)
    candles = np.zeros((len(raw.keys()), 5))
    for idx, ts in enumerate(sorted(raw.keys())):
        candles[idx][0] = int(ts)
        candles[idx][1] = raw[ts]['open']
        candles[idx][2] = raw[ts]['close']
        candles[idx][3] = raw[ts]['high']
        candles[idx][4] = raw[ts]['low']
    return candles


def load_candles_numpy(path):
    candles = pd.read_json(path).values
    for i in range(len(candles)):
        candles[0] /= 1000

    return candles


def load_cryptowatch_candles(path):
    candles = np.array(pd.read_json(path)['result'][0])
    lows = candles[:,3]
    closes = candles[:,4]
    highs = candles[:,2]
    candles[:, 2] = closes
    candles[:, 3] = highs
    candles[:, 4] = lows
    return candles