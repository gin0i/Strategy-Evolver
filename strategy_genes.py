from genetic_common import *
import talib
import numpy as np
from backtest_tools.td_sequential import compute_td_seq
from backtest_tools.backtest_utils import heikin_ashi
import json

SHOULD_BUY = 42
SHOULD_SELL = 666

HIGH = 3
LOW = 4

class CandleSizeGene(Gene):
    def __init__(self):
        self.allels = [1]
        self.current_allel_idx = 0

class OversoldLevelGene(Gene):
    def __init__(self):
        self.allels = [i for i in range(11, 48)]
        self.max = max(self.allels)
        self.current_allel_idx = 3

class OverboughtLevelGene(Gene):
    def __init__(self):
        self.allels = [i for i in range(55, 90)]
        self.max = max(self.allels)
        self.current_allel_idx = 3

class MavgPeriodGene(Gene):
    def __init__(self):
        self.allels = [i for i in range(20, 1800, 4)]
        self.max = max(self.allels)
        self.current_allel_idx = 3


class TDSeqTypeGene(Gene):
    def __init__(self):
        self.allels = ['STRICT', 'SOFT']
        self.current_allel_idx = 1

class TDSeqChromosome(Chromosome):
    def __init__(self):
        super().init_genes( [TDSeqTypeGene])

    def express(self):
        strictness = self.genes['TDSeqTypeGene'].express()
        def compute(prices):
            soft, strict = compute_td_seq(prices)
            if strictness == 'SOFT':
                return soft
            elif strictness == 'STRICT':
                return strict
        return compute

    def max_period(self):
        return 10

    def sumup(self):
        return str(self.genes['TDSeqTypeGene'].express()) + " TD sequential"

class BBandsMaGene(Gene):
    def __init__(self):
        lower_bound, upper_bound = talib.MA_Type.SMA, talib.MA_Type.T3
        self.allels = [1,2,3,4,5,6,8]
        self.current_allel_idx = 3

class NbdevUpGene(Gene):
    def __init__(self):
        self.allels = [i / 2.0 for i in range(1, 5)]
        self.current_allel_idx = 3

class NbdevDnGene(Gene):
    def __init__(self):
        self.allels = [i / 2.0 for i in range(1, 5)]
        self.current_allel_idx = 3

class BBandsPeriodGene(Gene):
    def __init__(self):
        self.allels = [i for i in range(2, 60)]
        self.max = max(self.allels)
        self.current_allel_idx = 5

class BollingerChromosome(Chromosome):
    def __init__(self):
        super().init_genes([BBandsPeriodGene, NbdevUpGene, NbdevDnGene, BBandsMaGene])

    def express(self):
        timeperiod = self.genes['BBandsPeriodGene'].express()
        nbdevup = self.genes['NbdevUpGene'].express()
        nbdevdn = self.genes['NbdevDnGene'].express()
        matype = self.genes['BBandsMaGene'].express()
        def compute(prices):
            results = np.zeros(prices[:, 2].shape)
            upper, mid, lower = talib.BBANDS(prices[:, 2], timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)
            results[prices[:, 2] >= upper] = SHOULD_SELL
            results[prices[:, 2] <= lower] = SHOULD_BUY
            return results
        return compute

    def max_period(self):
        return self.genes['BBandsPeriodGene'].express()

    def sumup(self):
        return str(self.genes['BBandsPeriodGene'].express()) + " BBands period with " + str(self.genes['NbdevUpGene'].express()) + " devup, " + str(self.genes['NbdevDnGene'].express()) + " devdown and type " + str(self.genes['BBandsMaGene'].express())

class MavgTypeGene(Gene):
    def __init__(self):
        self.allels = [talib.EMA, talib.DEMA, talib.T3, talib.SMA]
        self.current_allel_idx = 1

class MavgChromosome(Chromosome):
    def __init__(self):
        super().init_genes( [MavgPeriodGene, MavgTypeGene])

    def express(self):
        timeperiod = self.genes['MavgPeriodGene'].express()
        mavg_func = self.genes['MavgTypeGene'].express()
        def compute(prices):
            results = np.zeros(prices[:, 2].shape)
            mavgs = mavg_func(prices[:, 2], timeperiod=timeperiod)
            diff = mavgs - prices[:, 2]
            results[diff > 0] = SHOULD_SELL
            results[diff < 0] = SHOULD_BUY

            return results
        return compute

    def max_period(self):
        return self.genes['MavgPeriodGene'].express()

    def sumup(self):
        return str(self.genes['MavgTypeGene'].express().__name__) +" mavg period " + str(self.genes['MavgPeriodGene'].express())


class CrossingMavgChromosome(MavgChromosome):
    def express(self):
        timeperiod = self.genes['MavgPeriodGene'].express()
        mavg_func = self.genes['MavgTypeGene'].express()

        def compute(prices):
            closes = prices[:, 2]
            results = np.zeros(prices[:, 2].shape)
            mavgs = mavg_func(prices[:, 2], timeperiod=timeperiod)
            diff = mavgs - closes
            crossings = np.append([0], np.diff(np.signbit(diff))) * diff/np.abs(diff)
            results[crossings > 0] = SHOULD_SELL
            results[crossings < 0] = SHOULD_BUY
            return results

        return compute

    def sumup(self):
        return 'Crossing ' + super().sumup()

class MomentumPeriodGene(Gene):
    def __init__(self):
        self.allels = [i for i in range(3, 60)]
        self.max = max(self.allels)
        self.current_allel_idx = 3

class MomentumTresholdGene(Gene):
    def __init__(self):
        self.allels = [i for i in range(4, 90)]
        self.max = max(self.allels)
        self.current_allel_idx = 5

class MomentumChromosome(Chromosome):
    def __init__(self):
        super().init_genes( [MomentumPeriodGene, MomentumTresholdGene])

    def express(self):
        timeperiod = self.genes['MomentumPeriodGene'].express()
        treshold = self.genes['MomentumTresholdGene'].express()
        def compute(prices):
            results = np.zeros(prices[:, 2].shape)
            momentum = talib.MOM(prices[:, 2], timeperiod=timeperiod)
            crossings = np.append([0], np.diff(np.sign(momentum - treshold)))
            results[crossings < 0] = SHOULD_SELL
            results[crossings > 0] = SHOULD_BUY
            return results
        return compute

    def max_period(self):
        return self.genes['MomentumPeriodGene'].express()

    def sumup(self):
        return "Momentum of period {} with treshold {}".format(self.genes['MomentumPeriodGene'].express(), self.genes['MomentumTresholdGene'].express())


class CandleGene(Gene):
    def __init__(self):
        # self.allels = [heikin_ashi] + 3*[CandleGene.normal]
        self.allels = [heikin_ashi] + 3 * [CandleGene.normal]
        self.current_allel_idx = 1

    @staticmethod
    def normal(candles):
        return candles

    def sumup(self):
        return self.express().__name__


class CandleChromosome(Chromosome):
    def __init__(self):
        super().init_genes( [CandleGene])

    def express(self):
        return self.genes['CandleGene'].express()

    def sumup(self):
        return self.genes['CandleGene'].sumup()

class IndicatorAGene(Gene):
    def __init__(self):
        # self.allels = ['OSCIL', 'MAVG', 'TDSEQ', 'BBAND', 'MOM', 'CMAVG']
        self.allels = ['OSCIL', 'MAVG', 'BBAND', 'MOM']
        self.current_allel_idx = 0

    def sumup(self):
        return self.express()

class IndicatorBGene(Gene):
    def __init__(self):
        #self.allels = ['OSCIL', 'MAVG', 'BBAND', 'MOM', 'CMAVG']
        self.allels = ['OSCIL', 'MAVG', 'BBAND', 'CMAVG']
        self.current_allel_idx = 1

    def sumup(self):
        return self.express()

class MetaChromosome(Chromosome):
    def __init__(self):
        super().init_genes([IndicatorAGene, IndicatorBGene])

    def express(self):
        indicator_a = self.genes['IndicatorAGene'].express()
        indicator_b = self.genes['IndicatorBGene'].express()
        return indicator_a, indicator_b

    def sumup(self):
        return '{} {}'.format(self.genes['IndicatorAGene'].express(), self.genes['IndicatorBGene'].express())


class OscillatorTypeGene(Gene):
    def __init__(self):
        self.allels = [talib.RSI, talib.CMO]
        self.current_allel_idx = 1


class OscillatorPeriodGene(Gene):
    def __init__(self):
        self.allels = [i for i in range(3, 60)]
        self.max = max(self.allels)
        self.current_allel_idx = 3


class OscillatorChromosome(Chromosome):
    def __init__(self):
        super().init_genes([OscillatorPeriodGene, OversoldLevelGene, OverboughtLevelGene, OscillatorTypeGene])

    def express(self):
        timeperiod = self.genes['OscillatorPeriodGene'].express()
        oscillator_function = self.genes['OscillatorTypeGene'].express()
        def compute(prices):
            results = np.zeros(prices[:, 2].shape)
            osc = oscillator_function(prices[:, 2], timeperiod=timeperiod)
            overbought_treshold = self.genes['OverboughtLevelGene'].express()
            oversold_treshold = self.genes['OversoldLevelGene'].express()
            results[osc >= overbought_treshold] = SHOULD_SELL
            results[osc <= oversold_treshold] = SHOULD_BUY
            return results
        return compute

    def max_period(self):
        return self.genes['OscillatorPeriodGene'].express()

    def sumup(self):
        oversold, overbought, = self.genes['OversoldLevelGene'].express(), self.genes['OverboughtLevelGene'].express()
        return "{} period {} with over-triggers {} {}".format(self.genes['OscillatorTypeGene'].express().__name__, self.genes['OscillatorPeriodGene'].express(), oversold, overbought)


class AroonPeriodGene(Gene):
    def __init__(self):
        self.allels = [i for i in range(3, 60)]
        self.max = max(self.allels)
        self.current_allel_idx = 3

class AroonUpGene(Gene):
    def __init__(self):
        self.allels = [i for i in range(40, 100)]
        self.max = max(self.allels)
        self.current_allel_idx = 20

class AroonDnGene(Gene):
    def __init__(self):
        self.allels = [i for i in range(0, 40)]
        self.max = max(self.allels)
        self.current_allel_idx = 20

class AroonChromosome(Chromosome):
    def __init__(self):
        super().init_genes([AroonPeriodGene, AroonDnGene, AroonUpGene])

    def express(self):
        timeperiod = self.genes['AroonPeriodGene'].express()
        triggerup = self.genes['AroonUpGene'].express()
        triggerdn = self.genes['AroonDnGene'].express()
        def compute(prices):
            results = np.zeros(prices[:, HIGH].shape)
            highs, lows = prices[:, HIGH], prices[:, LOW]
            aroondown, aroonup = talib.AROON(highs, lows, timeperiod=timeperiod)
            results[np.logical_and(aroonup > triggerup, aroondown < triggerdn)] = SHOULD_BUY
            results[np.logical_and(aroonup < triggerup, aroondown > triggerdn)] = SHOULD_SELL
            return results
        return compute

    def max_period(self):
        return self.genes['AroonPeriodGene'].express()

    def sumup(self):
        period = self.genes['AroonPeriodGene'].express()
        triggerup, triggerdn = self.genes['AroonUpGene'].express(), self.genes['AroonDnGene'].express()
        return "AROON of period {} with up/down triggers {}/{}".format(period, triggerup, triggerdn)


class MACDChromosome(Chromosome):
    def __init__(self):
        self.genes_classes = [OscillatorPeriodGene, OversoldLevelGene, OverboughtLevelGene]
        super().init_genes()
        [self.mutate() for i in range(5)]

    def sumup(self):
        oversold, overbought, = self.genes['OversoldLevelGene'].express(), self.genes['OverboughtLevelGene'].express()
        return "Rsi period {} with over-triggers {} {}".format(self.genes['OscillatorPeriodGene'].express(), oversold, overbought)

