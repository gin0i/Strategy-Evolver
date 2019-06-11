from genetic_common import *
import talib
from numba import cuda
from context import NumpyContext
from gpgpu.gpu_backtest import simulate_all_strategies, CASH, cuda_run_kernel, buy_standalone, sell_standalone
from bot.minute_trade import *
from strategy_genes import *
from bot.transactions import *
from datetime import datetime
import talib
import random
from pprint import pprint
from backtest_tools.backtest_utils import *
import sys
import numpy as np
from pony.orm import *

SHOULD_BUY = 42
SHOULD_SELL = 666


MAVG = 0
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

db = Database()

# Fix fee in strategy/context
# Fix custom initialisation of population

signals_cache = {}

class StrategyIndividual(Individual):
    # Keep a map imported for chroosomes correspondence
    def __init__(self, chromosome_start_params=None):
        self.chromosomes = {
            'OSCIL': OscillatorChromosome(),
            'MAVG': MavgChromosome(),
            'CMAVG': CrossingMavgChromosome(),
            'TDSEQ': TDSeqChromosome(),
            'META': MetaChromosome(),
            'BBAND': BollingerChromosome(),
            'CANDLE': CandleChromosome(),
            'MOM': MomentumChromosome(),
        }
        # Factorise in other function
        if chromosome_start_params:
            for name, values in chromosome_start_params.items():
                self.chromosomes[name].set_genes_allels(values)

    # Keep a map imported for messages depending on the candle size
    def sumup(self):
        chromosome_a_name, chromosome_b_name = self.express()
        candle_func, output = self.candle_details()
        output += ' hourly candles'
        for name in sorted([chromosome_a_name, chromosome_b_name]):
            output += " with " + self.chromosomes[name].sumup()
        return output

    def express(self):
        return self.chromosomes['META'].express()

    def candle_details(self):
        func = self.chromosomes['CANDLE'].express()
        return func, func.__name__

    def mutate(self):
        self.chromosomes['CANDLE'].mutate()
        if random.random() > 0.33: self.chromosomes['META'].mutate()
        chromosome_a_name, chromosome_b_name = self.express()
        for chromosome in [chromosome_a_name, chromosome_b_name]:
            if random.random() > 0.33: self.chromosomes[chromosome].mutate()


class StratEvaluator(object):
    # Create evaluator model and evaluator manager
    def __init__(self, all_candles, starting_cash):
        self.all_candles = all_candles
        self.starting_cash = starting_cash
        self.buffers = {}

    # Detach in user-implementable function
    # Factorise buys-getting function
    def get_individual_score(self, results, id, min_buys=0):
        individual_result = results[id]
        final_cash = individual_result[-1, CASH]
        buys = individual_result[:, BUYS][individual_result[:, BUYS] > 0]
        buys_number = len(buys)
        if buys_number < min_buys:
            return 0
        return final_cash

    # Factorise save function in other def
    # Factorise evaluate individual_on_period
    # Get configuration from config holder
    #   min buys
    #   save outputs
    def evaluate_on_period(self, population, scores, min_buys, session_name, period_candles, save_outputs=True):
        outputs = simulate_all_strategies(self.buffers, period_candles, population, self.starting_cash, 0, init_strat, cuda_run_kernel, session_name)
        if save_outputs:
            from tempfile import TemporaryFile
            outfile = TemporaryFile()
            np.save(outfile, outputs)
        for indy_id, indy in enumerate(population):
            ind_name = indy.sumup()
            if ind_name not in scores:
                scores[ind_name] = {}
            scores[ind_name][session_name] = self.get_individual_score(outputs, indy_id, min_buys=min_buys)

    # Put this in evaluator manager and values in model?
    def evaluate_population(self, pop):
        step_scores = {}
        for exchange in self.all_candles.keys():
            step = 300
            for i in range(0, len(self.all_candles[exchange]) - step, step):
                session_name = str(exchange) + '-' + str(step) + '-' + str(i)
                period_candles = self.all_candles[exchange][i:i + step]
                self.evaluate_on_period(pop, step_scores, 2, session_name, period_candles)
            for step in [330, 1100]:
               for i in range(0, len(self.all_candles[exchange]) - step, step):
                   session_name = str(exchange) + '-' + str(step) + '-' + str(i)
                   period_candles = self.all_candles[exchange][i:i + step]
                   self.evaluate_on_period(pop, step_scores, 2, session_name, period_candles)
        return self.finalize_scores(pop, step_scores)

    # Detach to user-defined file
    # Factorise saving in DB to
    def set_total_score(self, indy, step_scores, all_scores):
        indy_scores = step_scores[indy.sumup()].values()
        min_res, max_res = min(indy_scores), max(indy_scores)
        mean_res = sum(indy_scores) / len(indy_scores)
        indy_score = (min_res + min_res + max_res + mean_res) / 4
        final_score = indy_score if (max_res != self.starting_cash) else 0
        profile = IndividualProfile.get(summary=indy.sumup())
        if profile is None:
            to_save = indy.to_profile(final_score)
            commit()
        all_scores[indy.sumup()] = final_score

    # Put this in evaluator manager
    def finalize_scores(self, pop, step_scores):
        individuals, scores = {}, {}
        for indy_id, indy in enumerate(pop):
            individuals[indy.sumup()] = indy
            self.set_total_score(indy, step_scores, scores)
        return scores, individuals


class StrategyEvolver(Evolver):
    def __init__(self, all_candles, starting_cash, base_pop_size, IndividualClass):
        super().__init__()
        self.popsize = base_pop_size
        self.IndividualClass = IndividualClass
        self.all_candles = all_candles
        self.starting_cash = starting_cash
        self.evaluator = StratEvaluator(all_candles, starting_cash)

    @db_session
    def evaluate_population(self, pop):
        return self.evaluator.evaluate_population(pop)

    #@timing
    @db_session
    def spawn_individuals(self, population):
        for i in range(self.popsize-len(population)):
            new_individual = self.IndividualClass()
            while Evolver.individual_evaluated(new_individual.sumup()):
                new_individual.mutate()

            population.append(new_individual)
        return population


def init_strat(strategy_id, indy, candle_data, signal_buffer, strategies_parameters, session_name):
    def compute_indicator(func, close_prices, chromosome_name):
        total_name = chromosome_name + session_name
        if total_name not in signals_cache:
            if len(signals_cache) > 2000:
                del signals_cache[random.choice(list(signals_cache.keys()))]
            signals_cache[total_name] = func(close_prices)
        return signals_cache[total_name]

    def translate_signal(chromosome_name, signal):
        chromosome = indy.chromosomes[chromosome_name]
        max_period = chromosome.max_period()
        strategies_parameters[strategy_id][signal] = max_period
        func = chromosome.express()
        signal_buffer[strategy_id][signal] = compute_indicator(func, candle_data, chromosome.sumup())

    candle_data_func, candle_type = indy.candle_details()
    candle_data = compute_indicator(candle_data_func, candle_data, candle_type)
    chromosome_a_name, chromosome_b_name = indy.express()
    translate_signal(chromosome_a_name, SIGNAL_A)
    translate_signal(chromosome_b_name, SIGNAL_B)



def launch_evolver_with_predefined_profiles(profiles, instrument_candles):
    volver = StrategyEvolver(instrument_candles, 50000, 400, StrategyIndividual)
    population = []
    for profile in profiles:
        population.append(StrategyIndividual(chromosome_start_params=profile))

    return volver.search(5050, population)



