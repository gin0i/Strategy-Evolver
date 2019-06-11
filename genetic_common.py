import copy
import random
import numpy as np
import time
import operator
from functools import lru_cache
import json
from pony.orm import *

MAXPOP = 32

db = Database()
#db.bind(provider='sqlite', filename=':memory:')
db.bind(provider='postgres', user='thomas', password='mcf', host='localhost', database='thomasdb')

class IndividualProfile(db.Entity):
    summary     =  PrimaryKey(str)
    parameters  =  Required(Json)
    score       =  Required(float)


db.generate_mapping(create_tables=True)
set_sql_debug(False)

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))
        return ret
    return wrap


def grouped(iterable, n):
    return zip(*[iter(iterable)]*n)

class Gene(object):
    def __init__(self):
        self.allels = []
        self.current_allel_idx = 0

    def mutate(self):
        halfway = max(1, int(len(self.allels)/3))
        boundaries = (-halfway, halfway)
        possible_idx_changes = range(*boundaries)
        self.current_allel_idx += random.choice(possible_idx_changes)
        self.current_allel_idx = self.current_allel_idx % len(self.allels)

    def express(self):
        return self.allels[self.current_allel_idx]

    def sumup(self):
        try:
            return self.allels[self.current_allel_idx].__name__
        except:
            return self.express()


class Chromosome(object):
    def init_genes(self, genes_classes):
        self.genes = {}
        self.genes_classes = genes_classes
        for gene in self.genes_classes:
            self.genes[gene.__name__] = gene()
        [self.mutate() for i in range(5)]

    def set_genes_allels(self, allels_values):
        for name, value in allels_values.items():
            self.genes[name].current_allel_idx = self.genes[name].allels.index(value)

    def mutate(self):
        genes_names = self.genes.keys()
        num_mutations = 1 if len(genes_names) == 1 else random.choice(range(1,len(genes_names)))
        to_mutate = random.sample(genes_names, num_mutations)
        for gene_name in to_mutate:
            self.genes[gene_name].mutate()

    def to_profile(self):
        o = {k: v.sumup() for k,v in self.genes.items()}
        return o


class Individual(object):
    def __init__(self, chromosomes_classes):
        self.chromosomes_classes = chromosomes_classes
        self.chromosomes = [c() for c in self.chromosomes_classes]
        [self.mutate() for i in range(7)]

    def mutate(self):
        to_mutate = random.sample(self.chromosomes, random.choice(range(1,len(self.chromosomes))))
        for chromosome in to_mutate:
            chromosome.mutate()

    def to_profile(self, score):
        o = {k: v.to_profile() for k,v in self.chromosomes.items()}
        return IndividualProfile(summary = self.sumup(), parameters=o, score=score)


class Evolver(object):
    def __init__(self):
        self.search_space = {}
        self.popsize = None

    def add_fresh_blood(self, pop):
        pop += self.spawn_individuals(pop)

    def search(self, times, initial_population):
        population = self.spawn_individuals(initial_population)
        for i in range(times):
            print("************* New Generation ({}), search space explored: {} *************".format(i, len(self.search_space)))
            population = self.reproduce(population)
            population = self.epurate(population)
            self.add_fresh_blood(population)
        return population

    def epurate(self, population):
        print("Epurating population of size", len(population))
        scores, individuals = self.evaluate_population(population)
        return self.select_fittests(scores, individuals)

    # @timing
    def select_fittests(self, scores, individuals):
        new_population, i = [], 0
        num_individuals = len(individuals)
        for name, score in sorted(scores.items(), key=operator.itemgetter(1), reverse=True):
            if i < num_individuals * 0.3 or i >= num_individuals * 0.90:
                if i <= 16:
                    print("Keeping individual {}: with {} because at rank {} on pop size {}".format(name, score, i, num_individuals))
                new_population.append(individuals[name])
            i += 1
            if len(new_population) >= self.popsize:
                break
        return new_population

    # @timing
    def reproduce(self, population):
        children = []
        for x in population:
            y = random.choice(population)
            while y.sumup() == x.sumup():
                y = random.choice(population)
            children += Evolver.mate_individuals(x, y, 2, self.search_space)
        for x in population[:10]:
            for i in range(4):
                y = random.choice(population)
                while y.sumup() == x.sumup():
                    y = random.choice(population)
                children += Evolver.mate_individuals(x, y, 2, self.search_space)
        return population + children

    @staticmethod
    def crossover_chromosomes(chrom_one, chrom_two):
        temp = copy.deepcopy(chrom_two)
        genes_names = chrom_two.genes.keys()
        to_cross = random.sample(genes_names, int(len(genes_names) / 2))
        for gene_name in to_cross:
            chrom_two.genes[gene_name] = chrom_one.genes[gene_name]
            chrom_one.genes[gene_name] = temp.genes[gene_name]

    @staticmethod
    def individual_evaluated(individual_summary):
        # queried = IndividualProfile.select(lambda p: p.summary == individual_summary)
        # exists = queried.first()
        exists = IndividualProfile.get(summary=individual_summary)
        return False if (exists is None) else True

    @staticmethod
    def mutate_and_add(offspring, all_offsprings, search_space, mutate_min=0):
        if random.random() > 0.33: [offspring.mutate() for i in range(mutate_min)]
        #while offspring.sumup() in search_space:
        while Evolver.individual_evaluated(offspring):
            offspring.mutate()
        # search_space[offspring.sumup()] = True
        all_offsprings.append(offspring)

    @staticmethod
    @db_session
    def mate_individuals(ind_a, ind_b, num, search_space):
        offsprings = []
        for i in range(int(num / 3)):
            temp_a = copy.deepcopy(ind_a)
            temp_b = copy.deepcopy(ind_b)
            for x, y in zip(sorted(temp_a.chromosomes.keys()), sorted(temp_b.chromosomes.keys())):
                Evolver.crossover_chromosomes(temp_a.chromosomes[x], temp_b.chromosomes[y])
            for child in [temp_a, temp_b]:
                minimum_mutations = random.choice(range(2))
                Evolver.mutate_and_add(child, offsprings, search_space, mutate_min=minimum_mutations)
        return offsprings
