import numpy as np


class DE(object):
    def __init__(self,
                 popul_size,
                 fitness_vec_size,
                 l_bound,
                 u_bound,
                 f,
                 cross_prob):
        self.l_bound = l_bound
        self.u_bound = u_bound
        self.population_size = popul_size
        self.fitness_vector_size = fitness_vec_size
        self.population = None
        self.mutants = list()
        self.trial = list()
        self.f = f
        self.cr = cross_prob

    def generate_population(self):
        rand_vals = (self.u_bound - self.l_bound) *\
            np.random.random((self.population_size, self.fitness_vector_size)) + self.l_bound
        self.population = np.array([Individual(p) for p in rand_vals])

    def mutate(self):
        for i, val in enumerate(self.population):
            indexes = np.random.permutation(np.delete(np.arange(len(self.population)), i))[:3]
            a, b, c = [x.phenotypes for x in self.population[indexes]]
            use_f = self.f or np.random.random() / 2.0 + 0.5
            self.mutants.append(a + use_f * (b - c))

    def crossover(self):
        for v, x in zip(self.mutants, [x.phenotypes for x in self.population]):
            tmp_vec = list()
            for (j, (vi, xi)) in enumerate(zip(v, x)):
                tmp_vec.append(vi if np.random.random() <= self.cr or
                               j == np.random.randint(0, self.fitness_vector_size) else xi)
            self.trial.append(Individual(tmp_vec))

    def select(self):
        self.population = np.array([x if x.fitness < u.fitness else u for x, u in zip(self.population, self.trial)])
        self.trial.clear()
        self.mutants.clear()


class Individual(object):
    def __init__(self, phenotypes):
        self.phenotypes = np.array(phenotypes)  # phenotype
        self.fitness = fitness_func(self.phenotypes)  # value of the fitness function

    def __str__(self):
        return '{0} = {1}'.format(self.phenotypes, self.fitness)


def fitness_func(arg_vec):
    # Sphere model (DeJong1)
    # return np.sum([x ** 2 for x in arg_vec])
    # Rosenbrock's saddle (DeJong2)
    # return sum([(100 * (xj - xi ** 2) ** 2 + (xi - 1) ** 2) for xi, xj in zip(arg_vec[:-1], arg_vec[1:])])
    # Rastrigin's function
    # return 10 * len(arg_vec) + np.sum([x ** 2 - 10 * np.cos(2 * np.pi * x) for x in arg_vec])
    # Ackley's Function
    s1 = -0.2 * np.sqrt(np.sum([x ** 2 for x in arg_vec]) / len(arg_vec))
    s2 = np.sum([np.cos(2 * np.pi * x) for x in arg_vec]) / len(arg_vec)
    return 20 + np.e - 20 * np.exp(s1) - np.exp(s2)


interval = (-30, 30)
eps = 1E-3
f = None  # differential weight[0, 2] if None random value from [0.5;1.0] is used
cr = 0.7  # crossover probability[0, 1]
ff_vec_size = 5
population_size = 50
max_epochs = 1000


def main():
    de = DE(population_size,
            ff_vec_size,
            *interval,
            f,
            cr)

    de.generate_population()
    print('Initial population')
    for ind in sorted(de.population, key=lambda x: x.fitness):
        print(ind)
    for i in range(max_epochs):
        de.mutate()
        de.crossover()
        de.select()
        print('{0}/{1} Current population:'.format(i + 1, max_epochs))
        print(sorted(de.population, key=lambda x: x.fitness)[0])

if __name__ == '__main__':
    main()
