import sys, os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


class Stats():
    def __init__(self, max_gen):
        self.MAX_GEN = max_gen
        self.generation = range(self.MAX_GEN - 1)

        self.coef_key = lambda x: x[0].coef_  # key for min and max for model coef

        self.b_models = {}
        self.gl_models = {}

        self.baseline_stats = getStats('baseline_eval')
        # Get all the models
        self.b_models['avg_fitness'] = self.getRegressionModels(self.baseline_stats, ' avg_fitness', self.MAX_GEN)
        self.b_models['best_fitness'] = self.getRegressionModels(self.baseline_stats, ' best_fitness', self.MAX_GEN)
        # compute the mean avg fitness and mean best fitness up to MAX_GEN
        self.concat_baseline_stats = pd.concat(self.baseline_stats, axis=0)
        self.baseline_stats['mean_avg_fitness'] = getMean(self.concat_baseline_stats, ' avg_fitness', self.MAX_GEN)
        self.baseline_stats['mean_best_fitness'] = getMean(self.concat_baseline_stats, ' best_fitness', self.MAX_GEN)
        # get linear models of the means
        self.b_models['mean_avg_fitness'] = getRegression(self.generation, self.baseline_stats['mean_avg_fitness'])
        self.b_models['mean_best_fitness'] = getRegression(self.generation, self.baseline_stats['mean_best_fitness'])
        # plot the means
        plot_mean_stats(self.baseline_stats['mean_avg_fitness'], self.baseline_stats['mean_best_fitness'], "Baseline mean average and mean best fitness")

        self.gl_stats = getStats('gl_eval')
        # Get all the models
        self.gl_models['avg_fitness'] = self.getRegressionModels(self.gl_stats, ' avg_fitness', self.MAX_GEN)
        self.gl_models['best_fitness'] = self.getRegressionModels(self.gl_stats, ' best_fitness', self.MAX_GEN)
        # compute the mean avg fitness and mean best fitness up to MAX_GEN
        self.concat_gl_stats = pd.concat(self.gl_stats, axis=0)
        self.gl_stats['mean_avg_fitness'] = getMean(self.concat_gl_stats, ' avg_fitness', self.MAX_GEN)
        self.gl_stats['mean_best_fitness'] = getMean(self.concat_gl_stats, ' best_fitness', self.MAX_GEN)
        # get linear models of the means
        self.gl_models['mean_avg_fitness'] = getRegression(self.generation, self.gl_stats['mean_avg_fitness'])
        self.gl_models['mean_best_fitness'] = getRegression(self.generation, self.gl_stats['mean_best_fitness'])
        # plot the means
        plot_mean_stats(self.gl_stats['mean_avg_fitness'], self.gl_stats['mean_best_fitness'], "Guided Learning mean average and mean best fitness")

    def getRegressionModels(self, df, col, max_gen):
        ret = []
        for run_key in df:
            reg = getRegression(self.generation, df[run_key][col][0:max_gen - 1])
            ret.append((reg[0], reg[1], run_key, df[run_key]))
        return ret

    def computeBaselineSlopes(self):
        # best fitness slopes
        self.b_models['highest_slope_best_fitness'] = max(self.b_models['best_fitness'], key=self.coef_key)
        self.b_models['lowest_slope_best_fitness'] = min(self.b_models['best_fitness'], key=self.coef_key)
        print("Baseline highest slope for best fitness: {}".format(self.b_models['highest_slope_best_fitness'][0].coef_))
        print("Baseline lowest slope for best fitness: {}".format(self.b_models['lowest_slope_best_fitness'][0].coef_))
        # avg fitness slopes
        self.b_models['highest_slope_avg_fitness'] = max(self.b_models['avg_fitness'], key=self.coef_key)
        self.b_models['lowest_slope_avg_fitness'] = min(self.b_models['avg_fitness'], key=self.coef_key)
        print("Baseline highest slope for avg fitness: {}".format(self.b_models['highest_slope_avg_fitness'][0].coef_))
        print("Baseline lowest slope for avg fitness: {}".format(self.b_models['lowest_slope_avg_fitness'][0].coef_))

    def computeGuidedLearningSlopes(self):
        # best fitness slopes
        self.gl_models['highest_slope_best_fitness'] = max(self.gl_models['best_fitness'], key=self.coef_key)
        self.gl_models['lowest_slope_best_fitness'] = min(self.gl_models['best_fitness'], key=self.coef_key)
        print("Guided Learning highest slope for best fitness: {}".format(self.gl_models['highest_slope_best_fitness'][0].coef_))
        print("Guided Learning lowest slope for best fitness: {}".format(self.gl_models['lowest_slope_best_fitness'][0].coef_))
        # avg fitness slopes
        self.gl_models['highest_slope_avg_fitness'] = max(self.gl_models['avg_fitness'], key=self.coef_key)
        self.gl_models['lowest_slope_avg_fitness'] = min(self.gl_models['avg_fitness'], key=self.coef_key)
        print("Guided Learning highest slope for avg fitness: {}".format(self.gl_models['highest_slope_avg_fitness'][0].coef_))
        print("Guided Learning lowest slope for avg fitness: {}".format(self.gl_models['lowest_slope_avg_fitness'][0].coef_))

    def plot_comparison_graphs(self):
        plt.figure(0, figsize=(30, 10))
        plt.plot(self.generation, self.baseline_stats['mean_avg_fitness'], 'b-', linestyle='--', label=" baseline mean average")
        plt.plot(self.generation, self.baseline_stats['mean_best_fitness'], 'r-', linestyle='-', label="baseline mean best")
        plt.plot(self.generation, self.b_models['mean_best_fitness'][1], 'g-', linestyle=':',
                 label='baseline linear regression')

        plt.plot(self.generation, self.gl_stats['mean_avg_fitness'], 'darkblue', linestyle='--', label="gl mean average")
        plt.plot(self.generation, self.gl_stats['mean_best_fitness'], 'maroon', linestyle='-', label="gl mean best")
        plt.plot(self.generation, self.gl_models['mean_best_fitness'][1], 'darkgreen', linestyle=':', label='gl linear regression')

        plt.title("Baseline vs. Guided Learning")
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.grid()
        plt.legend(loc="best")
        plt.show()

        plt.figure(0, figsize=(30, 10))
        plt.plot(self.generation, self.b_models['highest_slope_best_fitness'][3][' best_fitness'][:self.MAX_GEN-1], 'b-', linestyle='-', label="Baseline highest slope best fitness")
        plt.plot(self.generation, self.b_models['highest_slope_best_fitness'][1], 'b-', linestyle='--', label="Baseline highest slope best fitness regression")
        plt.plot(self.generation, self.gl_models['highest_slope_best_fitness'][3][' best_fitness'][:self.MAX_GEN - 1], 'r-', linestyle='-', label="Baseline highest slope best fitness")
        plt.plot(self.generation, self.gl_models['highest_slope_best_fitness'][1], 'r-', linestyle='--', label="GL highest slope best fitness")
        runInterventions = getRunInterventions('gl_eval', self.gl_models['highest_slope_best_fitness'][2])
        for intervention in runInterventions:
            plt.plot((intervention, intervention), (0, 2000))
        plt.title("Baseline vs. Guided Learning - Best Fitness - Highest Slope")
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.grid()
        plt.legend(loc="best")
        plt.show()

def getStats(root):
    if root[-1] == '/':
        root = root[0:-1]
    ret = {}
    for directory in os.listdir(root):
        if directory[0] != ".":
            # ignore hidden files and directories
            df = pd.read_csv('{}/{}/fitness.csv'.format(root, directory), dtype=np.float64)
            ret[directory] = df
    return ret

def getRegression(x, y):
    new_x = np.asarray(x).reshape(-1, 1)
    new_y = np.asarray(y).reshape(-1, 1)
    model = LinearRegression()
    model = model.fit(new_x, new_y)
    return model, model.predict(new_x)

def getRunInterventions(root, run_name):
    if root[-1] == '/':
        root = root[0:-1]
    interventions = []
    for directory in os.listdir(os.path.join(root, run_name, 'keras2neat')):
        if directory[0] != '.' and directory[-4:] != '.svg':
            interventions.append(directory.split("_gen_", 1)[1])
    return interventions

def getMean(dataframe, col, last_gen):
    mean = []
    for gen in range(1, last_gen):
        maf = dataframe[dataframe['gen'] == gen][col].mean()
        try:
            mean.append(int(maf))
        except ValueError:
            print("NaN @ gen: {}".format(gen))
            raise
    return mean

def plot_mean_stats(mean_avg_fitness, mean_best_fitness, title):
    generation = range(len(mean_avg_fitness))

    plt.figure(0, figsize=(30, 10))
    plt.plot(generation, mean_avg_fitness, 'b-', linestyle='--', label="mean average")
    plt.plot(generation, mean_best_fitness, 'r-', linestyle='-', label="mean best")

    model, regression = getRegression(generation, mean_best_fitness)
    plt.plot(generation, regression, 'g-', linestyle=':', label='linear regression')

    plt.title(title)
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    plt.show()


if __name__ == '__main__':
    stats = Stats(150)
    stats.computeGuidedLearningSlopes()
    print()
    stats.computeBaselineSlopes()
    print()
    stats.plot_comparison_graphs()

