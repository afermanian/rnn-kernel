import os
import pickle

import pandas as pd
from matplotlib import rc
import matplotlib.pylab as plt
import seaborn as sns

sns.set()

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


def plot_spirals_adversarial(experiment_dir):
    df_adv = pd.read_csv(os.path.join(experiment_dir, 'adversarial_accuracy.csv'))
    sns.lineplot(x='epsilon', y='acc_test_adv', hue='reg_lambda', data=df_adv, palette='colorblind', style='reg_lambda', markers=True)
    plt.xlabel(r'$\varepsilon$')
    plt.ylabel(r'Adversarial accuracy')
    plt.savefig(os.path.join(experiment_dir, 'spirals_adversarial.pdf'))

def plot_taylor_convergence(experiment_dir):
    df = pd.read_csv(os.path.join(experiment_dir, 'taylor_convergence.csv'))
    config = pickle.load(open(os.path.join(experiment_dir, 'config.pkl'), 'rb'))
    sns.lineplot(x='Step N', y='Error', hue='Activation', data=df, palette='colorblind', markers=True, style='Activation')
    plt.xlabel(r'Step $N$')
    plt.yscale('log')
    plt.savefig(os.path.join(experiment_dir, 'approx.pdf'), bbox_inches='tight')

    plt.figure()
    selection = df[(df['Activation'] == 'sigmoid') & (df['Step N'] == config['order'])]
    sns.scatterplot(x=selection['Weight L2 norm'], y=selection['Error'], palette='colorblind')
    plt.xlabel('Frobenius norm of the weights')
    plt.ylabel(r'Error for $N=5$')
    plt.yscale('log')
    plt.savefig(os.path.join(experiment_dir, 'error_norm_weights.pdf'), bbox_inches='tight')
