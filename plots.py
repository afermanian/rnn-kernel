import distutils.spawn
import os
import pickle

from matplotlib import rc
import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns

sns.set()

if distutils.spawn.find_executable('latex'):
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)


def plot_spirals_adversarial(experiment_dir: str):
    """Saves the plot with the curve of accuracy vs size of the adversarial perturbations after the adversarial
    experiment has been run.

    :param experiment_dir: path to directory where the experiment is saved
    :return: None
    """
    df_adv = pd.read_csv(os.path.join(experiment_dir, 'adversarial_accuracy.csv'))
    sns.lineplot(x='epsilon', y='acc_test_adv', hue='model', data=df_adv, palette='colorblind',
                 style='model', markers=True)
    plt.xlabel(r'$\varepsilon$')
    plt.ylabel(r'Adversarial accuracy')
    plt.legend()
    plt.savefig(os.path.join(experiment_dir, 'spirals_adversarial.pdf'))


def plot_spirals_training_norms(experiment_dir: str):
    """Saves the plot with the norms during training of two RNN trained on the spirals experiment.

    :param experiment_dir: path to directory where the experiment is saved
    :return:
    """
    df_norms = pd.read_csv(os.path.join(experiment_dir, 'training_norms.csv'))
    plt.figure()
    sns.lineplot(x='epoch', y='norm_frobenius', hue='model', palette='colorblind', style='model',
                 data=df_norms, markers=True)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel(r'Frobenius norm of the weights')
    plt.savefig(os.path.join(experiment_dir, 'comparison_norm_Frobenius.pdf'))

    plt.figure()
    sns.lineplot(x='epoch', y='norm_kernel_smoothed', hue='model', palette='colorblind', style='model',
                 data=df_norms, markers=True)
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel(r'RKHS norm (N=3)')
    plt.savefig(os.path.join(experiment_dir, 'comparison_norm_kernel.pdf'))


def plot_taylor_convergence(experiment_dir: str):
    """Saves the plot with the curve of accuracy vs size of the adversarial perturbations after the Taylor experiment
    has been run.

    :param experiment_dir: path to directory where the experiment is saved
    :return: None
    """
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
    plt.ylabel('Error for $N={}$'.format(config['order']))
    plt.yscale('log')
    plt.savefig(os.path.join(experiment_dir, 'error_norm_weights.pdf'), bbox_inches='tight')

