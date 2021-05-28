import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

sns.set()


df_adv = pd.read_csv('results/spirals_adversarial.csv')
fig, ax = plt.subplots()
sns.lineplot('epsilon', 'acc_test_adv', hue='model', data=df_adv, palette='colorblind', style='model', markers=True)
ax.legend().set_title('')
plt.xlabel(r'$\varepsilon$')
plt.ylabel(r'Adversarial accuracy')
plt.savefig('figures/spirals_adversarial.pdf')
plt.show()