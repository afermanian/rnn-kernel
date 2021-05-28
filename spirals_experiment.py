from get_results import get_ex_results, get_RNN_model
from adversarial import PGDL2
import pandas as pd
from generate_data import generate_spirals
import torch
from train_models import evaluate_model
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

sns.set()

experiment = 'spirals_penalization'
run_nums ='all'

df = get_ex_results(experiment, run_nums)
print(df.head())

df['accuracy_test'] = df['accuracy_test'].astype(float)
df['reg_lambda'] = df['reg_lambda'].astype(str)

print(df.groupby(['reg_lambda'])['accuracy_test'].mean())
print(df.groupby(['reg_lambda'])['accuracy_test'].std())

epsilon_grid = [0., 0.2, 0.4, 0.6, 0.8, 1.0]
steps = 50

df_1 = df[df['reg_lambda'] == 'nan']
df_2 = df[df['reg_lambda'] == '0.1']

n_runs = df_1.shape[0]
df_adv = pd.DataFrame(columns=['acc_test_adv', 'epsilon', 'model'])

for epsilon in epsilon_grid:
    print(epsilon)
    for i in range(n_runs):
        model_1 = get_RNN_model(df_1, experiment, df_1.index.astype(str)[i])
        model_2 = get_RNN_model(df_2, experiment, df_2.index.astype(str)[i])

        X_test, y_test = generate_spirals(1000, length=100)

        if epsilon == 0.:
            test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
            acc_test_1 = evaluate_model(model_1, test_dataloader)

            test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
            acc_test_2 = evaluate_model(model_2, test_dataloader)

        else:
            pgd_1 = PGDL2(model_1, epsilon, steps=steps)
            Xadv_1 = pgd_1(X_test, y_test)
            test_dataset = torch.utils.data.TensorDataset(Xadv_1, y_test)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
            acc_test_1 = evaluate_model(model_1, test_dataloader)

            pgd_2 = PGDL2(model_2, epsilon, steps=steps)
            Xadv_2 = pgd_2(X_test, y_test)
            test_dataset = torch.utils.data.TensorDataset(Xadv_2, y_test)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
            acc_test_2 = evaluate_model(model_2, test_dataloader)

        df_adv = df_adv.append({'epsilon': epsilon, 'acc_test_adv': acc_test_1, 'model': 'RNN'}, ignore_index=True)
        df_adv = df_adv.append({'epsilon': epsilon, 'acc_test_adv': acc_test_2, 'model': 'Penalized RNN'},
                               ignore_index=True)


print(df_adv.head())
df_adv.to_csv('results/spirals_adversarial.csv')

df_adv['acc_test_adv'] = df_adv['acc_test_adv'].astype(float)
fig, ax = plt.subplots()
sns.lineplot('epsilon', 'acc_test_adv', hue='model', data=df_adv, palette='colorblind', style='model', markers=True)
ax.legend().set_title('')
plt.xlabel(r'$\varepsilon$')
plt.ylabel(r'Adversarial accuracy')
plt.savefig('figures/spirals_adversarial.pdf')
plt.show()

