import pandas as pd
import os
import matplotlib.pyplot as plt

def means_plotter():
    means = []
    epoch_list = [10,15,20]
    my_path_fig = os.path.join('./', 'csv_files', 'rel_error_means')

    for epoch in epoch_list:
        my_path = os.path.join('./', 'csv_files', f'epoch_{epoch}', 'data_frame.csv')
        df = pd.read_csv(my_path)
        my_mean = df.rel_error.mean()
        means.append(my_mean)

    plt.figure(num=2, figsize=(12, 6))
    plt.clf()
    plt.scatter(epoch_list, means, label='means')
    plt.legend()
    plt.plot()
    plt.savefig(os.path.join(my_path_fig))

if __name__ == '__main__':
    means_plotter()