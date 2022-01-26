import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm

def means_plotter():
    means = []
    epoch_list = np.linspace(20, 100, 9, dtype='int')
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

def weights_anal():
    epochs_list = np.linspace(10, 100, 10, dtype='int')
    for epoch_num in tqdm(epochs_list):
        my_path_weights = os.path.join('./', 'csv_files', f'epoch_{epoch_num}')
        if not os.path.exists(my_path_weights):
            os.makedirs(my_path_weights)
        my_path = os.path.join('./', 'saved', 'models', 'my_model', '0126_110510', f'checkpoint-epoch{epoch_num}.pth')
        checkpoint = torch.load(my_path)
        weights = checkpoint['state_dict']
        layers = list(weights.keys())
        for i in [78, 77, 66]:
            ln = weights[layers[i]]
            plt.figure(num=0, figsize=(12, 6))
            if i == 78:
                plt.clf()
                plt.plot(ln.cpu())
            elif i == 77:
                plt.clf()
                im = plt.imshow(ln.cpu(), interpolation='nearest', aspect='auto')
                plt.colorbar(im)
            elif i == 66:
                plt.clf()
                data = ln.cpu().reshape(512, 256)
                im = plt.imshow(data, interpolation='nearest', aspect='auto')
                plt.colorbar(im)
            
            plt.savefig(os.path.join(my_path_weights, f'{layers[i]}.png'))

if __name__ == '__main__':
    means_plotter()
    # weights_anal()