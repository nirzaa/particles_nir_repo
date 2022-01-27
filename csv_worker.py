from numpy import dtype
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
import plotly.express as px
from tqdm import tqdm
import numpy as np
from torchsummary import summary

def means_plotter():
    means = []
    stds = []
    epoch_list = np.linspace(10, 100, 10, dtype='int')
    my_path_fig = os.path.join('./', 'csv_files', 'rel_error_means')

    for epoch in epoch_list:
        my_path = os.path.join('./', 'csv_files', 'project2', '5_class', f'epoch_{epoch}', 'data_frame.csv')
        df = pd.read_csv(my_path)
        my_mean = df.rel_error.mean()
        my_std = df.rel_error.std()
        means.append(my_mean)
        stds.append(my_std)

    plt.figure(num=2, figsize=(12, 6))
    plt.clf()
    plt.scatter(epoch_list, means, label='means')
    plt.errorbar(epoch_list, means, xerr=0, yerr=stds)
    plt.legend()
    plt.plot()
    plt.savefig(os.path.join(my_path_fig))

def weights_plotter():
    my_folder = os.path.join('./', 'saved', 'models', 'my_model', '0126_093256')
    for idx, file in enumerate(os.listdir(my_folder)):
        cntr = idx + 1
        # check the files which are end with specific extension
        if cntr % 10 == 0:
            saved_folder = os.path.join('./', 'csv_files', f'epoch_{cntr}')
            if not os.path.exists(saved_folder):
                os.makedirs(saved_folder)
            if file.endswith('.pth'):
                if file.startswith('checkpoint'):
                    checkpoint = torch.load(os.path.join(my_folder, file))
                    state_dict = checkpoint['state_dict']
                    layers = list(state_dict.keys())
                    weights = list(state_dict.values())
            # line 47 in trainer
            # print('Model output')
            # print('='*50)
            # summary(vgg, (3, 224, 224)) 
            print('Model architecture')
            print('='*50)
            for cntr1, layer in enumerate(layers):
                print(cntr1, layer, weights[cntr1].shape)
            for cntr1, layer in enumerate(tqdm(layers)):
                if cntr1 == 78:
                    plt.figure(num=0, figsize=(12, 6))
                    plt.clf()
                    plt.plot(weights[cntr1])
                    plt.savefig(os.path.join(saved_folder, f'epoch{cntr1}_{layer}.png'))
                elif cntr1 == 7:
                    plt.figure(num=0, figsize=(12, 6))
                    plt.clf()
                    plt.plot(weights[cntr1])
                    plt.savefig(os.path.join(saved_folder, f'epoch{cntr1}_{layer}.png'))
                elif cntr1 == 66:
                    plt.figure(num=0, figsize=(12, 6))
                    plt.clf()
                    plot_weights = weights[cntr1].reshape((weights[cntr1].shape[0], weights[cntr1].shape[1]))
                    plt.imshow(plot_weights, vmin=plot_weights.min(), vmax=plot_weights.max(),
                        extent =[0, plot_weights.shape[0], 0, plot_weights.shape[1]],
                        interpolation ='nearest', origin ='lower',
                        aspect='auto')
                    plt.colorbar()
                    plt.savefig(os.path.join(saved_folder, f'epoch{cntr1}_{layer}.png'))
                elif cntr1 == 77:
                    plt.figure(num=0, figsize=(12, 6))
                    plt.clf()
                    plot_weights = weights[cntr1].reshape((weights[cntr1].shape[0], weights[cntr1].shape[1]))
                    plt.imshow(plot_weights, vmin=plot_weights.min(), vmax=plot_weights.max(),
                        extent =[0, plot_weights.shape[0], 0, plot_weights.shape[1]],
                        interpolation ='nearest', origin ='lower',
                        aspect='auto')
                    plt.colorbar()
                    plt.savefig(os.path.join(saved_folder, f'epoch{cntr1}_{layer}.png'))
                    # plot_weights = weights[cntr1].reshape((weights[cntr1].shape[0], weights[cntr1].shape[1]))
                    # px.imshow(plot_weights)
                    # plt.savefig(os.path.join(saved_folder, f'epoch{cntr1}_{layer}.png'))



if __name__ == '__main__':
    means_plotter()
    weights_plotter()