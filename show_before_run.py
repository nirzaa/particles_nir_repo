from data_loader import EcalDataIO
import os
import numpy as np
from collections import Counter
import torch
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'
 
en_dep = EcalDataIO.ecalmatio(os.path.join('D:', os.sep, 'local_github', 'particles_nir_repo', 'data', 'raw', 'signal.al.elaser.IP05.edeplist.mat'))  # Dict with 100000 samples {(Z,X,Y):energy_stamp}
# energies = EcalDataIO.xymatio(os.path.join('D:', os.sep, 'local_github', 'particles_nir_repo', 'data', 'raw', 'signal.al.elaser.IP05.energy.mat'))
energies = EcalDataIO.energymatio(os.path.join('D:', os.sep, 'local_github', 'particles_nir_repo', 'data', 'raw', 'signal.al.elaser.IP05.energy.mat'))

en_list = [energy for energy in energies.values()]
num_classes = 100
x_lim = 13

######### Energy bins Generation ########
for i, en in enumerate(en_list):
    bin_num = num_classes
    final_list = [0] * bin_num  # The 20 here is the bin number - it may be changed of course.

    # bin_list = np.linspace(0, x_lim, int(bin_num * 0.5))  # Generate the bin limits
    bin_list = np.linspace(0, x_lim, bin_num)  # Generate the bin limits

    # binplace = np.digitize(en_list, bin_list)  # Divide the list into bins
    binplace = np.digitize(en, bin_list)  # Divide the list into bins
    bin_partition = Counter(binplace)  # Count the number of showers for each bin.
    for k in bin_partition.keys():
        final_list[int(k) - 1] = bin_partition[k]
    n = sum(final_list)
    # final_list = [f / n for f in final_list]    # Bin Normalization by sum
    final_list = torch.Tensor(final_list)  # Wrap it in a tensor - important for training and testing.
    if i == 0:
        final = final_list
        final = final.unsqueeze(axis=0)
    else:
        final_list = final_list.unsqueeze(axis=0)
        final = torch.cat((final, final_list), axis=0)
#########################################
sumy = final.sum(axis=0)
rng = np.linspace(1, x_lim, num_classes, dtype='float')
plt.bar(rng, sumy)
plt.show()

exit()