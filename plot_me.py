from data_loader import EcalDataIO
import os

i = 5
edep_file = os.path.join('D:', os.sep, 'local_github', 'particles_nir_repo', 'data', 'raw', f'signal.al.elaser.IP0{i}.edeplist.mat')
en_file = os.path.join('D:', os.sep, 'local_github', 'particles_nir_repo', 'data', 'raw', f'signal.al.elaser.IP0{i}.energy.mat')
en_dep = EcalDataIO.ecalmatio(edep_file)  # Dict with 100000 samples {(Z,X,Y):energy_stamp}
energies = EcalDataIO.energymatio(en_file)

# Eliminate multiple numbers of some kind
min_shower_num = 1
max_shower_num = 20
if min_shower_num > 0:
    del_list = []
    for key in energies:
        if len(energies[key]) < min_shower_num or len(energies[key]) >= max_shower_num:
            del_list.append(key)
    for d in del_list:
        del energies[d]
        del en_dep[d]
