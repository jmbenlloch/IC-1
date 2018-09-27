import pandas as pd
import sys
import argparse
import tables as tb
import numpy as np
from glob import glob
import invisible_cities.io.pmaps_io as pio
from invisible_cities.io.mcinfo_io import load_mchits
from invisible_cities.core.core_functions import weighted_mean_and_std

#Read voxels and likelihoods
def read_data(files):
    partial_voxels = []
    partial_lhoods = []

    for f in files:
        try:
            with tb.open_file(f) as h5in:
                partial_voxels.append(pd.DataFrame.from_records(h5in.root.RECO.RESET.read()))
                partial_lhoods.append(pd.DataFrame.from_records(h5in.root.RECO.Likelihood.read()))
        except tb.HDF5ExtError:
            print('Error in file {}'.format(f))

    voxels = pd.concat(partial_voxels)
    likelihoods = pd.concat(partial_lhoods)
    return voxels, likelihoods

#Compute barycenter
def compute_barycenter(pmaps, voxels, likelihoods):
    events = np.unique(likelihoods.event.values)
    nevents = events.shape[0]
    positions_reset = np.empty([nevents, 3])
    positions_true  = np.empty([nevents, 3])
    energies_true   = np.empty([nevents])
    energies_reset  = np.empty([nevents])

    pmts_charge  = np.empty([nevents])
    sipms_charge = np.empty([nevents])

    for index, evt in enumerate(events):
        if not (index % 100):
            print(index, evt)
        #Read RESET info and compute barycenter
        data = voxels[voxels.event == evt]
        nvoxels = data.shape[0]

        positions = np.empty([nvoxels, 3])
        energies  = np.empty([nvoxels])

        i = 0
        for _, row in data.iterrows():
            positions[i] = row.x, row.y, row.z
            energies [i] = row.E
            i = i +1

        positions_reset[index], _ = weighted_mean_and_std(positions, energies, axis=0)
        energies_reset[index] = energies.sum()

        # MC True
        hits_evt = hits_dict[evt]
        nhits = len(hits_evt)

        positions = np.empty([nhits, 3])
        energies  = np.empty([nhits])

        for i, hit in enumerate(hits_evt):
            positions[i] = hit.pos
            energies [i] = hit.energy

        positions_true[index], _ = weighted_mean_and_std(positions, energies, axis=0)
        energies_true[index] = energies.sum()

        pmap = pmaps[evt]
        pmt_sum  = 0
        sipm_sum = 0
        for s2 in pmap.s2s:
            pmt_sum  = pmt_sum  + s2.pmts .sum_over_sensors.sum()
            sipm_sum = sipm_sum + s2.sipms.sum_over_sensors.sum()

        pmts_charge [index] = pmt_sum
        sipms_charge[index] = sipm_sum

    return events, positions_true, positions_reset, pmts_charge, \
           sipms_charge, energies_true, energies_reset


#Build dataframe
def build_dataframe(df_file, events, positions_true, positions_reset,
                    pmts_charge, sipms_charge, energies_true, energies_reset):
    nevents = events.shape[0]
    df_data = np.empty([nevents,11])
    index = 0
    for d in zip(events, positions_true, positions_reset, pmts_charge, sipms_charge, energies_true, energies_reset):
        df_data[index][0]   = d[0]
        df_data[index][1:4] = d[1]
        df_data[index][4:7] = d[2]
        df_data[index][7]   = d[3]
        df_data[index][8]   = d[4]
        df_data[index][9]  = d[5]
        df_data[index][10]  = d[6]
        index = index + 1

    reset_df = pd.DataFrame(df_data, columns=['evt', 'x_mc', 'y_mc', 'z_mc', 'x', 'y', 'z', 'pmts', 'sipms', 'E_mc', 'E'])
    reset_df.to_hdf(df_file, key='reset_df', mode='w')
    return reset_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-reset", required=True)
    parser.add_argument("-mc", required=True)
    parser.add_argument("-pmaps", required=True)
    parser.add_argument("-o", required=True)

    args = parser.parse_args(sys.argv[1:])
    pmaps_file = args.pmaps
    mc_file = args.mc
    path = args.reset + '/*h5'
    df_file = args.o

    #Get files
    files = glob(path)
    files.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))

    #Load pmaps and hits
    pmaps = pio.load_pmaps('/home/jmbenlloch/reset/toy/data/detsim_test.h5')
    hits_dict = load_mchits(mc_file)


    voxels, likelihoods = read_data(files)
    compute_barycenter(pmaps, voxels, likelihoods)
    events, positions_true, positions_reset, pmts_charge, sipms_charge, energies_true, energies_reset = compute_barycenter(pmaps, voxels, likelihoods)
    build_dataframe(df_file, events, positions_true, positions_reset, pmts_charge, sipms_charge, energies_true, energies_reset)

