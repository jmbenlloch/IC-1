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
def read_data(filein):
    voxels = {}
    with tb.open_file(filein) as h5in:
        basename = 'voxels_{}'
        for i in range(100,1001,100):
            table_name = basename.format(i)
            if i==1000:
                table_name = 'RESET'
            table = getattr(h5in.root.RECO, table_name)
            voxels[i] = pd.DataFrame.from_records(table.read())

        likelihoods = pd.DataFrame.from_records(h5in.root.RECO.Likelihood.read())

    return voxels, likelihoods

#Compute barycenter
def compute_barycenter_mc(pmaps, events):
    events = np.unique(likelihoods.event.values)
    nevents = events.shape[0]
    positions_true  = np.empty([nevents, 3])
    energies_true   = np.empty([nevents])

    pmts_charge  = np.empty([nevents])
    sipms_charge = np.empty([nevents])

    for index, evt in enumerate(events):
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

    return positions_true, pmts_charge, sipms_charge, energies_true

#Compute barycenter
def compute_barycenter(voxels, events):
    nevents = events.shape[0]
    positions_reset = np.empty([nevents, 3])
    energies_reset  = np.empty([nevents])

    for index, evt in enumerate(events):
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

    return positions_reset, energies_reset

def compute_barycenters(voxels, events):
    positions_reset = {}
    energies_reset = {}

    for key, value in voxels.items():
        positions, energies = compute_barycenter(value, events)

        positions_reset[key] = positions
        energies_reset [key] = energies

    return positions_reset, energies_reset

#Build dataframe
def build_dataframe(iterations, events, positions_true, positions_reset,
                    pmts_charge, sipms_charge, energies_true, energies_reset):
    nevents = events.shape[0]
    df_data = np.empty([nevents,12])
    index = 0
    for d in zip(events, positions_true, positions_reset, pmts_charge, sipms_charge, energies_true, energies_reset):
        df_data[index][0]   = iterations
        df_data[index][1]   = d[0]
        df_data[index][2:5] = d[1]
        df_data[index][5:8] = d[2]
        df_data[index][8]   = d[3]
        df_data[index][9]   = d[4]
        df_data[index][10]  = d[5]
        df_data[index][11]  = d[6]
        index = index + 1

    reset_df = pd.DataFrame(df_data, columns=['iterations', 'evt', 'x_mc', 'y_mc', 'z_mc', 'x', 'y', 'z', 'pmts', 'sipms', 'E_mc', 'E'])
    return reset_df

def build_complete_dataframe(positions_reset, energies_reset):
    dfs = []
    for iters in positions_reset.keys():
        df = build_dataframe(iters, events, positions_true, positions_reset[iters], pmts_charge, sipms_charge, energies_true, energies_reset[iters])
        dfs.append(df)

    data = pd.concat(dfs)
    return data

def write_data(df_file, reset_file):
    reset_df.to_hdf(df_file, key='reset_df', mode='w')

    with tb.open_file(df_file, 'a') as h5df, \
         tb.open_file(reset_file) as h5in:
        h5in.root.RECO.Likelihood.copy(h5df.root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-reset", required=True)
    parser.add_argument("-pmaps", required=True)
    parser.add_argument("-o", required=True)

    args = parser.parse_args(sys.argv[1:])
    pmaps_file = args.pmaps
    filein = args.reset
    df_file = args.o

    #Load pmaps and hits
    pmaps     = pio.load_pmaps(pmaps_file)
    hits_dict = load_mchits(pmaps_file)

    voxels, likelihoods = read_data(filein)
    events = np.unique(likelihoods.event.values)

    positions_true, pmts_charge, sipms_charge, energies_true = compute_barycenter_mc(pmaps, events)
    positions_reset, energies_reset = compute_barycenters(voxels, events)

    reset_df = build_complete_dataframe(positions_reset, energies_reset)
    write_data(df_file, filein)
