import numpy as np
import sys
import argparse
import tables as tb
import pandas as pd
import matplotlib.pylab as plt
import invisible_cities.io.pmaps_io as pmapio
import invisible_cities.reco.pmaps_functions as pmapsf
import invisible_cities.database.load_db as db
from invisible_cities.io.table_io import make_table
from invisible_cities.reco.xy_algorithms import barycenter
from invisible_cities.io.mcinfo_io import load_mchits
from invisible_cities.core.core_functions import weighted_mean_and_std
from invisible_cities.core.core_functions import shift_to_bin_centers
from glob import glob


def compute_position_and_charges(hits_dict, pmaps, sipm_xs, sipm_ys):
    nevents   = len(pmaps)
    positions = np.empty([nevents, 3])
    pmts      = np.empty(nevents)
    sipm_es   = np.zeros(nevents)
    evts      = np.zeros(nevents)

    for idx, evt in enumerate(pmaps.keys()):
        hits = hits_dict[evt]
        print(idx, evt)
        # MC True position
        nhits    = len(hits)
        pos_tmp  = np.zeros([nhits, 3])
        energies = np.zeros([nhits])

        for i, hit in enumerate(hits):
            pos_tmp [i] = hit.pos
            energies[i] = hit.energy

        positions[idx], _ = weighted_mean_and_std(pos_tmp, energies, axis=0)

        # Select pmap
        pmap = pmaps[evt]

        # PMTs and Sipms energy
        pmt_sum = 0
        sipm_sum = np.zeros(1792)

        for s2 in pmap.s2s:
            pmt_sum  =  pmt_sum + s2.pmts .sum_over_sensors.sum()
            ids = s2.sipms.ids
            #if no sensors continue
            if ids.shape[0] == 0:
                continue
            sipm_sum[ids] = sipm_sum[ids] + s2.sipms.sum_over_times

        pmts   [idx] = pmt_sum
        sipm_es[idx] = sipm_sum.sum()
        evts   [idx] = evt

    return evts, positions, pmts, sipm_es

class ResetPartialMapTable(tb.IsDescription):
    evt   = tb.Int32Col  (pos=0)
    pmts  = tb.Float32Col(pos=1)
    sipms = tb.Float32Col(pos=2)
    x_mc  = tb.Float32Col(pos=3)
    y_mc  = tb.Float32Col(pos=4)
    z_mc  = tb.Float32Col(pos=5)

# writers
def map_writer(hdf5_file, table_name, *, compression='ZLIB4'):
    map_table  = make_table(hdf5_file,
                            group       = 'ResetMap',
                            name        = table_name,
                            fformat     = ResetPartialMapTable,
                            description = 'Probability map',
                            compression = compression)

    def write_map(evts, pmts, sipms, positions):
        for evt, pmt, sipm, pos in zip(evts, pmts, sipms, positions):
            row = map_table.row
            row["evt"  ] = evt
            row["pmts" ] = pmt
            row["sipms"] = sipm
            row["x_mc" ] = pos[0]
            row["y_mc" ] = pos[1]
            row["z_mc" ] = pos[2]
            row.append()
    return write_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-pmaps", required=True)
    parser.add_argument("-o", required=True)

    args = parser.parse_args(sys.argv[1:])
    pmaps_file = args.pmaps
    map_file = args.o

    # Load pmaps
    pmaps = pmapio.load_pmaps(pmaps_file)
    hits_dict = load_mchits(pmaps_file)

    # Get sensor positions
    data_sipm = db.DataSiPM(0)
    sipm_xs = data_sipm.X
    sipm_ys = data_sipm.Y

    evts, positions, pmts, sipms = compute_position_and_charges(hits_dict, pmaps, sipm_xs, sipm_ys)

    with tb.open_file(map_file, 'w') as h5out:
        write_dst  = map_writer(h5out, 'dst')
        write_dst(evts, pmts, sipms, positions)
