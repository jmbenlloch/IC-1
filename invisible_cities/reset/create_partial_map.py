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

    sipm_es     = np.zeros(nevents*1792)
    sipm_xdists = np.zeros(nevents*1792)
    sipm_ydists = np.zeros(nevents*1792)

    idx = 0
    for evt in pmaps.keys():
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

        # if no charge skip event
        if pmt_sum == 0:
            continue

        pmts[idx] = pmt_sum

        # Sipms
        x_dists = positions[idx][0] - sipm_xs
        y_dists = positions[idx][1] - sipm_ys

        sipm_es    [idx*1792:(idx+1)*1792] = sipm_sum
        sipm_xdists[idx*1792:(idx+1)*1792] = x_dists
        sipm_ydists[idx*1792:(idx+1)*1792] = y_dists

        idx = idx + 1

    return positions[:idx], pmts[:idx], sipm_es[:idx*1792], sipm_xdists[:idx*1792], sipm_ydists[:idx*1792]

def compute_position_and_charges_ipmts(hits_dict, pmaps, pmt_xs, pmt_ys, sipm_xs, sipm_ys):
    nevents   = len(pmaps)
    positions = np.empty([nevents, 3])

    pmts       = np.empty([nevents, 12])
    pmt_xdists = np.zeros([nevents, 12])
    pmt_ydists = np.zeros([nevents, 12])

    sipm_es     = np.zeros(nevents*1792)
    sipm_xdists = np.zeros(nevents*1792)
    sipm_ydists = np.zeros(nevents*1792)

    idx = 0
    for evt in pmaps.keys():
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
        pmt_sum = np.zeros(12)
        sipm_sum = np.zeros(1792)

        for s2 in pmap.s2s:
            pmt_sum = pmt_sum + s2.pmts.sum_over_times
            ids = s2.sipms.ids
            #if no sensors continue
            if ids.shape[0] == 0:
                continue
            sipm_sum[ids] = sipm_sum[ids] + s2.sipms.sum_over_times

        # if no charge skip event
        if pmt_sum.sum() == 0:
            continue

        pmts[idx] = pmt_sum

        # Pmts
        # relative positions
        #pmt_xdists[idx] = positions[idx][0] - pmt_xs
        #pmt_ydists[idx] = positions[idx][1] - pmt_ys
        # absolute positions
        pmt_xdists[idx] = positions[idx][0]
        pmt_ydists[idx] = positions[idx][1]

        # Sipms
        sipm_x_dists = positions[idx][0] - sipm_xs
        sipm_y_dists = positions[idx][1] - sipm_ys

        sipm_es    [idx*1792:(idx+1)*1792] = sipm_sum
        sipm_xdists[idx*1792:(idx+1)*1792] = sipm_x_dists
        sipm_ydists[idx*1792:(idx+1)*1792] = sipm_y_dists

        idx = idx + 1

    return pmts[:idx], pmt_xdists[:idx], pmt_ydists[:idx], sipm_es[:idx*1792], sipm_xdists[:idx*1792], sipm_ydists[:idx*1792]


def compute_histogram(xs, ys, energies, nbins, range=None):
    values, xedges, yedges = np.histogram2d(xs, ys, weights=energies, bins=nbins, range=range)
    ones = np.ones_like(xs)
    counts, xedges, yedges = np.histogram2d(xs, ys, weights=ones, bins=nbins, range=range)

    xedges = shift_to_bin_centers(xedges)
    yedges = shift_to_bin_centers(yedges)

    xedges = np.repeat(xedges, len(xedges))
    yedges = np.tile(yedges, len(yedges))

    return xedges, yedges, values, counts

def build_coords(xs, ys, zs):
    xbins = len(xs)
    ybins = len(ys)
    zbins = len(zs)

    xs = np.repeat(xs, ybins*zbins)
    ys = np.tile(np.repeat(ys, zbins), xbins)
    zs = np.tile(zs, xbins*ybins)

    return xs, ys, zs


def compute_histogram3d(positions, energies, bins, range=None):
    ones = np.ones_like(energies)
    values, edges = np.histogramdd(positions, weights=energies, range=range, bins=bins)
    counts, _     = np.histogramdd(positions, weights=ones, range=range, bins=bins)

    xedges = shift_to_bin_centers(edges[0])
    yedges = shift_to_bin_centers(edges[1])
    zedges = shift_to_bin_centers(edges[2])

    xs, ys, zs = build_coords(xedges, yedges, zedges)

    return xs, ys, zs, values, counts


class ResetPartialMapTable(tb.IsDescription):
    x           = tb.Float32Col(pos=0)
    y           = tb.Float32Col(pos=1)
    value       = tb.Float32Col(pos=2)

class ResetPartial3dMapTable(tb.IsDescription):
    x           = tb.Float32Col(pos=0)
    y           = tb.Float32Col(pos=1)
    z           = tb.Float32Col(pos=2)
    value       = tb.Float32Col(pos=3)

# writers
def map_writer(hdf5_file, table_name, *, compression='ZLIB4'):
    map_table  = make_table(hdf5_file,
                            group       = 'ResetMap',
                            name        = table_name,
                            fformat     = ResetPartialMapTable,
                            description = 'Probability map',
                            compression = compression)

    def write_map(probs, xs, ys):
        for p, x, y in zip(probs, xs, ys):
            row = map_table.row
            row["x"    ] = x
            row["y"    ] = y
            row["value"] = p
            row.append()

    return write_map

# writers
def map3d_writer(hdf5_file, table_name, *, compression='ZLIB4'):
    map_table  = make_table(hdf5_file,
                            group       = 'ResetMap',
                            name        = table_name,
                            fformat     = ResetPartial3dMapTable,
                            description = 'Probability map',
                            compression = compression)

    def write_map(probs, xs, ys, zs):
        for p, x, y, z in zip(probs, xs, ys, zs):
            row = map_table.row
            row["x"    ] = x
            row["y"    ] = y
            row["z"    ] = z
            row["value"] = p
            row.append()

    return write_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-pmaps" , required=True)
    parser.add_argument("-o"     , required=True)
    parser.add_argument("-pmts"  , required=True)
    parser.add_argument("-sipms" , required=True)
    parser.add_argument("-zbins" , required=False)
    parser.add_argument("-ipmts" , action="store_true")
    parser.add_argument("-full3d", action="store_true")

    args = parser.parse_args(sys.argv[1:])
    pmaps_file = args.pmaps
    map_file   = args.o
    pmt_bins   = int(args.pmts)
    sipm_bins  = int(args.sipms)

    # Load pmaps
    pmaps = pmapio.load_pmaps(pmaps_file)
    hits_dict = load_mchits(pmaps_file)

    # Get sensor positions
    data_sipm = db.DataSiPM(0)
    data_pmt  = db.DataPMT (0)
    sipm_xs   = data_sipm.X
    sipm_ys   = data_sipm.Y
    pmt_xs    = data_pmt.X
    pmt_ys    = data_pmt.Y

    if not args.ipmts:
        positions, pmts, sipm_es, sipm_xs, sipm_ys = compute_position_and_charges(hits_dict, pmaps, sipm_xs, sipm_ys)
        xs = positions[:,0]
        ys = positions[:,1]
    else:
        pmts, pmt_xs, pmt_ys, sipm_es, sipm_xs, sipm_ys = compute_position_and_charges_ipmts(hits_dict, pmaps, pmt_xs, pmt_ys, sipm_xs, sipm_ys)

    with tb.open_file(map_file, 'w') as h5out:
        # PMT maps
        nbins = pmt_bins
        if not args.ipmts:
            if not args.full3d:
                range = [[-200, 200], [-200, 200]]
                xedges, yedges, values, counts = compute_histogram(xs, ys, pmts, nbins, range=range)
                write_pmt_values  = map_writer(h5out, 'pmt_values')
                write_pmt_counts  = map_writer(h5out, 'pmt_counts')
                write_pmt_values(values.flatten(), xedges, yedges)
                write_pmt_counts(counts.flatten(), xedges, yedges)
            else:
                range = [[-200, 200], [-200, 200], [0, 600]]
                bins  = [nbins, nbins, int(args.zbins)]
                xs, ys, zs, values, counts = compute_histogram3d(positions, pmts, bins, range)
                write_pmt_values  = map3d_writer(h5out, 'pmt_values')
                write_pmt_counts  = map3d_writer(h5out, 'pmt_counts')
                write_pmt_values(values.flatten(), xs, ys, zs)
                write_pmt_counts(counts.flatten(), xs, ys, zs)

        else:
            for i in np.arange(12):
                range = [[-200, 200], [-200, 200]]
                xedges, yedges, values, counts = compute_histogram(pmt_xs[:,i], pmt_ys[:,i], pmts[:,i], nbins, range=range)

                write_pmt_values  = map_writer(h5out, 'pmt{}_values'.format(i))
                write_pmt_counts  = map_writer(h5out, 'pmt{}_counts'.format(i))
                write_pmt_values(values.flatten(), xedges, yedges)
                write_pmt_counts(counts.flatten(), xedges, yedges)

        # SiPM maps
        if not args.full3d:
            nbins = sipm_bins
            range = [[-30, 30], [-30, 30]]
            xedges, yedges, values, counts = compute_histogram(sipm_xs, sipm_ys, sipm_es, nbins, range=range)
            write_sipm_values = map_writer(h5out, 'sipm_values')
            write_sipm_counts = map_writer(h5out, 'sipm_counts')
            write_sipm_values(values.flatten(), xedges, yedges)
            write_sipm_counts(counts.flatten(), xedges, yedges)
        else:
            bins = [sipm_bins, sipm_bins, int(args.zbins)]
            range = [[-30, 30], [-30, 30], [0, 600]]
            sipm_positions = (sipm_xs, sipm_ys, np.repeat(positions[:, 2], 1792))
            xs, ys, zs, values, counts = compute_histogram3d(sipm_positions, sipm_es, bins, range)
            write_sipm_values = map3d_writer(h5out, 'sipm_values')
            write_sipm_counts = map3d_writer(h5out, 'sipm_counts')
            write_sipm_values(values.flatten(), xs, ys, zs)
            write_sipm_counts(counts.flatten(), xs, ys, zs)
