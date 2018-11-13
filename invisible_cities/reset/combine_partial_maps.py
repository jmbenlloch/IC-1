import tables as tb
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
from glob import glob
from invisible_cities.io.table_io import make_table


def combine_maps(files, pmt_bins, sipm_bins):
    pmts_values  = np.zeros(pmt_bins)
    pmts_counts  = np.zeros(pmt_bins)
    sipms_values = np.zeros(sipm_bins)
    sipms_counts = np.zeros(sipm_bins)

    for f in files:
        with tb.open_file(f) as h5in:
            pmts_xs     = h5in.root.ResetMap.pmt_counts.cols.x[:]
            pmts_ys     = h5in.root.ResetMap.pmt_counts.cols.y[:]
            pmts_counts = pmts_counts + h5in.root.ResetMap.pmt_counts.cols.value[:]
            pmts_values = pmts_values + h5in.root.ResetMap.pmt_values.cols.value[:]

            sipms_xs     = h5in.root.ResetMap.sipm_counts.cols.x[:]
            sipms_ys     = h5in.root.ResetMap.sipm_counts.cols.y[:]
            sipms_counts = sipms_counts + h5in.root.ResetMap.sipm_counts.cols.value[:]
            sipms_values = sipms_values + h5in.root.ResetMap.sipm_values.cols.value[:]

    pmts_counts [pmts_counts  == 0] = 1
    sipms_counts[sipms_counts == 0] = 1

    pmt_factors  = (pmts_values  / pmts_counts ).flatten()
    sipm_factors = (sipms_values / sipms_counts).flatten()

    return pmts_xs, pmts_ys, pmt_factors, sipms_xs, sipms_ys, sipm_factors

def get_nbins(files):
    pmt_bins, sipm_bins = 0, 0
    with tb.open_file(files[0]) as h5in:
        pmt_bins  = h5in.root.ResetMap.pmt_counts .shape[0]
        sipm_bins = h5in.root.ResetMap.sipm_counts.shape[0]

        pmt_bins  = int(np.sqrt(pmt_bins ))
        sipm_bins = int(np.sqrt(sipm_bins))

        return pmt_bins, sipm_bins

def make_plot(xs, ys, values, filename):
    nbins = np.unique(xs).shape[0]
    plt.figure(figsize=(10,8))
    plt.hist2d(xs, ys, weights=values, bins=nbins)
    plt.colorbar()
    plt.savefig(filename)


class ResetMapTable(tb.IsDescription):
    x           = tb.Float32Col(pos=0)
    y           = tb.Float32Col(pos=1)
    factor      = tb.Float32Col(pos=2)
    uncertainty = tb.Float32Col(pos=3)


def map_writer(hdf5_file, table_name, *, compression='ZLIB4'):
    map_table  = make_table(hdf5_file,
                            group       = 'ResetMap',
                            name        = table_name,
                            fformat     = ResetMapTable,
                            description = 'Probability map',
                            compression = compression)

    def write_map(probs, xs, ys):
        for p, x, y in zip(probs, xs, ys):
            row = map_table.row
            row["x"          ] = x
            row["y"          ] = y
            row["factor"     ] = p
            row["uncertainty"] = 0
            row.append()

    return write_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-maps", required=True)
    parser.add_argument("-o", required=True)
    args = parser.parse_args(sys.argv[1:])
    maps_path = args.maps
    map_file = args.o
    files = glob(maps_path + '/*h5')

    pmt_plot  = map_file + '_pmts.png'
    sipm_plot = map_file + '_sipms.png'

    pmt_bins, sipm_bins = get_nbins(files)

    pmts_xs, pmts_ys, pmt_values, sipms_xs, sipms_ys, sipm_values = combine_maps(files, pmt_bins, sipm_bins)

    make_plot(pmts_xs , pmts_ys , pmt_values , pmt_plot)
    make_plot(sipms_xs, sipms_ys, sipm_values, sipm_plot)

    with tb.open_file(map_file, 'w') as h5out:
        writer_pmt  = map_writer(h5out, 'PMT')
        writer_sipm = map_writer(h5out, 'SiPM')

        writer_pmt (pmt_values , pmts_xs , pmts_ys)
        writer_sipm(sipm_values, sipms_xs, sipms_ys)
