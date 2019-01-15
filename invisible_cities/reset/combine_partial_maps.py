import tables as tb
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
from glob import glob
from invisible_cities.io.table_io import make_table


def combine_maps(files, pmt_bins, sipm_bins, norm=False):
    pmts_values  = np.zeros(pmt_bins)
    pmts_counts  = np.zeros(pmt_bins)
    sipms_values = np.zeros(sipm_bins)
    sipms_counts = np.zeros(sipm_bins)
    pmts_ivalues = np.zeros([pmt_bins, 12])
    pmts_icounts = np.zeros([pmt_bins, 12])
    ipmts = False
    pmts_xs = []
    pmts_ys = []
    pmts_zs = []
    first_file = True

    for i, f in enumerate(files):
        print(i, f)
        with tb.open_file(f) as h5in:
            if 'pmt_counts' in h5in.root.ResetMap:
                print("Pmt sum")
                if first_file:
                    pmts_xs.append(h5in.root.ResetMap.pmt_counts.cols.x[:])
                    pmts_ys.append(h5in.root.ResetMap.pmt_counts.cols.y[:])
                    try:
                        pmts_zs.append(h5in.root.ResetMap.pmt_counts.cols.z[:])
                    except:
                        pass
                pmts_counts = pmts_counts + h5in.root.ResetMap.pmt_counts.cols.value[:]
                pmts_values = pmts_values + h5in.root.ResetMap.pmt_values.cols.value[:]
            else:
                print("Individual pmts")
                ipmts = True
                for i in range(12):
                    counts_table = "pmt{}_counts".format(i)
                    values_table = "pmt{}_values".format(i)
                    if first_file:
                        pmts_xs.append(h5in.root.ResetMap[counts_table].cols.x[:])
                        pmts_ys.append(h5in.root.ResetMap[counts_table].cols.y[:])
                    pmts_icounts[:,i] = pmts_icounts[:,i] + h5in.root.ResetMap[counts_table].cols.value[:]
                    pmts_ivalues[:,i] = pmts_ivalues[:,i] + h5in.root.ResetMap[values_table].cols.value[:]

            sipms_xs     = h5in.root.ResetMap.sipm_counts.cols.x[:]
            sipms_ys     = h5in.root.ResetMap.sipm_counts.cols.y[:]
            sipms_zs     = []
            try:
                sipms_zs = h5in.root.ResetMap.sipm_counts.cols.z[:]
            except:
                pass
            sipms_counts = sipms_counts + h5in.root.ResetMap.sipm_counts.cols.value[:]
            sipms_values = sipms_values + h5in.root.ResetMap.sipm_values.cols.value[:]

            first_file = False

    sipms_counts[sipms_counts == 0] = 1
    sipm_factors = (sipms_values / sipms_counts).flatten()

    if ipmts:
        pmts_icounts [pmts_icounts  == 0] = 1
        pmt_factors  = pmts_ivalues / pmts_icounts
    else:
        pmts_counts [pmts_counts  == 0] = 1
        pmt_factors  = (pmts_values  / pmts_counts ).flatten()

    if norm:
        pmt_factors = pmt_factors.max() / pmt_factors
        pmt_factors[np.isinf(pmt_factors)] = 0

        sipm_factors = sipm_factors.max() / sipm_factors
        sipm_factors[np.isinf(sipm_factors)] = 0

    return pmts_xs, pmts_ys, pmts_zs, pmt_factors, sipms_xs, sipms_ys, sipms_zs, sipm_factors

def get_nbins(files):
    pmt_bins, sipm_bins = 0, 0
    with tb.open_file(files[0]) as h5in:
        if 'pmt_counts' in h5in.root.ResetMap:
            pmt_bins  = h5in.root.ResetMap.pmt_counts .shape[0]
        else:
            pmt_bins  = h5in.root.ResetMap.pmt0_counts .shape[0]
        sipm_bins = h5in.root.ResetMap.sipm_counts.shape[0]
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

class Reset3dMapTable(tb.IsDescription):
    x           = tb.Float32Col(pos=0)
    y           = tb.Float32Col(pos=1)
    z           = tb.Float32Col(pos=2)
    factor      = tb.Float32Col(pos=3)
    uncertainty = tb.Float32Col(pos=4)


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


def map3d_writer(hdf5_file, table_name, *, compression='ZLIB4'):
    map_table  = make_table(hdf5_file,
                            group       = 'ResetMap',
                            name        = table_name,
                            fformat     = Reset3dMapTable,
                            description = 'Probability map',
                            compression = compression)

    def write_map(probs, xs, ys, zs):
        for p, x, y, z in zip(probs, xs, ys, zs):
            row = map_table.row
            row["x"          ] = x
            row["y"          ] = y
            row["z"          ] = z
            row["factor"     ] = p
            row["uncertainty"] = 0
            row.append()

    return write_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-maps", required=True)
    parser.add_argument("-o", required=True)
    parser.add_argument("-norm", action="store_true")
    parser.add_argument("-full3d", action="store_true")

    args = parser.parse_args(sys.argv[1:])
    maps_path = args.maps
    map_file = args.o
    norm = args.norm

    files = glob(maps_path + '/*h5')


    pmt_bins, sipm_bins = get_nbins(files)

    pmts_xs, pmts_ys, pmts_zs, pmt_values, sipms_xs, sipms_ys, sipms_zs, sipm_values = combine_maps(files, pmt_bins, sipm_bins, norm)

    if len(pmts_xs) == 1: # individual pmts
        pmt_plot  = map_file + '_pmts.png'
        make_plot(pmts_xs[0] , pmts_ys[0] , pmt_values , pmt_plot)
    else:
        for i in range(12):
            pmt_plot  = map_file + '_pmt{}.png'.format(i)
            make_plot(pmts_xs[i] , pmts_ys[i] , pmt_values[:,i] , pmt_plot)

    sipm_plot = map_file + '_sipms.png'
    make_plot(sipms_xs, sipms_ys, sipm_values, sipm_plot)

    with tb.open_file(map_file, 'w') as h5out:
        if not args.full3d:
            writer_sipm = map_writer(h5out, 'SiPM')
            writer_sipm(sipm_values, sipms_xs, sipms_ys)
        else:
            writer_sipm  = map3d_writer(h5out, 'SiPM')
            writer_sipm(sipm_values , sipms_xs, sipms_ys, sipms_zs)

        if pmt_values.shape[-1] == 12: #ipmts
            for i in range(pmt_values.shape[-1]):
                writer_pmt = map_writer(h5out, 'PMT{}'.format(i))
                writer_pmt (pmt_values[:,i] , pmts_xs[i] , pmts_ys[i])
        else:
            if not args.full3d:
                writer_pmt  = map_writer(h5out, 'PMT')
                writer_pmt (pmt_values , pmts_xs[0] , pmts_ys[0])
            else:
                writer_pmt  = map3d_writer(h5out, 'PMT')
                writer_pmt (pmt_values , pmts_xs[0] , pmts_ys[0], pmts_zs[0])
