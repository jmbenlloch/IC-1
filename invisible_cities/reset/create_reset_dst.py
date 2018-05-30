from invisible_cities.reset.utils import produce_reset_dst
from invisible_cities.reco.dst_functions import load_lifetime_xy_corrections
from invisible_cities.io.voxels_io import dst_writer

import tables as tb
import sys

lt_corr_file = '/home/jmbenlloch/reset_data/mc/4735/corrections_MC_4734.h5'
file_in = ''
file_out = ''

def lt_corr_and_dst(file_in, file_out):
    zcorrection = load_lifetime_xy_corrections(lt_corr_file, group='XYcorrections', node='Lifetime')
    dst = produce_reset_dst(file_in, zcorrection)

    with tb.open_file(file_out, 'w') as h5out:
        write_dst = dst_writer(h5out)
        write_dst(dst)

if __name__ == '__main__':
    lt_corr_and_dst(sys.argv[1], sys.argv[2])

