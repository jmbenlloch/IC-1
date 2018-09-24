import invisible_cities.database.load_db as dbf
import invisible_cities.reco.dst_functions as dstf
import invisible_cities.io.pmaps_io as pio
import invisible_cities.reco.pmaps_functions  as pmapsf

import invisible_cities.reset.reset_ander as rst_ander

from invisible_cities.icaro.hst_functions import hist2d
from invisible_cities.icaro.hst_functions import labels
from invisible_cities.io.table_io import make_table


import numpy as np
import tables as tb
import numba
import matplotlib.pylab as plt


# # Writer
class ResetDST(tb.IsDescription):
    evt = tb.UInt32Col (pos=0)
    x   = tb.Float64Col(pos=1)
    y   = tb.Float64Col(pos=2)
    z   = tb.Float64Col(pos=3)
    E   = tb.Float64Col(pos=4)
    #Iteration = tb.UInt32Col(pos=5)


def map_writer(hdf5_file, table_name, *, compression='ZLIB4'):
    map_table  = make_table(hdf5_file,
                            group       = 'ResetMap',
                            name        = table_name,
                            fformat     = ResetDST,
                            description = 'Reset dst',
                            compression = compression)

    def write_map(evt, x, y, z, E):
        row = map_table.row
        row["evt"] = evt
        row["x"  ] = x
        row["y"  ] = y
        row["z"  ] = z
        row["E"  ] = E
        row.append()

    return write_map



##########
# Params #
##########
run_number = -4374
slice_width = 2
rMax = 198
sizeX = 10
sizeY = 10
sipm_thr = 0
point_dist = 20
sipm_dist = 20
iterations = 10
fCathode = True
fAnode = True
e_thres = 0

param_file  = "/home/jmbenlloch/reset/toy/data/reset_map_ic.h5"
input_file  = '/home/jmbenlloch/reset/toy/data/detsim_test.h5'
output_file = 'reset_ander.h5'

############
DataSiPM = dbf.DataSiPM(run_number)
DataPMT = dbf.DataPMT(run_number)

pmaps = pio.load_pmaps(input_file)

sipm_xy_map = dstf.load_xy_corrections(param_file, group="ResetMap", node="SiPM")
pmt_xy_map  = dstf.load_xy_corrections(param_file, group="ResetMap", node="PMT")

with tb.open_file(output_file, 'w') as h5out:
    write_reset = map_writer(h5out, 'RESET')

    for evt, pmap in pmaps.items():
#        if evt > 500:
#            break
        print(evt)

        s1 = pmap.s1s[0]
        for s2 in pmap.s2s:
            s2_rebin = pmapsf.rebin_peak(s2, slice_width)
            #Get time
            t0  = s1.time_at_max_energy

            for tbin, t in enumerate(s2_rebin.times):
                z = (s2_rebin.times[tbin] - t0) / 1000.
                s2e = s2_rebin.pmts.sum_over_sensors[tbin]

                charge  = s2_rebin.sipms.time_slice(tbin)
                # For some reason there are empty slices in tdetsim pmaps...
                if charge.sum() == 0:
                    continue
                ids     = s2_rebin.sipms.ids

                vox = rst_ander.CreateVoxels(DataSiPM, ids, charge, point_dist, sipm_thr, sizeX, sizeY, rMax)

                anode_response = rst_ander.CreateSiPMresponse(DataSiPM, ids, charge, sipm_dist, sipm_thr, vox)

                cath_response = np.array([s2e])

                voxDX, voxDY = rst_ander.computeDiff(DataSiPM, vox, anode_response)

                selVox, selSens = rst_ander.createSel(voxDX, voxDY, anode_response, sipm_dist)

                sipm_prob, pmt_prob = rst_ander.computeProb(pmt_xy_map, sipm_xy_map, voxDX, voxDY, vox[0], vox[1])

                imageIter = vox

                for j in range(iterations):
                    imageIter = rst_ander.MLEM_step(voxDX, voxDY, imageIter, selVox, selSens, anode_response, cath_response, pmt_prob, sipm_prob, sipm_dist=sipm_dist, eThres=e_thres, fCathode = fCathode, fAnode = fAnode)

                for vid in np.arange(imageIter.shape[1]):
                    write_reset(evt, imageIter[0, vid], imageIter[1, vid], z, imageIter[2, vid])
