import invisible_cities.database.load_db as dbf
import invisible_cities.reco.dst_functions as dstf
import invisible_cities.io.pmaps_io as pio
import invisible_cities.reco.pmaps_functions  as pmapsf

import invisible_cities.reset.reset_ander    as rst_ander
import invisible_cities.reset.reset_ander_io as rst_io

from invisible_cities.icaro.hst_functions import hist2d
from invisible_cities.icaro.hst_functions import labels


import numpy as np
import tables as tb
import numba
import matplotlib.pylab as plt

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
iterations = 100
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
    write_reset = rst_io.reset_voxels_writer(h5out, 'RESET')
    write_lhood = rst_io.reset_lhood_writer(h5out, 'Likelihood')

    for evt, pmap in pmaps.items():
        if evt > 10:
            break
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
                    imageIter, lhood = rst_ander.MLEM_step(voxDX, voxDY, imageIter, selVox, selSens, anode_response, cath_response, pmt_prob, sipm_prob, sipm_dist=sipm_dist, eThres=e_thres, fCathode = fCathode, fAnode = fAnode)
                    write_lhood(evt, j, lhood)

                for vid in np.arange(imageIter.shape[1]):
                    write_reset(evt, imageIter[0, vid], imageIter[1, vid], z, imageIter[2, vid])
