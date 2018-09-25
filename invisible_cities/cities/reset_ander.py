from argparse import Namespace
import invisible_cities.reco.dst_functions as dstf
import os

import invisible_cities.reset.reset_ander    as rst_ander
import invisible_cities.reset.reset_ander_io as rst_io
import invisible_cities.reco.pmaps_functions  as pmapsf

import invisible_cities.reco.corrections    as corrf
import pdb

from time import time
import numpy as np

from .. reset.reset_ander_io  import reset_lhood_writer
from .. reset.reset_ander_io  import reset_voxels_writer
from .  base_cities import PCity
from .  base_cities import ResetCity

class Reset_ander(ResetCity):
    """Read PMAPS and produces a Reset Voxels"""

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.sipm_xy_map = dstf.load_xy_corrections(self.conf.sipm_param, group="ResetMap", node="SiPM")
        self.pmt_xy_map  = dstf.load_xy_corrections(self.conf.pmt_param,  group="ResetMap", node="PMT")

    def get_writers(self, h5out):
        return Namespace(dst = self.write_parameters,
                         lhood = reset_lhood_writer(h5out, 'Likelihood'),
                         voxels = reset_voxels_writer(h5out, 'RESET'),
                         mc  = self.get_mc_info_writer(h5out))

    def create_dst_event(self, pmapVectors, filter_output):
        s1_index = np.argmax(filter_output.s1_peaks)
        s1 = pmapVectors.pmaps.s1s[s1_index]

        t0  = s1.time_at_max_energy

        for s2_index in np.where(filter_output.s2_peaks)[0]:
            s2 = pmapVectors.pmaps.s2s[s2_index]
            s2_rebin = pmapsf.rebin_peak(s2, self.conf.rebin_factor)
            for tbin, t in enumerate(s2_rebin.times):
                z = (s2_rebin.times[tbin] - t0) / 1000.
                s2e = s2_rebin.pmts.sum_over_sensors[tbin]

                charge  = s2_rebin.sipms.time_slice(tbin)
                # For some reason there are empty slices in tdetsim pmaps...
                if charge.sum() == 0:
                    continue
                ids     = s2_rebin.sipms.ids

                vox = rst_ander.CreateVoxels(self.DataSiPM, ids, charge, self.conf.sipm_dist,
                                             self.conf.sipm_thr, self.conf.x_size,
                                             self.conf.y_size, self.conf.rmax)
                anode_response = rst_ander.CreateSiPMresponse(self.DataSiPM, ids, charge,
                                                              self.conf.sipm_dist, self.conf.sipm_thr, vox)
                cath_response = np.array([s2e])
                voxDX, voxDY = rst_ander.computeDiff(self.DataSiPM, vox, anode_response)
                selVox, selSens = rst_ander.createSel(voxDX, voxDY, anode_response, self.conf.sipm_dist)
                sipm_prob, pmt_prob = rst_ander.computeProb(self.pmt_xy_map, self.sipm_xy_map,
                                                            voxDX, voxDY, vox[0], vox[1])
                imageIter = vox

                for j in range(self.conf.iterations):
                    imageIter, lhood = rst_ander.MLEM_step(voxDX, voxDY, imageIter, selVox, selSens, anode_response, cath_response, pmt_prob, sipm_prob, sipm_dist=self.conf.sipm_dist, eThres=0, fCathode = self.conf.use_pmts, fAnode = self.conf.use_sipms)
                    self.writers.lhood(self.evt_number, j, lhood)

                for vid in np.arange(imageIter.shape[1]):
                    self.writers.voxels(self.evt_number, imageIter[0, vid], imageIter[1, vid], z, imageIter[2, vid])

        return []

    def write_parameters(self, h5out):
        pass
