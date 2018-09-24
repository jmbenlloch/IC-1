from argparse import Namespace
import os
import invisible_cities.reset.utils as rst_utils

import invisible_cities.reset.reset_gpu as rstf
import invisible_cities.reco.corrections    as corrf
import pdb

from time import time
import numpy as np

from .. io.voxels_io  import voxels_writer
from .  base_cities import PCity
from .  base_cities import ResetCity

class Reset(ResetCity):
    """Read PMAPS and produces a Reset Voxels"""

    def __init__(self, **kwds):
        super().__init__(**kwds)

    def get_writers(self, h5out):
        return Namespace(dst = voxels_writer(h5out),
                         mc  = self.get_mc_info_writer(h5out))

    def create_dst_event(self, pmapVectors, filter_output):
        s1_index = np.argmax(filter_output.s1_peaks)
        s2_index = np.argmax(filter_output.s2_peaks)

        s1 = pmapVectors.pmaps.s1s[s1_index]

        ZCorr = None
        if self.conf.lifetime_corr:
            ZCorr = corrf.LifetimeCorrection(self.conf.lifetime_value,
                                             self.conf.lifetime_error)

#        voxels_out = []
        voxels_dt  = np.dtype([('event', 'i4'), ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('E', 'f4')])
        voxels_out = np.empty(1000000, dtype=voxels_dt)
        voxel_idx  = 0

        # RESET has to be run for all the S2 in the event
        for s2_index in np.where(filter_output.s2_peaks)[0]:
            s2 = pmapVectors.pmaps.s2s[s2_index]

            t0 = time()
            reset_data   = rst_utils.prepare_data(s1, s2, self.conf.rebin_factor, self.DataSiPM,
                                              self.conf.nsipms, self.conf.sipm_thr,
                                              self.conf.dist, ZCorr)
            t1 = time()
            print("prepare_data: {}".format(t1-t0))

            # Check some data has passed all the filters
            if reset_data.slices.nslices == 0:
                continue

            slices_start = rst_utils.slices_start(reset_data.voxels_data,
                                                  self.conf.x_size,
                                                  self.conf.y_size)

            t0 = time()
            voxels, slices = self.rst.run(reset_data.voxels_data, reset_data.slices,
                                     reset_data.energies, slices_start,
                                     self.conf.iterations)
            t1 = time()
            print("run_reset: {}".format(t1-t0))

            t0 = time()
            print("voxels: ", len(voxels))
            for i, v in enumerate(voxels):
                # Filter voxels with zero energy
                if v[2] > 0:
                    #voxel = {}
                    #voxel['event'] = pmapVectors.events
                    #voxel['x']     = v[0]
                    #voxel['y']     = v[1]
                    #voxel['z']     = reset_data.zs[slices[i]]
                    #voxel['E']     = v[2]
                    #voxels_out.append(voxel)

                    voxel = voxels_out[voxel_idx]
                    voxel['event'] = pmapVectors.events
                    voxel['x']     = v[0]
                    voxel['y']     = v[1]
                    voxel['z']     = reset_data.zs[slices[i]]
                    voxel['E']     = v[2]
                    voxel_idx = voxel_idx + 1
            t1 = time()
            print("array_fill: {}".format(t1-t0))
#            import pdb
#            pdb.set_trace()

#        return voxels_out
        return voxels_out[:voxel_idx]

    def write_parameters(self, h5out):
        pass
