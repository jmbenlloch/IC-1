"""
code: detsim.py
Simulation of sensor responses starting from Nexus output.

An HDF5 file containing Nexus output is given as input, and the simulated
detector response resulting from the Geant4 ionization tracks stored in this
file is produced.
"""

import numpy  as np
import tables as tb

from argparse import Namespace

from .. cities.base_cities           import City
from .. io.mcinfo_io                 import load_mchits

from .. io.pmaps_io                  import pmap_writer
from .. io.run_and_event_io          import run_and_event_writer
from .. io.mcinfo_io                 import mc_info_writer

from   .. evm.ic_containers  import S12Params
from   .. core.system_of_units_c import units
from   .. types.ic_types import minmax
import invisible_cities.reco.peak_functions as pkf
from .. evm.pmaps             import S1
from .. evm.pmaps             import S2
from .. evm.pmaps             import PMap



class Irenetoy(City):
    """Toy irene that will look for peaks from toy diomira"""

    parameters = tuple("""
        s1_tmin s1_tmax s1_stride s1_lmin s1_lmax s1_rebin_stride
        s2_tmin s2_tmax s2_stride s2_lmin s2_lmax s2_rebin_stride
        s1_e_thr s2_e_thr write_mc_info""".split())


    def __init__(self, **kwds):
        super().__init__(**kwds)
        conf = self.conf
        self.s1params = S12Params(time   = minmax(min = conf.s1_tmin,
                                                   max = conf.s1_tmax),
                                   stride              = conf.s1_stride,
                                   length = minmax(min = conf.s1_lmin,
                                                   max = conf.s1_lmax),
                                   rebin_stride        = conf.s1_rebin_stride)

        self.s2params = S12Params(time   = minmax(min = conf.s2_tmin,
                                                   max = conf.s2_tmax),
                                   stride              = conf.s2_stride,
                                   length = minmax(min = conf.s2_lmin,
                                                   max = conf.s2_lmax),
                                   rebin_stride        = conf.s2_rebin_stride)

        self.cnt.init(n_events_tot = 0)


    def file_loop(self):
        """
        The file loop of TDetSim:
        1. read the input Nexus files
        2. pass the hits to the event loop

        """
        for filename in self.input_files:
            with tb.open_file(filename, "r") as h5in:
                mc_info         = self.get_mc_info(h5in)
                nevts, pmtrwf, sipmrwf, _ = self.get_rwf_vectors(h5in)
                events_info  = self.get_run_and_event_info(h5in)
                self.event_loop(nevts, mc_info, pmtrwf, sipmrwf, events_info)


    def event_loop(self, nevts, mc_info, pmtrwf, sipmrwf, events_info):
        """
        The event loop of TDetSim:
        1. diffuse and apply energy smearing to all hits in each event
        2. create true voxels from the diffused/smeared hits

        """
        write = self.writers

        for evt in np.arange(nevts):
            evt_number, timestamp = self.event_and_timestamp(evt, events_info)
            print("Event {}".format(evt_number))

            pmts  = pmtrwf [evt]
            pmt_sum = pmts.sum(0)
            sipms = sipmrwf[evt]
            # Search S1
            s1_indx, s1_ene = pkf.indices_and_wf_above_threshold(pmt_sum, self.conf.s1_e_thr)
            s1s = pkf.find_peaks(pmts, s1_indx, self.s1params.time, self.s1params.length,
                                 self.s1params.stride, self.s1params.rebin_stride,
                                 Pk=S1, pmt_ids=range(12), sipm_wfs=None, thr_sipm_s2=0)
            # Search S2
            s2_indx, s2_ene = pkf.indices_and_wf_above_threshold(pmt_sum, self.conf.s2_e_thr)
            s2s = pkf.find_peaks(pmts, s2_indx, self.s2params.time, self.s2params.length,
                                 self.s2params.stride, self.s2params.rebin_stride,
                                 Pk=S2, pmt_ids=range(12), sipm_wfs=sipms, thr_sipm_s2=0)

            pmap = PMap(s1s, s2s)

            write.run_and_event(self.run_number, evt_number, timestamp)
            write.pmap         (pmap, evt_number)
            if self.monte_carlo:
                write.mc(mc_info, evt_number)

        self.cnt.n_events_tot += nevts


    def get_writers(self, h5out):
        writers = Namespace(
            run_and_event = run_and_event_writer(h5out),
            pmap          = pmap_writer(h5out),
            mc            = mc_info_writer(h5out) if self.monte_carlo else None,
        )
        return writers


    def write_parameters(self, h5out):
        pass
