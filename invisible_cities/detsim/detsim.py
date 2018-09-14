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

from .. detsim.detsim_functions      import simulate_sensors
from .. detsim.detsim_functions      import sipm_lcone
from .. detsim.detsim_functions      import pmt_lcone

class Detsim(City):
    """Simulates detector response for events produced by Nexus"""

    parameters = tuple("""zmin
      A_sipm d_sipm
      ze_sipm ze_pmt slice_width_sipm E_to_Q_sipm uniformlight_frac_sipm
      s2_threshold_sipm slice_width_pmt E_to_Q_pmt uniformlight_frac_pmt
      s2_threshold_pmt peak_space""".split())

    def __init__(self, **kwds):
        """actions:
        1. inits base city
        2. inits event counter

        """
        super().__init__(**kwds)

        self.light_function_sipm = sipm_lcone(self.conf.A_sipm,
                                              self.conf.d_sipm,
                                              self.conf.ze_sipm)
        self.light_function_pmt  = pmt_lcone (self.conf.ze_pmt)

        self.cnt.init(n_events_tot = 0)


    def file_loop(self):
        """
        The file loop of TDetSim:
        1. read the input Nexus files
        2. pass the hits to the event loop

        """
        for filename in self.input_files:
            mchits_dict = load_mchits(filename, self.conf.event_range)
            with tb.open_file(filename, "r") as h5in:
                mc_info     = self.get_mc_info(h5in)
                self.event_loop(mchits_dict, mc_info)


    def event_loop(self, mchits_dict, mc_info):
        """
        The event loop of TDetSim:
        1. diffuse and apply energy smearing to all hits in each event
        2. create true voxels from the diffused/smeared hits

        """
        write = self.writers

        for evt_number, mchits in mchits_dict.items():

            print("Event {}".format(evt_number))
            pmap = simulate_sensors(mchits, self.DataSiPM, self.conf.slice_width_sipm,
                                    self.light_function_sipm, self.conf.E_to_Q_sipm,
                                    self.conf.s2_threshold_sipm,
                                    self.DataPMT, self.conf.slice_width_pmt,
                                    self.light_function_pmt, self.conf.E_to_Q_pmt,
                                    self.conf.s2_threshold_pmt, self.conf.peak_space)

            write.run_and_event(self.run_number, evt_number, 0)
            write.pmap         (pmap, evt_number)
            if self.monte_carlo:
                write.mc(mc_info, evt_number)

        self.cnt.n_events_tot += len(mchits_dict)


    def get_writers(self, h5out):
        writers = Namespace(
            run_and_event = run_and_event_writer(h5out),
            pmap          = pmap_writer(h5out),
            mc            = mc_info_writer(h5out) if self.monte_carlo else None,
        )
        return writers


    def write_parameters(self, h5out):
        pass
