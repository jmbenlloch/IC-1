"""
-----------------------------------------------------------------------
                                 Hypathia
-----------------------------------------------------------------------

From ancient Greek ‘Υπατια: highest, supreme.

This city reads true waveforms from detsim and compute pmaps from them
without simulating the electronics. This includes:
    - Rebin 1-ns waveforms to 25-ns waveforms to match those produced
      by the detector.
    - Produce a PMT-summed waveform.
    - Apply a threshold to the PMT-summed waveform.
    - Find pulses in the PMT-summed waveform.
    - Match the time window of the PMT pulse with those in the SiPMs.
    - Build the PMap object.
"""
import numpy  as np
import tables as tb

from functools import partial

from .. database       import load_db

from .. reco                  import sensor_functions     as sf
from .. reco                  import tbl_functions        as tbl
from .. reco                  import peak_functions       as pkf
from .. core. random_sampling import NoiseSampler         as SiPMsNoiseSampler
from .. io  .        pmaps_io import          pmap_writer
from .. io  .       mcinfo_io import       mc_info_writer
from .. io  .run_and_event_io import run_and_event_writer
from .. io  .      trigger_io import       trigger_writer
from .. io  . event_filter_io import  event_filter_writer

from .. dataflow            import dataflow as fl
from .. dataflow.dataflow   import push
from .. dataflow.dataflow   import pipe
from .. dataflow.dataflow   import sink

from .  components import city
from .  components import print_every
from .  components import zero_suppress_wfs
from .  components import WfType
from .  components import sensor_data
from .  components import wf_from_files
from .  components import check_empty_pmap
from .  components import check_nonempty_indices
from .  components import get_number_of_active_pmts
from .  components import build_pmap
from .  components import compute_and_write_pmaps
from .  components import simulate_sipm_response
from .  components import calibrate_sipms


@city
def hypathia(files_in, file_out, compression, event_range, print_mod, detector_db, run_number,
          sipm_noise_cut, filter_padding, thr_sipm, thr_sipm_type, pmt_wfs_rebin, pmt_pe_rms,
          s1_lmin, s1_lmax, s1_tmin, s1_tmax, s1_rebin_stride, s1_stride, thr_csum_s1,
          s2_lmin, s2_lmax, s2_tmin, s2_tmax, s2_rebin_stride, s2_stride, thr_csum_s2, thr_sipm_s2):
    if   thr_sipm_type.lower() == "common":
        # In this case, the threshold is a value in pes
        sipm_thr = thr_sipm

    elif thr_sipm_type.lower() == "individual":
        # In this case, the threshold is a percentual value
        noise_sampler = SiPMsNoiseSampler(detector_db, run_number)
        sipm_thr      = noise_sampler.compute_thresholds(thr_sipm)

    else:
        raise ValueError(f"Unrecognized thr type: {thr_sipm_type}. "
                          "Only valid options are 'common' and 'individual'")

    #### Define data transformations
    sd = sensor_data(files_in[0], WfType.mcrd)

    # Raw WaveForm to Corrected WaveForm
    mcrd_to_rwf      = fl.map(rebin_pmts(pmt_wfs_rebin),
                              args = "pmt",
                              out  = "rwf")

    # Add single pe fluctuation to pmts
    simulate_pmt = fl.map(partial(sf.charge_fluctuation, single_pe_rms=pmt_pe_rms),
                          args = "rwf",
                          out = "ccwfs")

    # Compute pmt sum
    pmt_sum          = fl.map(pmts_sum, args = 'ccwfs',
                              out  = 'pmt')

    # Find where waveform is above threshold
    zero_suppress    = fl.map(zero_suppress_wfs(thr_csum_s1, thr_csum_s2),
                              args = ("pmt", "pmt"),
                              out  = ("s1_indices", "s2_indices", "s2_energies"))

    # SiPMs simulation
    simulate_sipm_response_  = fl.map(simulate_sipm_response(detector_db, run_number,
                                                             sd.SIPMWL, sipm_noise_cut,
                                                             filter_padding),
                                     args="sipm", out="sipm_sim")

    # Sipm calibration function expects waveform as int16
    discretize_signal = fl.map(lambda rwf: rwf.astype(np.int16),
                              args="sipm_sim", out="sipm")

    # SiPMs calibration
    sipm_rwf_to_cal  = fl.map(calibrate_sipms(detector_db, run_number, sipm_thr),
                              item = "sipm")

    event_count_in  = fl.spy_count()
    event_count_out = fl.spy_count()

    with tb.open_file(file_out, "w", filters = tbl.filters(compression)) as h5out:

        # Define writers...
        write_event_info_   = run_and_event_writer(h5out)
        write_mc_           = mc_info_writer      (h5out) if run_number <= 0 else (lambda *_: None)
        write_trigger_info_ = trigger_writer      (h5out, get_number_of_active_pmts(detector_db, run_number))

        # ... and make them sinks
        write_event_info   = sink(write_event_info_  , args=(   "run_number",     "event_number", "timestamp"   ))
        write_mc           = sink(write_mc_          , args=(           "mc",     "event_number"                ))
        write_trigger_info = sink(write_trigger_info_, args=( "trigger_type", "trigger_channels"                ))

        compute_pmaps, empty_indices, empty_pmaps = compute_and_write_pmaps(
                                             detector_db, run_number,
                                             s1_lmax, s1_lmin, s1_rebin_stride, s1_stride, s1_tmax, s1_tmin,
                                             s2_lmax, s2_lmin, s2_rebin_stride, s2_stride, s2_tmax, s2_tmin, thr_sipm_s2,
                                             h5out, compression, sipm_rwf_to_cal)

        return push(source = wf_from_files(files_in, WfType.mcrd),
                    pipe   = pipe(
                                fl.slice(*event_range, close_all=True),
                                print_every(print_mod),
                                event_count_in.spy,
                                mcrd_to_rwf,
                                simulate_pmt,
                                pmt_sum,
                                zero_suppress,
                                simulate_sipm_response_,
                                discretize_signal,
                                compute_pmaps,
                                event_count_out.spy,
                                fl.fork(write_mc,
                                        write_event_info,
                                        write_trigger_info)),
                    result = dict(events_in  = event_count_in .future,
                                  events_out = event_count_out.future,
                                  over_thr   = empty_indices  .future,
                                  full_pmap  = empty_pmaps    .future))


def rebin_pmts(rebin_stride):
    def rebin_pmts(rwf):
        # dummy data for times and widths
        times     = np.zeros(rwf.shape[1])
        widths    = times
        waveforms = rwf
        _, _, rebinned_wfs = pkf.rebin_times_and_waveforms(times, widths, waveforms, rebin_stride=rebin_stride)
        return rebinned_wfs
    return rebin_pmts


def pmts_sum(rwfs):
    return rwfs.sum(axis=0)

