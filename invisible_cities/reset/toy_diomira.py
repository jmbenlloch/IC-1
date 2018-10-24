import tables as tb
import numpy as np
import argparse
import sys
from functools import partial
from argparse  import Namespace
import matplotlib.pylab as plt
import invisible_cities.core.core_functions_c as core_c
import invisible_cities.reco.tbl_functions as tbl

from invisible_cities.io.rwf_io           import rwf_writer
from invisible_cities.io.mcinfo_io        import mc_info_writer
from invisible_cities.io.run_and_event_io import run_and_event_writer


def read_file(wf_file):
    h5in   = tb.open_file(wf_file)
    pmtrd  = h5in.root.pmtrd
    sipmrd = h5in.root.sipmrd
    events = h5in.root.Run.events
    mcinfo = tbl.get_mc_info(h5in)
    return pmtrd, sipmrd, events, mcinfo

def rebin_waveforms(waveforms, rebin_factor):
    nsensors = waveforms.shape[0]
    wf_len = int(waveforms.shape[1] / rebin_factor)
    waveforms_rebin = np.empty((nsensors, wf_len), dtype=np.int16)

    for sidx in np.arange(nsensors):
        wf = waveforms[sidx].astype('double')
        waveforms_rebin[sidx] = core_c.rebin_array(wf, rebin_factor)

    return waveforms_rebin

def get_writers(h5out, npmts, pmt_wl, nsipms, sipm_wl):
    RWF = partial(rwf_writer,  h5out,   group_name='RD')
    writers = Namespace(
        run_and_event = run_and_event_writer(h5out),
        mc            = mc_info_writer(h5out),
        pmt  = RWF(table_name='pmtrwf' , n_sensors=npmts , waveform_length=pmt_wl ),
        sipm = RWF(table_name='sipmrwf', n_sensors=nsipms, waveform_length=sipm_wl))

    return writers

def event_loop(writers, nevts, pmtrd, sipmrd, events, mcinfo, rebin_pmt):
    for evt in np.arange(nevts):
        print("Evt: ", evt)
        #pmts_wfs  = rebin_waveforms(pmtrd [evt], rebin_pmt)
        sipms_wfs = sipmrd[evt]

        #rebin pmts
        npmts = 12
        pmt_wl = 32000
        rebin_pmt = 25
        reshaped = pmtrd[evt].reshape((npmts, pmt_wl, rebin_pmt))
        pmts_wfs = np.apply_along_axis(np.sum, 2, reshaped).astype('int16')

        writers.pmt (pmts_wfs)
        writers.sipm(sipms_wfs)

        evtid, timestamp = events[evt]
        writers.mc(mcinfo, evtid)
        run_number = 0
        writers.run_and_event(run_number, evtid, timestamp)

        if evt > 3:
            break

def toy_diomira(wf_file, rwf_file):
    pmtrd, sipmrd, events, mcinfo = read_file(wf_file)
    rebin_pmt     = 25
    rebin_sipm    =  1

    nevts, npmts ,  pmt_wl =  pmtrd.shape
    nevts, nsipms, sipm_wl = sipmrd.shape
    pmt_wl  = int(pmt_wl  / rebin_pmt )
    sipm_wl = int(sipm_wl / rebin_sipm)

    h5out = tb.open_file(rwf_file, 'w')
    writers = get_writers(h5out, npmts, pmt_wl, nsipms, sipm_wl)

    event_loop(writers, nevts, pmtrd, sipmrd, events, mcinfo, rebin_pmt)

    h5out.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True)
    parser.add_argument("-o", required=True)
    args = parser.parse_args(sys.argv[1:])
    wf_file  = args.i
    rwf_file = args.o

    toy_diomira(wf_file, rwf_file)
