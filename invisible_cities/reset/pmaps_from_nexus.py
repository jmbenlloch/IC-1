import tables as tb
import sys
import argparse
import numpy as np
from argparse import Namespace
import invisible_cities.reco.tbl_functions as tbl
from invisible_cities.evm.ic_containers import S12Params
from invisible_cities.types.ic_types import minmax
import invisible_cities.reco.peak_functions as pkf

from invisible_cities.evm.pmaps             import S1
from invisible_cities.evm.pmaps             import S2
from invisible_cities.evm.pmaps import PMap

from invisible_cities.core.system_of_units_c import units

from invisible_cities.io.pmaps_io                  import pmap_writer
from invisible_cities.io.run_and_event_io          import run_and_event_writer
from invisible_cities.io.mcinfo_io import mc_info_writer

###########
# Paramas #
###########

run_number = 0
timestamp  = 0
s1_position = 100

s1_e_thr = 1
s2_e_thr = 10

s1_tmin   = 99 * units.mus # position of S1 in MC files at 100 mus
s1_tmax   = 101 * units.mus # change tmin and tmax if S1 not at 100 mus
s1_stride =   1       # minimum number of 25 ns bins in S1 searches
s1_lmin   =   1       # 8 x 25 = 200 ns
s1_lmax   =  20       # 20 x 25 = 500 ns
s1_rebin_stride = 1    # Do not rebin S1 by default


# Set parameters to search for S2
s2_tmin   =    101 * units.mus # assumes S1 at 100 mus, change if S1 not at 100 mus
s2_tmax   =    1300 * units.mus # end of the window
s2_stride =     40       #  40 x 25 = 1   mus
s2_lmin   =     80       # 40 x 25 = 1 mus
s2_lmax   = 200000       # maximum value of S2 width
s2_rebin_stride = 40 # Rebin by default


s1params = S12Params(time   = minmax(min = s1_tmin,
                                     max = s1_tmax),
                     stride              = s1_stride,
                     length = minmax(min = s1_lmin,
                                     max = s1_lmax),
                     rebin_stride        = s1_rebin_stride)

s2params = S12Params(time   = minmax(min = s2_tmin,
                                     max = s2_tmax),
                     stride              = s2_stride,
                     length = minmax(min = s2_lmin,
                                     max = s2_lmax),
                     rebin_stride = s2_rebin_stride)


def get_writers(h5out):
    writers = Namespace(
        run_and_event = run_and_event_writer(h5out),
        pmap          = pmap_writer(h5out),
        mc            = mc_info_writer(h5out))
    return writers


def sipm_id_to_index(sid):
    return (sid//1000-1) * 64 + sid % 1000


def create_waveforms(evt, idx, extents, sensors_data):
    start_position = 0 if idx == 0 else int(extents[idx-1][1])
    last_position  = int(extents[idx][1] + 1)

    sipms = np.zeros([1792, 1300])
    pmts  = np.zeros([12, 1300 * 40]) # hack to rebin to 25ns (find_peaks)

    data = sensors_data[start_position:last_position]

    for d in data:
        time = int(d[1] + s1_position)
        if d[0] > 100:
            sipms[sipm_id_to_index(d[0]), time] = d[2]
        else:
            pmts[d[0], time * 40 + 1] = d[2]

    return sipms, pmts


def process_file(events, sensors_data, extents, h5out, writers, mcinfo):
    for idx, evt in enumerate(events):
        print("Evt {}\t, eventid: {}".format(idx, evt))

        sipms, pmts = create_waveforms(evt, idx, extents, sensors_data)
        pmt_sum = pmts.sum(0)

        try:
            # Search S1
            s1_indx, s1_ene = pkf.indices_and_wf_above_threshold(pmt_sum, s1_e_thr)
            s1s = pkf.find_peaks(pmts, s1_indx, s1params.time, s1params.length,
                                 s1params.stride, s1params.rebin_stride,
                                 Pk=S1, pmt_ids=range(12), sipm_wfs=None, thr_sipm_s2=0)

            # Search S2
            s2_indx, s2_ene = pkf.indices_and_wf_above_threshold(pmt_sum, s2_e_thr)
            s2s = pkf.find_peaks(pmts, s2_indx, s2params.time, s2params.length,
                                 s2params.stride, s2params.rebin_stride,
                                 Pk=S2, pmt_ids=range(12), sipm_wfs=sipms, thr_sipm_s2=0)
        except IndexError:
            continue

        pmap = PMap(s1s, s2s)

        #break
        writers.mc(mcinfo, evt)
        writers.pmap (pmap, evt)
        writers.run_and_event(run_number, evt, timestamp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True)
    parser.add_argument("-o", required=True)

    args = parser.parse_args(sys.argv[1:])
    filein  = args.i
    fileout = args.o

    h5in = tb.open_file(filein)
    h5out = tb.open_file(fileout, 'w')

    writers = get_writers(h5out)
    mcinfo = tbl.get_mc_info(h5in)

    events = h5in.root.MC.events.cols.evt_number[:]
    sensors_data = h5in.root.MC.waveforms[:]
    extents = h5in.root.MC.extents[:]

    process_file(events, sensors_data, extents, h5out, writers, mcinfo)
    h5in.close()
    h5out.close()
