import numpy as np
import invisible_cities.io.pmap_io as pmapio
import invisible_cities.filters.s1s2_filter as s1s2filt
import invisible_cities.reco.pmaps_functions  as pmapsf
import invisible_cities.reco.corrections    as corrf
import invisible_cities.reco.pmaps_functions_c  as pmapsfc
from invisible_cities.core.system_of_units import pes, mm, mus, ns
#import reset_functions as rstf
#import reset_functions_compact as rstf
import reset_functions_gpuarray as rstf

import time

# Initialize RESET
run_number = 4495
nsipms     = 1792
npmts      = 1
dist       = 20.
sipm_dist  = 20.
pmt_dist   = 200 # max distance included in the param file
sipm_thr   = 5.
x_size     = 2.
y_size     = 2.
rmax       = 198
pmt_param  = "/home/jmbenlloch/reset_data/mapas/PMT_Map_corr_keV.h5"
sipm_param = "/home/jmbenlloch/reset_data/mapas/SiPM_Map_corr_z0.0_keV.h5"

reset = rstf.RESET(run_number, nsipms, npmts, dist, sipm_dist, pmt_dist, sipm_thr, x_size, y_size, rmax, sipm_param, pmt_param)

#Lifetime correction
ZCorr = corrf.LifetimeCorrection(1093.77, 23.99)

def rebin_s2si(s2, s2si, rf):
    """given an s2 and a corresponding s2si, rebin them by a factor rf"""
    assert rf >= 1 and rf % 1 == 0
    t, e, sipms = pmapsf.rebin_s2si_peak(s2[0], s2[1], s2si, rf)
    s2d_rebin = [t, e]
    s2sid_rebin = sipms

    return s2d_rebin, s2sid_rebin

def rebin_s2pmt(s2pmt, stride):
    """rebin: s2 times (taking mean), s2 energies, and s2 sipm qs, by stride"""
    # cython rebin_array is returning memoryview so we need to cast as np array
    return   [corefc.rebin_array(s2pmt.t , stride, remainder=True, mean=True),               corefc.rebin_array(s2pmt.E , stride, remainder=True)]

pmap_conf = {}
pmap_conf['s1_emin'] = 54 * pes
pmap_conf['s1_emax'] = 2400. * pes
pmap_conf['s1_wmin'] = 7*25 * ns
pmap_conf['s1_wmax'] = 16*25 * ns
pmap_conf['s1_hmin'] = 12. * pes
pmap_conf['s1_hmax'] = 400. * pes
pmap_conf['s1_num'] = 1
pmap_conf['s1_ethr'] = 0.5 * pes

pmap_conf['s2_emin'] = 200000. * pes
pmap_conf['s2_emax'] = 2.e6 * pes
pmap_conf['s2_wmin'] = 4.* mus
pmap_conf['s2_wmax'] = 500. * mus
pmap_conf['s2_hmin'] = 600. * pes
pmap_conf['s2_hmax'] = 60000. * pes
pmap_conf['s2_numMin'] = 1
pmap_conf['s2_numMax'] = 3
pmap_conf['s2_ethr'] = 0. * pes
pmap_conf['s2_nsipmmin'] = 1
pmap_conf['s2_nsipmmax'] = 1792

def refresh_selector(param_val):
    selector = s1s2filt.S12Selector(s1_nmin = param_val['s1_num'], s1_nmax = param_val['s1_num'], s1_emin = param_val['s1_emin'], s1_emax = param_val['s1_emax'], s1_ethr = param_val['s1_ethr'],
                                    s1_wmin = param_val['s1_wmin'], s1_wmax = param_val['s1_wmax'], s1_hmin = param_val['s1_hmin'], s1_hmax = param_val['s1_hmax'],
                                    s2_nmin = param_val['s2_numMin'], s2_nmax = param_val['s2_numMax'], s2_emin = param_val['s2_emin'], s2_emax = param_val['s2_emax'],
                                    s2_wmin = param_val['s2_wmin'], s2_wmax = param_val['s2_wmax'], s2_hmin = param_val['s2_hmin'], s2_hmax = param_val['s2_hmax'],
                                    s2_nsipmmin = param_val['s2_nsipmmin'], s2_nsipmmax = param_val['s2_nsipmmax'], s2_ethr = param_val['s2_ethr'])

    return selector
select = refresh_selector(pmap_conf)

pmap_file = '/home/jmbenlloch/reset_data/4495/pmaps/pmaps.gdcsnext.189_4495.root.h5'
s1_file, s2_file, s2si_file, = pmapio.load_pmaps(pmap_file)

common_events = set(s1_file.keys()) & set(s2_file.keys()) & set(s2si_file.keys())
s1_all = dict({k:v for k,v in s1_file.items() if k in common_events})
s2_all = dict({k:v for k,v in s2_file.items() if k in common_events})
s2si_all =  dict({k:v for k,v in s2si_file.items() if k in common_events})

evt = 21215
s1_cut = select.select_valid_peaks(s1_all[evt], select.s1_ethr, select.s1e, select.s1w, select.s1h)
s2_cut = select.select_valid_peaks(s2_all[evt], select.s2_ethr, select.s2e, select.s2w, select.s2h)
s2si_cut = select.select_s2si(s2si_all[evt], select.nsi)

s1 = s1_all[evt].peak_waveform(0)
t0 = s1.t[np.argmax(s1.E)]

s2_cut   = [peakno for peakno, v in s2_cut.items() if v == True]
s2si_cut = [peakno for peakno, v in s2si_cut.items() if v == True]

valid_peaks = set(s2_cut) & set(s2si_cut)

slice_width = 2.
iterations = 100

for no in valid_peaks:
    s2 = s2si_all[evt].s2d[no]
    s2si = s2si_all[evt].s2sid[no]
    s2, s2si = rebin_s2si(s2, s2si, slice_width)

    for tbin, e in enumerate(s2[1]):
        #if tbin != 2:
#        if tbin != 7:
#        if tbin != 3:
#            continue
        print ("\n\nTime bin: {}".format(tbin))

        tstart = time.time()
        slice_ = pmapsfc.sipm_ids_and_charges_in_slice(s2si, tbin)
        tend = time.time()
        print("sipm_ids_and_charges_in_slice time: {}".format(tend-tstart))

        if len(slice_[0]) <= 0:
            continue

        tstart = time.time()
        z = (np.average(s2[0][tbin], weights=s2[1][tbin]) - t0)/1000.
        tend = time.time()
        print("Z compute time: {}".format(tend-tstart))

        if(z>550):
            continue

        tstart = time.time()
        sensor_ids = slice_[0].astype('i4')
        charges = np.array(slice_[1]*ZCorr(z).value, dtype='f4')
        s2e = e * ZCorr(z).value
        tend = time.time()
        print("castings: {}".format(tend-tstart))

        tstart = time.time()
        reset.run(sensor_ids, charges, s2e, iterations)
        tend = time.time()
        print("Reset time: {}".format(tend-tstart))

reset._destroy_context()

