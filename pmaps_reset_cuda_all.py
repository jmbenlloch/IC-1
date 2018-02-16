import numpy as np
import invisible_cities.io.pmap_io as pmapio
import invisible_cities.filters.s1s2_filter as s1s2filt
import invisible_cities.reco.pmaps_functions  as pmapsf
import invisible_cities.reco.corrections    as corrf
import invisible_cities.reco.pmaps_functions_c  as pmapsfc
from invisible_cities.core.system_of_units import pes, mm, mus, ns
#import reset_functions as rstf
#import reset_functions_compact as rstf
import reset_functions_event as rstf
import invisible_cities.database.load_db as dbf
from operator import itemgetter

import pdb

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

data_sipm = dbf.DataSiPM(run_number)
def create_voxels(data_sipm, sensor_ids, charges, dist):
    xmin = np.float32(data_sipm.X[sensor_ids].values.min()-dist)
    xmax = np.float32(data_sipm.X[sensor_ids].values.max()+dist)
    ymin = np.float32(data_sipm.Y[sensor_ids].values.min()-dist)
    ymax = np.float32(data_sipm.Y[sensor_ids].values.max()+dist)
    charge = np.float32(charges.mean())
    return xmin, xmax, ymin, ymax, charge

#TODO: check rounding here. Probably we are missing one row/column
def nvoxels(xmin, xmax, xsize, ymin, ymax, ysize):
    return (xmax - xmin)/ xsize * (ymax - ymin)/ ysize

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

    voxels_data = []
    sensors_ids = []
    charges = []
    slices_start_charges = []
    s2es = []
    for tbin, e in enumerate(s2[1]):
        slice_ = pmapsfc.sipm_ids_and_charges_in_slice(s2si, tbin)
        z = (np.average(s2[0][tbin], weights=s2[1][tbin]) - t0)/1000.
        charge = np.array(slice_[1]*ZCorr(z).value, dtype='f4')
        s2e = e * ZCorr(z).value
        selC = (charge > sipm_thr)
        if selC.any():
            voxels_data.append(create_voxels(data_sipm, slice_[0][selC], charge, dist))
            sensors_ids.append(slice_[0])
            charges.append(charge)
            slices_start_charges.append(charge.shape[0])
            s2es.append(s2e)

    sensors_ids = np.concatenate(sensors_ids).astype('i4')
    charges     = np.concatenate(charges)    .astype('f4')
    s2es        = np.array(s2es).astype('f4')
    #cumsum of slices start to get addresses
    slices_start_charges = np.array(slices_start_charges).cumsum()
    #Shift all elements one position to use them as indexes for mem
    slices_start_charges = np.concatenate(([0], slices_start_charges)).astype('i4')

    xmin   = np.array(list(map(itemgetter(0), voxels_data)), dtype='f4')
    xmax   = np.array(list(map(itemgetter(1), voxels_data)), dtype='f4')
    ymin   = np.array(list(map(itemgetter(2), voxels_data)), dtype='f4')
    ymax   = np.array(list(map(itemgetter(3), voxels_data)), dtype='f4')
    charge = np.array(list(map(itemgetter(4), voxels_data)), dtype='f4')

    nslices = xmin.shape[0]
    slices_start = np.empty(nslices)
    for i in range(nslices):
        slices_start[i] = nvoxels(xmin[i], xmax[i], x_size, ymin[i], ymax[i], y_size)
    slices_start = slices_start.cumsum()
    #Shift all elements one position to use them as indexes for mem
    slices_start = np.concatenate(([0], slices_start)).astype('i4')

    #reset.run(sensor_ids, charges, s2e, iterations)
    reset.run(xmin, xmax, ymin, ymax, charge, slices_start, iterations,
              sensors_ids, charges, slices_start_charges, s2es)

reset._destroy_context()

