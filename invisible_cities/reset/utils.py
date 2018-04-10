import invisible_cities.filters.s1s2_filter as s1s2filt
import invisible_cities.reco.corrections    as corrf
import invisible_cities.database.load_db as dbf
import invisible_cities.reco.pmaps_functions  as pmapsf
import invisible_cities.io.pmaps_io as pmapio

from invisible_cities.evm.ic_containers import VoxelsLimits
from invisible_cities.evm.ic_containers import ResetSlices
from invisible_cities.evm.ic_containers import SensorsParams
from invisible_cities.evm.ic_containers import ResetData


import numpy as np
import tables as tb

import pdb

def refresh_selector(param_val):
    selector = s1s2filt.S12Selector(s1_nmin = param_val['s1_num'],
                                    s1_nmax = param_val['s1_num'],
                                    s1_emin = param_val['s1_emin'],
                                    s1_emax = param_val['s1_emax'],
                                    s1_ethr = param_val['s1_ethr'],
                                    s1_wmin = param_val['s1_wmin'],
                                    s1_wmax = param_val['s1_wmax'],
                                    s1_hmin = param_val['s1_hmin'],
                                    s1_hmax = param_val['s1_hmax'],
                                    s2_nmin = param_val['s2_numMin'],
                                    s2_nmax = param_val['s2_numMax'],
                                    s2_emin = param_val['s2_emin'],
                                    s2_emax = param_val['s2_emax'],
                                    s2_wmin = param_val['s2_wmin'],
                                    s2_wmax = param_val['s2_wmax'],
                                    s2_hmin = param_val['s2_hmin'],
                                    s2_hmax = param_val['s2_hmax'],
                                    s2_nsipmmin = param_val['s2_nsipmmin'],
                                    s2_nsipmmax = param_val['s2_nsipmmax'],
                                    s2_ethr = param_val['s2_ethr'])

    return selector


#def load_and_select_peaks(pmap_file, evt, select):
#    s1_file, s2_file, s2si_file, = pmapio.load_pmaps(pmap_file)
#
#    common_events = set(s1_file.keys()) & set(s2_file.keys()) & set(s2si_file.keys())
#    s1_all = dict({k:v for k,v in s1_file.items() if k in common_events})
#    s2_all = dict({k:v for k,v in s2_file.items() if k in common_events})
#    s2si_all =  dict({k:v for k,v in s2si_file.items() if k in common_events})
#
#    s1_cut = select.select_valid_peaks(s1_all[evt], select.s1_ethr, select.s1e, select.s1w, select.s1h)
#    s2_cut = select.select_valid_peaks(s2_all[evt], select.s2_ethr, select.s2e, select.s2w, select.s2h)
#    s2si_cut = select.select_s2si(s2si_all[evt], select.nsi)
#
#    s2_cut   = [peakno for peakno, v in s2_cut.items() if v == True]
#    s2si_cut = [peakno for peakno, v in s2si_cut.items() if v == True]
#
#    valid_peaks = set(s2_cut) & set(s2si_cut)
#
#    return s1_all, s2_all, s2si_all, valid_peaks

def load_and_select_peaks(pmap_file, evt, select):
    pmaps = pmapio.load_pmaps(pmap_file)
    pmap = pmaps[evt]

    s1_cut = select.select_s1(pmaps[21215].s1s)
    s2_cut = select.select_s2(pmaps[21215].s2s)

    s1_index = np.argmax(s1_cut)
    s2_index = np.argmax(s2_cut)

    return pmap.s1s[s1_index], pmap.s2s[s2_index]

def create_voxels(data_sipm, sensor_ids, charges, dist):
    xmin = np.float32(data_sipm.X[sensor_ids].values.min()-dist)
    xmax = np.float32(data_sipm.X[sensor_ids].values.max()+dist)
    ymin = np.float32(data_sipm.Y[sensor_ids].values.min()-dist)
    ymax = np.float32(data_sipm.Y[sensor_ids].values.max()+dist)
    charge = np.float32(charges.mean())
    return xmin, xmax, ymin, ymax, charge


def nvoxels(xmin, xmax, xsize, ymin, ymax, ysize):
    return ((xmax - xmin)/ xsize + 1) * ((ymax - ymin)/ ysize + 1)


def rebin_s2si(s2, s2si, rf):
    """given an s2 and a corresponding s2si, rebin them by a factor rf"""
    assert rf >= 1 and rf % 1 == 0
    t, e, sipms = pmapsf.rebin_s2si_peak(s2[0], s2[1], s2si, rf)
    s2d_rebin = [t, e]
    s2sid_rebin = sipms

    return s2d_rebin, s2sid_rebin

def prepare_data(s1, s2, slice_width, evt, data_sipm,
                 nsipms, sipm_thr, dist, zcorrection, stop_slice=1e6):
    #Rebin data
    s2_rebin = pmapsf.rebin_peak(s2, slice_width)

    #Get time
    t0 = s1.time_at_max_energy

    #Alloc mem
    max_slices = len(s2_rebin.times)

    charges    = np.empty((max_slices * nsipms), dtype='f4')
    sensor_ids = np.empty((max_slices * nsipms), dtype='i4')

    xmins   = np.empty((max_slices), dtype='f4')
    xmaxs   = np.empty((max_slices), dtype='f4')
    ymins   = np.empty((max_slices), dtype='f4')
    ymaxs   = np.empty((max_slices), dtype='f4')
    avg_charges = np.empty((max_slices), dtype='f4')

    zs           = np.empty((max_slices), dtype='f4')
    energies     = np.empty((max_slices), dtype='f4')
    slices_start = np.zeros((max_slices+1), dtype='i4')

    # Fill the arrays
    nsensors = 0
    nslices  = 0
    #for tbin, e in enumerate(s2[1]):
    for tbin, t in enumerate(s2_rebin.times):
        #Stop condition for testing purposes
        if tbin > stop_slice:
            break

        # Apply lifetime correction
        z       = (s2_rebin.times[tbin] - t0) / 1000.
        charge  = s2_rebin.sipms.time_slice(0) * zcorrection(z).value
        selC    = (charge > sipm_thr)
        charge  = charge[selC]
        s2e     = s2_rebin.total_energy * zcorrection(z).value
        ids     = s2_rebin.sipms.ids[selC]
        sensors = selC.sum()

        if selC.any():
            xmin, xmax, ymin, ymax, avg_charge = create_voxels(data_sipm, ids, charge, dist)
            charges   [nsensors:nsensors+sensors] = charge
            sensor_ids[nsensors:nsensors+sensors] = ids

            xmins[nslices] = xmin
            xmaxs[nslices] = xmax
            ymins[nslices] = ymin
            ymaxs[nslices] = ymax
            avg_charges[nslices] = avg_charge

            slices_start[nslices+1] = slices_start[nslices] + sensors
            energies    [nslices]   = s2e
            zs          [nslices]   = z

            nslices  = nslices  + 1
            nsensors = nsensors + sensors

    voxels = VoxelsLimits(nslices,
                    xmins      [:nslices], xmaxs[:nslices],
                    ymins      [:nslices], ymaxs[:nslices],
                    avg_charges[:nslices])

    slices = ResetSlices(nslices,
                         nsipms,
                         nsensors,
                         slices_start[:nslices+1],
                         sensor_ids  [:nsensors],
                         charges     [:nsensors])

    data = ResetData(voxels, slices, energies, zs)
    return data


def slices_start(voxels, xsize, ysize):
    nslices = voxels.xmin.shape[0]
    # One extra position to the right to have the ending of the last slice
    slices_start = np.zeros(nslices+1)
    for i in range(nslices):
        slices_start[i+1] = nvoxels(voxels.xmin[i], voxels.xmax[i], xsize,
                                    voxels.ymin[i], voxels.ymax[i], ysize)
    slices_start = slices_start.cumsum().astype('i4')
    return slices_start

import math
from invisible_cities.evm.ic_containers import ResetRatios

def compute_sipm_ratio(sipm_dist, pitch, xsize, ysize):
    sipms_per_voxel = int(math.floor(2 * sipm_dist / pitch) + 1)**2
    voxels_per_sipm = int((2 * sipm_dist)**2 / (xsize * ysize))

    ratios = ResetRatios(sipms_per_voxel, voxels_per_sipm)
    return ratios


def compute_pmt_ratio(pmt_dist, npmts, xsize, ysize):
    pmts_per_voxel = npmts
    voxels_per_pmt = int((2 * pmt_dist)**2 / (xsize * ysize))

    ratios = ResetRatios(pmts_per_voxel, voxels_per_pmt)
    return ratios

class Voxel(tb.IsDescription):
    x = tb.Float32Col()
    y = tb.Float32Col()
    z = tb.Float32Col()
    E = tb.Float32Col()


def write_hdf5(voxels, slices, zs):
    h5file = tb.open_file("reset.h5", mode="w", title="voxels")
    group = h5file.create_group("/", 'RESET', 'voxels')
    table = h5file.create_table(group, 'voxels', Voxel, "voxels")

    voxel = table.row
    for i, v in enumerate(voxels):
        voxel['x'] = v[0]
        voxel['y'] = v[1]
        voxel['z'] = zs[slices[i]]
        voxel['E'] = v[2]
        voxel.append()
        table.flush()

def read_corrections_file(filename, node):
    corr_h5 = tb.open_file(filename)
    corr_table = getattr(corr_h5.root.ResetMap, node)
    corrections_dt = np.dtype([('x', 'f4'), ('y', 'f4'), ('factor', 'f4')])

    # we need to explicitly build it to get into memory only (x,y,factor)
    # to check: struct.unpack('f', bytes(pmts_corr.data)[i*4:(i+1)*4])
    params = np.array(list(zip(corr_table.col('x'),
                               corr_table.col('y'),
                               corr_table.col('factor'))),
                      dtype=corrections_dt)

    step  =  params[1][1] - params[0][1]
    xmin  =  params[0][0]
    ymin  =  params[0][1]

    nbins = (params[-1][0] - params[0][0]) / step + 1
    nbins = nbins.astype('i4')
    corr_h5.close()

    return SensorsParams(xmin, ymin, step, nbins, params)

