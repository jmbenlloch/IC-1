import invisible_cities.filters.s1s2_filter as s1s2filt
import invisible_cities.reco.pmaps_functions_c  as pmapsfc
import invisible_cities.reco.corrections    as corrf
import invisible_cities.database.load_db as dbf
import invisible_cities.reco.pmaps_functions  as pmapsf
import invisible_cities.io.pmap_io as pmapio

from invisible_cities.evm.ic_containers import Voxels
from invisible_cities.evm.ic_containers import ResetSlices

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


def load_and_select_peaks(pmap_file, evt, select):
    s1_file, s2_file, s2si_file, = pmapio.load_pmaps(pmap_file)

    common_events = set(s1_file.keys()) & set(s2_file.keys()) & set(s2si_file.keys())
    s1_all = dict({k:v for k,v in s1_file.items() if k in common_events})
    s2_all = dict({k:v for k,v in s2_file.items() if k in common_events})
    s2si_all =  dict({k:v for k,v in s2si_file.items() if k in common_events})

    s1_cut = select.select_valid_peaks(s1_all[evt], select.s1_ethr, select.s1e, select.s1w, select.s1h)
    s2_cut = select.select_valid_peaks(s2_all[evt], select.s2_ethr, select.s2e, select.s2w, select.s2h)
    s2si_cut = select.select_s2si(s2si_all[evt], select.nsi)

    s2_cut   = [peakno for peakno, v in s2_cut.items() if v == True]
    s2si_cut = [peakno for peakno, v in s2si_cut.items() if v == True]

    valid_peaks = set(s2_cut) & set(s2si_cut)

    return s1_all, s2_all, s2si_all, valid_peaks


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

def prepare_data(s1s, s2s, s2sis, slice_width, evt, peak, data_sipm,
                 nsipms, sipm_thr, dist, zcorrection):
    #Rebin data
    s2 = s2sis[evt].s2d[peak]
    s2si = s2sis[evt].s2sid[peak]
    s2, s2si = rebin_s2si(s2, s2si, slice_width)

    #Get time
    s1 = s1s[evt].peak_waveform(peak)
    t0 = s1.t[np.argmax(s1.E)]

    #Alloc mem
    max_slices = s2[1].shape[0]

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
    for tbin, e in enumerate(s2[1]):
        slice_ = pmapsfc.sipm_ids_and_charges_in_slice(s2si, tbin)
        sensors = slice_[0].shape[0]
        z      = (np.average(s2[0][tbin], weights=s2[1][tbin]) - t0)/1000.

        # Apply lifetime correction
        charge = slice_[1] * zcorrection(z).value
        s2e    = e * zcorrection(z).value
        selC = (charge > sipm_thr)

        if selC.any():
            xmin, xmax, ymin, ymax, avg_charge = create_voxels(data_sipm, slice_[0][selC], charge, dist)
            charges   [nsensors:nsensors+sensors] = charge
            sensor_ids[nsensors:nsensors+sensors] = slice_[0]

            xmins[nslices] = xmin
            xmaxs[nslices] = xmax
            ymins[nslices] = ymin
            ymaxs[nslices] = ymax
            avg_charges[nslices] = avg_charge

            slices_start[nslices+1] = slices_start[nslices] + charge.shape[0]
            energies    [nslices]   = s2e
            zs          [nslices]   = z

            nslices  = nslices  + 1
            nsensors = nsensors + sensors

        #TODO: REMOVE!!!
        if tbin > 4:
            break

    voxels = Voxels(xmins      [:nslices], xmaxs[:nslices],
                    ymins      [:nslices], ymaxs[:nslices],
                    avg_charges[:nslices])

    slices = ResetSlices(slices_start[:nslices+1],
                         sensor_ids  [:nsensors],
                         charges     [:nsensors])

    return voxels, slices, energies, zs


def slices_start(voxels, xsize, ysize):
    nslices = voxels.xmin.shape[0]
    # One extra position to the right to have the ending of the last slice
    slices_start = np.zeros(nslices+1)
    for i in range(nslices):
        slices_start[i+1] = nvoxels(voxels.xmin[i], voxels.xmax[i], xsize,
                                    voxels.ymin[i], voxels.ymax[i], ysize)
    slices_start = slices_start.cumsum().astype('i4')
    return slices_start


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

