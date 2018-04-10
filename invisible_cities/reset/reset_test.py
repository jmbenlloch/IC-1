import math
import invisible_cities.reset.reset_gpu as rstf
import invisible_cities.reset.reset_serial as rst_serial
import invisible_cities.reset.memory as rst_mem
import pycuda.driver as cuda
import pycuda

from pytest                 import fixture
import os

import invisible_cities.reset.utils as rst_util
import numpy as np
from invisible_cities.core.system_of_units import pes, mm, mus, ns
import invisible_cities.reco.corrections    as corrf
import invisible_cities.database.load_db as dbf

from invisible_cities.evm.ic_containers import ResetTest
from invisible_cities.evm.ic_containers import ResetVoxels

# Initialize RESET
iterations = 100
pitch      = 10
run_number = 4495
nsipms     = 1792
npmts      = 1
dist       = 20.
sipm_dist  = 20.
pmt_dist   = 200 # max distance included in the param file
sipm_thr   = 5.
x_size     = 10.
y_size     = 10.
rmax       = 198
slice_width = 2.


pmt_param_file  = os.path.expandvars('$ICDIR/database/test_data/reset_pmt_map.h5')
sipm_param_file = os.path.expandvars('$ICDIR/database/test_data/reset_sipm_map.h5')
pmap_file = os.path.expandvars('$ICDIR/database/test_data/reset_4495_21215.h5')

@fixture(scope="session")
def pmap_conf():
    conf = {}
    conf['s1_emin'] = 54 * pes
    conf['s1_emax'] = 2400. * pes
    conf['s1_wmin'] = 7*25 * ns
    conf['s1_wmax'] = 16*25 * ns
    conf['s1_hmin'] = 12. * pes
    conf['s1_hmax'] = 400. * pes
    conf['s1_num'] = 1
    conf['s1_ethr'] = 0.5 * pes
    conf['s2_emin'] = 200000. * pes
    conf['s2_emax'] = 2.e6 * pes
    conf['s2_wmin'] = 4.* mus
    conf['s2_wmax'] = 500. * mus
    conf['s2_hmin'] = 600. * pes
    conf['s2_hmax'] = 60000. * pes
    conf['s2_numMin'] = 1
    conf['s2_numMax'] = 3
    conf['s2_ethr'] = 0. * pes
    conf['s2_nsipmmin'] = 1
    conf['s2_nsipmmax'] = 1792

    return conf

@fixture(scope="session")
def data_sipm():
    return dbf.DataSiPM(run_number)

# Prepare data
@fixture(scope="session")
def reset_data(pmap_conf, data_sipm):
    # Read file and select peaks
    selector = rst_util.refresh_selector(pmap_conf)
    s1s, s2s, s2sis, peaks = rst_util.load_and_select_peaks(pmap_file, 21215, selector)

    # Lifetime correction
    ZCorr = corrf.LifetimeCorrection(1093.77, 23.99)

    # Take the first peak
    peak = next(iter(peaks))
    evt = 21215
    total_slices = 4
    return rst_util.prepare_data(s1s, s2s, s2sis, slice_width, evt, peak, data_sipm, nsipms, sipm_thr, dist, ZCorr, total_slices)

@fixture(scope="session")
def slices_start(reset_data):
    return rst_util.slices_start(reset_data.voxels_data, x_size, y_size)

##########
# serial #
##########

@fixture(scope="session")
def serial_version(reset_data, slices_start, data_sipm):
    voxels_data = reset_data.voxels_data
    slices = reset_data.slices
    energies = reset_data.energies

    rst_voxels = rst_serial.create_voxels(voxels_data, slices_start, x_size, y_size, rmax)
    rst_voxels_orig = ResetVoxels(rst_voxels.nslices,
                                  rst_voxels.nvoxels,
                                  rst_voxels.voxels.copy(),
                                  rst_voxels.slice_ids,
                                  rst_voxels.slice_start,
                                  rst_voxels.address)

    anode_response = rst_serial.create_anode_response(slices)
    cath_response = energies

    sipm_params = rst_util.read_corrections_file(sipm_param_file, 'SiPM')
    sipm_ratios = rst_util.compute_sipm_ratio(sipm_dist, pitch, x_size, y_size)
    xs = data_sipm.X
    ys = data_sipm.Y

#    return rst_voxels

    sipm_probs = rst_serial.compute_probabilities(rst_voxels, xs, ys, nsipms, sipm_ratios.sns_per_voxel, sipm_dist, sipm_params, anode_response)

    pmt_params = rst_util.read_corrections_file(pmt_param_file, 'PMT')
    pmt_ratios = rst_util.compute_pmt_ratio(pmt_dist, npmts, x_size, y_size)

    xs_pmts = np.array([0.], dtype='f4')
    ys_pmts = np.array([0.], dtype='f4')

    pmt_probs = rst_serial.compute_probabilities(rst_voxels, xs_pmts, ys_pmts, npmts, pmt_ratios.sns_per_voxel, pmt_dist, pmt_params, cath_response)

    sipm_sns_probs = rst_serial.compute_sensor_probs(sipm_probs, rst_voxels.nslices, nsipms, rst_voxels.slice_ids)
    pmt_sns_probs = rst_serial.compute_sensor_probs(pmt_probs, rst_voxels.nslices, npmts, rst_voxels.slice_ids)

    sipm_fwd_denom = rst_serial.forward_denoms(nsipms, rst_voxels.nslices, rst_voxels.voxels, sipm_sns_probs)
    pmt_fwd_denom = rst_serial.forward_denoms(npmts, rst_voxels.nslices, rst_voxels.voxels, pmt_sns_probs)

    rst_serial.compute_mlem(iterations, rst_voxels, nsipms, sipm_probs, sipm_sns_probs, npmts, pmt_probs, pmt_sns_probs)

    data = ResetTest(rst_voxels_orig, anode_response, cath_response, sipm_probs, sipm_sns_probs, pmt_probs, pmt_sns_probs, rst_voxels.voxels)

    return data


########
# cuda #
########


@fixture(scope="session")
def cuda_version(reset_data, slices_start, data_sipm):
    voxels_data = reset_data.voxels_data
    slices = reset_data.slices
    energies = reset_data.energies

    sipm_ratios = rst_util.compute_sipm_ratio(sipm_dist, pitch, x_size, y_size)
    pmt_ratios = rst_util.compute_pmt_ratio(pmt_dist, npmts, x_size, y_size)

    rst = rstf.RESET(run_number, nsipms, npmts, dist, sipm_dist, pmt_dist,
                     x_size, y_size, rmax, sipm_param_file,
                     pmt_param_file)

    voxels_data_d = rst_mem.copy_voxels_data_h2d(voxels_data)
    slices_data_d = rst_mem.copy_slice_data_h2d(slices)
    slices_start_nc_d = cuda.to_device(slices_start)

    #Create voxels
    rst_voxels_d, slice_ids_h = rstf.create_voxels(rst.cudaf, voxels_data_d, rst.xsize,
                                               rst.ysize, rst.rmax, rst.max_voxels,
                                               slices_start_nc_d, int(slices_start[-1]))
    rst_voxels_h = rst_mem.copy_voxels_d2h(rst_voxels_d)

    #Create anode response
    anode_d  = rstf.create_anode_response(rst.cudaf, slices_data_d)
    nsensors = nsipms * voxels_data_d.nslices
    anode_h  = cuda.from_device(anode_d, (nsensors,), np.dtype('f4'))

    #Create cathode response
    cath_d   = rstf.create_cath_response(energies)
    nsensors = npmts * voxels_data_d.nslices
    cath_h   = cuda.from_device(cath_d, (nsensors,), np.dtype('f4'))

    #Compute probabilities for sipms
    sipm_probs_d = rstf.compute_probabilites(rst.cudaf, rst_voxels_d,
                          rst.nsipms, sipm_ratios.sns_per_voxel, rst.sipm_dist,
                          rst.xs_sipms_d, rst.ys_sipms_d, rst.sipm_param,
                          slices_start_nc_d, rst.xsize, rst.ysize, anode_d)

    sipm_probs_h = rst_mem.copy_probs_d2h(sipm_probs_d, rst_voxels_h.nvoxels,
                                          rst_voxels_h.nslices, nsipms)

    #Compute probabilities for pmts
    pmt_probs_d = rstf.compute_probabilites(rst.cudaf, rst_voxels_d,
                         rst.npmts, pmt_ratios.sns_per_voxel, rst.pmt_dist,
                         rst.xs_pmts_d, rst.ys_pmts_d, rst.pmt_param,
                         slices_start_nc_d, rst.xsize, rst.ysize, cath_d)

    pmt_probs_h = rst_mem.copy_probs_d2h(pmt_probs_d, rst_voxels_h.nvoxels,
                                         rst_voxels_h.nslices, npmts)

    #Compute sensor probabilities for sipms
    sipm_sns_probs_d = rstf.compute_sensor_probs(rst.cudaf, rst_voxels_d,
                              rst.nsipms, sipm_ratios.voxel_per_sns, rst.sipm_dist,
                              rst.xs_sipms_d, rst.ys_sipms_d, rst.sipm_param,
                              slices_start_nc_d, voxels_data_d, rst.xsize,
                              rst.ysize, sipm_probs_d.sensor_start)

    sipm_sns_probs_h = rst_mem.copy_sensor_probs_d2h(sipm_sns_probs_d,
                                                     sipm_probs_d.nprobs)

    #Compute sensor probabilities for pmts
    pmt_sns_probs_d = rstf.compute_sensor_probs(rst.cudaf, rst_voxels_d,
                             rst.npmts, pmt_ratios.voxel_per_sns, rst.pmt_dist,
                             rst.xs_pmts_d, rst.ys_pmts_d, rst.pmt_param,
                             slices_start_nc_d, voxels_data_d, rst.xsize,
                             rst.ysize, pmt_probs_d.sensor_start)

    pmt_sns_probs_h = rst_mem.copy_sensor_probs_d2h(pmt_sns_probs_d,
                                                    pmt_probs_d.nprobs)

    #TODO refactor fwd_denom
    sipm_denoms   = pycuda.gpuarray.zeros(int(nsipms * voxels_data_d.nslices), np.dtype('f4'))
    sipm_denoms_d = sipm_denoms.gpudata

    block_sipm = 1024
    grid_sipm  = math.ceil(sipm_sns_probs_d.nsensors / block_sipm)

    cuda_forward_denom = rst.cudaf.get_function('forward_denom')

    cuda_forward_denom(sipm_denoms_d, sipm_sns_probs_d.sensor_start,
                   sipm_sns_probs_d.sensor_start_ids, sipm_sns_probs_d.probs,
                   sipm_sns_probs_d.voxel_ids, rst_voxels_d.voxels,
                   sipm_sns_probs_d.nsensors,
                   block=(block_sipm, 1, 1), grid=(grid_sipm, 1))

    voxels_h = rstf.compute_mlem(rst.cudaf, rst_voxels_d, voxels_data_d.nslices,
                             rst.npmts,  pmt_probs_d,  pmt_sns_probs_d,
                             rst.nsipms, sipm_probs_d, sipm_sns_probs_d)

    #fwd
    sipm_denoms_h = cuda.from_device(sipm_denoms_d,
                                     (rst.nsipms * rst_voxels_h.nslices,),
                                     np.dtype('f4'))

    rst._destroy_context()

    data = ResetTest(rst_voxels_h, anode_h, cath_h, sipm_probs_h, sipm_sns_probs_h,
                     pmt_probs_h, pmt_sns_probs_h, voxels_h)

    return data


def test_create_voxels(serial_version, cuda_version):
    voxels_s = serial_version.voxels_ini
    voxels_c = cuda_version.voxels_ini

    voxels_s.nvoxels == voxels_c.nvoxels
    np.testing.assert_array_equal(voxels_s.voxels,      voxels_c.voxels)
    np.testing.assert_array_equal(voxels_s.slice_ids,   voxels_c.slice_ids)
    np.testing.assert_array_equal(voxels_s.address,     voxels_c.address)
    np.testing.assert_array_equal(voxels_s.slice_start, voxels_c.slice_start)

def test_anode_response(serial_version, cuda_version):
    anode_s = serial_version.anode
    anode_c = cuda_version.  anode
    np.testing.assert_array_almost_equal(anode_s, anode_s)

def test_cath_response(serial_version, cuda_version):
    cath_s = serial_version.cathode
    cath_c = cuda_version.  cathode
    np.testing.assert_array_almost_equal(cath_s, cath_s)

def test_sipm_probs(serial_version, cuda_version):
    probs_s = serial_version.sipm_probs
    probs_c = cuda_version.  sipm_probs
    np.testing.assert_array_equal(probs_s.voxel_start,  probs_c.voxel_start)
    np.testing.assert_array_equal(probs_s.sensor_start, probs_c.sensor_start)
    np.testing.assert_array_equal(probs_s.sensor_ids,   probs_c.sensor_ids)
    np.testing.assert_array_equal(probs_s.probs,        probs_c.probs)
    np.allclose(probs_s.fwd_nums, probs_c.fwd_nums)

def test_pmt_probs(serial_version, cuda_version):
    probs_s = serial_version.pmt_probs
    probs_c = cuda_version.  pmt_probs
    np.testing.assert_array_equal(probs_s.voxel_start,  probs_c.voxel_start)
    np.testing.assert_array_equal(probs_s.sensor_start, probs_c.sensor_start)
    np.testing.assert_array_equal(probs_s.sensor_ids,   probs_c.sensor_ids)
    np.testing.assert_array_equal(probs_s.probs,        probs_c.probs)
    np.allclose(probs_s.fwd_nums, probs_c.fwd_nums)

def test_sipm_sns_probs(serial_version, cuda_version):
    sns_s = serial_version.sipm_sns_probs
    sns_c = cuda_version.  sipm_sns_probs

    np.testing.assert_array_equal(sns_s.sensor_start,     sns_c.sensor_start)
    np.testing.assert_array_equal(sns_s.sensor_start_ids, sns_c.sensor_start_ids)
    np.testing.assert_array_equal(sns_s.voxel_ids,        sns_c.voxel_ids)
    np.testing.assert_array_equal(sns_s.probs,            sns_c.probs)

def test_pmt_sns_probs(serial_version, cuda_version):
    sns_s = serial_version.pmt_sns_probs
    sns_c = cuda_version.  pmt_sns_probs

    np.testing.assert_array_equal(sns_s.sensor_start,     sns_c.sensor_start)
    np.testing.assert_array_equal(sns_s.sensor_start_ids, sns_c.sensor_start_ids)
    np.testing.assert_array_equal(sns_s.voxel_ids,        sns_c.voxel_ids)
    np.testing.assert_array_equal(sns_s.probs,            sns_c.probs)

def test_reset(serial_version, cuda_version):
    voxels_s = serial_version.voxels
    voxels_c = cuda_version.  voxels
    np.allclose(voxels_s['x'], voxels_c['x'])
    np.allclose(voxels_s['y'], voxels_c['y'])
    np.allclose(voxels_s['E'], voxels_c['E'])
