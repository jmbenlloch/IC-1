import invisible_cities.reset.utils as rst_utils
import invisible_cities.reset.reset_serial as rst_serial
import numpy as np
import math

from invisible_cities.core.system_of_units import pes, mm, mus, ns
import invisible_cities.reco.corrections    as corrf
import invisible_cities.database.load_db as dbf
import reset_functions_event as rstf

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
#x_size     = 2.
#y_size     = 2.

x_size     = 10.
y_size     = 10.

rmax       = 198 #
slice_width = 2.
pmt_param  = "/home/jmbenlloch/reset_data/mapas/PMT_Map_corr_keV.h5"
sipm_param = "/home/jmbenlloch/reset_data/mapas/SiPM_Map_corr_z0.0_keV.h5"

pmap_file = '4495_21215.h5'

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

# Read file and select peaks
selector = rst_utils.refresh_selector(pmap_conf)
s1s, s2s, s2sis, peaks = rst_utils.load_and_select_peaks(pmap_file, 21215, selector)

# Lifetime correction
ZCorr = corrf.LifetimeCorrection(1093.77, 23.99)

# Sensors info
data_sipm = dbf.DataSiPM(run_number)

# Take the first peak
peak = next(iter(peaks))
evt = 21215
sipm_thr = sipm_thr

# Prepare data
voxels_data, slices, energies, zs = rst_utils.prepare_data(s1s, s2s, s2sis, slice_width, evt, peak, data_sipm, nsipms, sipm_thr, dist, ZCorr)
slices_start = rst_utils.slices_start(voxels_data, x_size, y_size)


#################
# Create voxels #
#################

#Serial
rst_voxels = rst_serial.create_voxels(voxels_data, slices_start, x_size, y_size, rmax)

#CUDA
rst = rstf.RESET(run_number, nsipms, npmts, dist, sipm_dist, pmt_dist, sipm_thr, x_size, y_size, rmax, sipm_param, pmt_param)

import pycuda.driver as cuda

nslices = int(voxels_data.xmin.shape[0])
voxels_data_d = rstf.copy_voxels_data_h2d(voxels_data)
slices_data_d = rstf.copy_slice_data_h2d(slices)

slices_start_nc_d = cuda.to_device(slices_start)


rst_voxels_d, slice_ids_h = rstf.create_voxels(rst.cudaf, voxels_data_d, rst.xsize,
                                               rst.ysize, rst.rmax, rst.max_voxels, nslices,
                                               slices_start_nc_d, int(slices_start[-1]), slices_start)

rst_voxels_h = rstf.copy_voxels_d2h(rst_voxels_d, nslices)

#Check
rst_voxels.nvoxels == rst_voxels_h.nvoxels
np.testing.assert_array_equal(rst_voxels.voxels, rst_voxels_h.voxels)
np.testing.assert_array_equal(rst_voxels.slice_ids, rst_voxels_h.slice_ids)
np.testing.assert_array_equal(rst_voxels.address, rst_voxels_h.address)
np.testing.assert_array_equal(rst_voxels.slice_start, rst_voxels_h.slice_start)


##################
# Anode response #
##################

#Serial
nslices = voxels_data.xmin.shape[0]
anode_response = rst_serial.create_anode_response(nslices, nsipms, slices)

#CUDA version
anode_d = rstf.create_anode_response(rst.cudaf, slices_data_d,
                                     rst.nsipms, nslices,
                                     np.int32(slices.charges.shape[0]))

total_sensors = nsipms * nslices
anode_h = cuda.from_device(anode_d, (total_sensors,), np.dtype('f4'))

#Check
np.testing.assert_array_almost_equal(anode_response[1792:(2*1792)], anode_h[1792:(2*1792)])

####################
# Cathode response #
####################

energies

#################
# Compute probs #
#################

pmt_param_file   = "/home/jmbenlloch/reset_data/mapas/PMT_Map_corr_keV.h5"
sipm_param_file  = "/home/jmbenlloch/reset_data/mapas/SiPM_Map_corr_z0.0_keV.h5"
sipm_params = rst_serial.read_corrections_file(sipm_param_file, 'SiPM')

sipms_per_voxel = int(math.floor(2 * sipm_dist / pitch) + 1)**2
voxels_per_sipm = int((2 * sipm_dist)**2 / (x_size * y_size))
xs = data_sipm.X
ys = data_sipm.Y

#Serial
probs, sensor_ids, voxel_starts, sensor_starts, nprobs, fwd_num = rst_serial.compute_probabilities_cuda(rst_voxels.voxels, rst_voxels.slice_ids, rst_voxels.nvoxels, nslices, xs, ys, nsipms, sipms_per_voxel, sipm_dist, sipm_params, anode_response)


#CUDA
sipm_probs_d, nprobs_h = rstf.compute_probabilites(rst.cudaf, rst_voxels_d, nslices, rst.nsipms, sipms_per_voxel,
                                                   rst.sipm_dist, rst.xs_sipms_d,
                                                   rst.ys_sipms_d, rst.sipm_param, rst.sipms_corr_d,
                                                   slices_start_nc_d,
                                                   rst.xsize, rst.ysize, anode_d)

sipm_probs_h = rstf.copy_probs_d2h(sipm_probs_d, nprobs, rst_voxels.nvoxels, nslices, nsipms)

#Check
np.testing.assert_array_equal(sipm_probs_h.voxel_start, voxel_starts)
np.testing.assert_array_equal(sipm_probs_h.sensor_start.data, sensor_starts)
np.testing.assert_array_equal(sipm_probs_h.sensor_ids, sensor_ids)
np.testing.assert_array_equal(probs, sipm_probs_h.probs)
np.testing.assert_array_almost_equal(fwd_num, sipm_probs_h.fwd_nums)

################
# Sensor probs #
################

sensor_probs, voxel_ids, sensor_starts, sensor_starts_c, sensor_starts_c_ids = rst_serial.compute_sensor_probs(probs, nprobs, nslices, nsipms, voxel_starts, sensor_starts, sensor_ids, rst_voxels.slice_ids)

#CUDA
sipm_sns_probs_d,  active_sipms = rstf.compute_sensor_probs(rst.cudaf,
                                                            rst_voxels_d, nslices, nsipms, sipms_per_voxel,
                                                            voxels_per_sipm, rst.sipm_dist, rst.xs_sipms_d,
                                                            rst.ys_sipms_d, rst.sipm_param, rst.sipms_corr_d,
                                                            slices_start_nc_d, voxels_data_d,
                                                            rst.xsize, rst.ysize, anode_d,
                                                            sipm_probs_d.sensor_start)
sipm_sns_probs_h = rstf.copy_sensor_probs_d2h(sipm_sns_probs_d, nprobs_h, active_sipms)

#Check
np.testing.assert_array_equal(sipm_sns_probs_h.sensor_start, sensor_starts_c)
np.testing.assert_array_equal(sipm_sns_probs_h.sensor_start_ids, sensor_starts_c_ids)
np.testing.assert_array_equal(sipm_sns_probs_h.voxel_ids, voxel_ids)
np.testing.assert_array_equal(sipm_sns_probs_h.probs, sensor_probs)

#################
# Forward denom #
#################

#Serial
sipm_fwd_denom = rst_serial.forward_denoms(nsipms, nslices, rst_voxels.voxels, sensor_probs, voxel_ids, sensor_starts_c, sensor_starts_c_ids)

#CUDA
import pycuda
sipm_denoms   = pycuda.gpuarray.zeros(int(nsipms * nslices), np.dtype('f4'))
sipm_denoms_d = sipm_denoms.gpudata

block_sipm = 1024
grid_sipm  = math.ceil(active_sipms / block_sipm)

cuda_forward_denom = rst.cudaf.get_function('forward_denom')

cuda_forward_denom(sipm_denoms_d, sipm_sns_probs_d.sensor_start,
                       sipm_sns_probs_d.sensor_start_ids, sipm_sns_probs_d.probs,
                       sipm_sns_probs_d.voxel_ids, rst_voxels_d.voxels, active_sipms,
                       block=(block_sipm, 1, 1), grid=(grid_sipm, 1))

sipm_denoms_h = cuda.from_device(sipm_denoms_d, (rst.nsipms * nslices,), np.dtype('f4'))

#Check
np.allclose(sipm_denoms_h, sipm_fwd_denom)

rst._destroy_context()
