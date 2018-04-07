import numpy as np
import math
import os

import pdb

import invisible_cities.database.load_db as dbf

import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda.tools    import make_default_context
from pycuda.scan     import ExclusiveScanKernel
from pycuda          import gpuarray

import invisible_cities.reset.utils  as rst_utils
import invisible_cities.reset.memory as rst_mem

from invisible_cities.evm.ic_containers import ResetProbs2
from invisible_cities.evm.ic_containers import ResetSnsProbs
from invisible_cities.evm.ic_containers import Scan
from invisible_cities.evm.ic_containers import ResetVoxels

voxels_dt      = np.dtype([('x', 'f4'), ('y', 'f4'), ('E', 'f4')])
corrections_dt = np.dtype([('x', 'f4'), ('y', 'f4'), ('factor', 'f4')])


class RESET:
    def __init__(self, run_number, nsipms, npmts, dist, sipm_dist,
                 pmt_dist, sipm_thr, xsize, ysize, rmax,
                 sipm_param, pmt_param):
        self.run_number = run_number
        self.nsipms    = np.int32(nsipms)
        self.npmts     = np.int32(npmts)
        self.dist      = np.float32(dist)
        self.sipm_dist = np.float32(sipm_dist)
        self.pmt_dist  = np.float32(pmt_dist)
        self.sipm_thr  = np.float32(sipm_thr)
        self.xsize     = np.float32(xsize)
        self.ysize     = np.float32(ysize)
        self.rmax      = np.float32(rmax)
        self.data_sipm = dbf.DataSiPM(run_number)
        self.data_pmt  = dbf.DataPMT(run_number)
        self.pitch     = 10. #hardcoded value!

        det_xsize = self.data_sipm.X.ptp()
        det_ysize = self.data_sipm.Y.ptp()
        self.max_voxels = int(det_xsize * det_ysize / (self.xsize * self.ysize))

        self._create_context()
        self._compile()
        self._load_xy_positions()
        self._load_parametrization(sipm_param, pmt_param)
#        self._mem_allocations()

    def _create_context(self):
        #create context
        cuda.init()
        self.ctx = make_default_context()

    def _destroy_context(self):
        self.ctx.detach()

    def _compile(self):
        source_file = os.path.expandvars("$ICDIR/reset/reset.cu")
        kernel_code = open(source_file).read()
        self.cudaf = SourceModule(kernel_code)

    def _load_xy_positions(self):
        #Get (x,y) positions
        self.xs_sipms_h = self.data_sipm.X.values.astype('f4')
        self.ys_sipms_h = self.data_sipm.Y.values.astype('f4')
        self.xs_sipms_d = cuda.to_device(self.xs_sipms_h)
        self.ys_sipms_d = cuda.to_device(self.ys_sipms_h)

        #TODO: Generalize for any number of PMTs
        #Need to choose somehow whether to use one or more PMTs
        #self.xs_pmts = DataPMT.X.values.astype('f4')
        #self.ys_pmts = DataPMT.Y.values.astype('f4')
        self.xs_pmts_h = np.array([0.], dtype='f4')
        self.ys_pmts_h = np.array([0.], dtype='f4')
        self.xs_pmts_d = cuda.to_device(self.xs_pmts_h)
        self.ys_pmts_d = cuda.to_device(self.ys_pmts_h)

    def _load_parametrization(self, sipm_param, pmt_param):
        pmt_param  = rst_utils.read_corrections_file(pmt_param,  'PMT')
        sipm_param = rst_utils.read_corrections_file(sipm_param, 'SiPM')

        pmts_corr_d  = cuda.to_device(pmt_param .params)
        sipms_corr_d = cuda.to_device(sipm_param.params)

        self.pmt_param  =  pmt_param._replace(params = pmts_corr_d)
        self.sipm_param = sipm_param._replace(params = sipms_corr_d)

    # sizes of gpuarrays need to be updated with .shape in order to
    # do scan only where there is real data
    def _mem_allocations(self):
        pass

    def run(self, voxels, slices, energies, slices_start, iterations):
        self.nslices = int(voxels.xmin.shape[0])
        voxels_data_d = rst_mem.copy_voxels_data_h2d(voxels)
        slices_data_d = rst_mem.copy_slice_data_h2d(slices)

        slices_start_nc_d = cuda.to_device(slices_start)

        rst_voxels, slice_ids_h = create_voxels(self.cudaf,
                      voxels_data_d, self.xsize,
                      self.ysize, self.rmax, self.max_voxels,
                      slices_start_nc_d, int(slices_start[-1]))

        anode_d = create_anode_response(self.cudaf, slices_data_d)
        cath_d  = create_cath_response(energies)

        sipms_per_voxel = int(math.floor(2 * self.sipm_dist / self.pitch) + 1)**2
        voxels_per_sipm = int((2 * self.sipm_dist)**2 / ( self.xsize * self.ysize))

        sipm_probs = compute_probabilites(self.cudaf,
                               rst_voxels, self.nsipms, sipms_per_voxel,
                               self.sipm_dist, self.xs_sipms_d,
                               self.ys_sipms_d, self.sipm_param,
                               slices_start_nc_d,
                               self.xsize, self.ysize, anode_d)

        sipm_sns_probs, nprobs_sns_sipms = compute_sensor_probs(self.cudaf,
                             rst_voxels, self.nslices, self.nsipms, sipms_per_voxel,
                             voxels_per_sipm, self.sipm_dist, self.xs_sipms_d,
                             self.ys_sipms_d, self.sipm_param,
                             slices_start_nc_d, voxels_data_d,
                             self.xsize, self.ysize,
                             sipm_probs.sensor_start)

        pmts_per_voxel = self.npmts
        voxels_per_pmt = int((2 * self.pmt_dist)**2 / ( self.xsize * self.ysize))
        pmt_probs = compute_probabilites(self.cudaf,
                               rst_voxels, self.npmts, pmts_per_voxel,
                               self.pmt_dist, self.xs_pmts_d,
                               self.ys_pmts_d, self.pmt_param,
                               slices_start_nc_d,
                               self.xsize, self.ysize, anode_d)

        pmt_sns_probs, nprobs_sns_pmts = compute_sensor_probs(self.cudaf,
                             rst_voxels, self.nslices, self.npmts, pmts_per_voxel,
                             voxels_per_pmt, self.pmt_dist, self.xs_pmts_d,
                             self.ys_pmts_d, self.pmt_param,
                             slices_start_nc_d, voxels_data_d,
                             self.xsize, self.ysize,
                             pmt_probs.sensor_start)

        voxels_h = []
        voxels_h = compute_mlem(self.cudaf, rst_voxels, self.nslices,
                     self.npmts,  nprobs_sns_pmts,  pmt_probs, pmt_sns_probs,
                     self.nsipms, nprobs_sns_sipms, sipm_probs, sipm_sns_probs)

        return voxels_h, slice_ids_h


def create_voxels(cudaf, voxels_data_d,
                  xsize, ysize, rmax, max_voxels, slices_start_d,
                  nvoxels):
    # Conservative approach valid for all events
    max_total_voxels = int(voxels_data_d.nslices * max_voxels)
    voxels_nc_d    = cuda.mem_alloc(max_total_voxels * voxels_dt.itemsize)
    active_d       = cuda.mem_alloc(max_total_voxels)
    slice_ids_nc_d = cuda.mem_alloc(max_total_voxels * 4)

    address   = gpuarray.empty(voxels_data_d.nslices * max_voxels, np.dtype('i4'))
    address_d = address.gpudata

    #TODO Fine tune this memalloc size
    voxels_d         = cuda.mem_alloc(max_total_voxels * voxels_dt.itemsize)
    slice_ids_d      = cuda.mem_alloc(max_total_voxels * 4)
    slices_start_c_d = cuda.mem_alloc((int(voxels_data_d.nslices+1)) * 4)

    create_voxels = cudaf.get_function('create_voxels')
    create_voxels(voxels_nc_d, slices_start_d,
                  voxels_data_d.xmin,
                  voxels_data_d.xmax,
                  voxels_data_d.ymin,
                  voxels_data_d.ymax,
                  voxels_data_d.charge,
                  xsize, ysize, rmax,
                  active_d, address_d, slice_ids_nc_d,
                  block=(1024, 1, 1), grid=(int(voxels_data_d.nslices), 1))

    scan = ExclusiveScanKernel(np.int32, "a+b", 0)
    address.shape = (nvoxels + 1,)
    scan(address)

    compact_voxels = cudaf.get_function('compact_voxels')
    compact_voxels(voxels_nc_d, voxels_d, slice_ids_nc_d, slice_ids_d,
                   address_d, active_d, slices_start_d, slices_start_c_d,
                  block=(1024, 1, 1), grid=(int(voxels_data_d.nslices), 1))

    slices_start_c_h = cuda.from_device(slices_start_c_d, (voxels_data_d.nslices+1,), np.dtype('i4'))
    nvoxels_compact = int(slices_start_c_h[-1])
    slice_ids_h = cuda.from_device(slice_ids_d, (nvoxels_compact,), np.dtype('i4'))

    rst_voxels = ResetVoxels(voxels_data_d.nslices, nvoxels_compact, voxels_d, slice_ids_d, slices_start_c_d, address)
    return rst_voxels, slice_ids_h

def create_anode_response(cudaf, slices_data_d):
    total_sensors = int(slices_data_d.nslices * slices_data_d.nsensors)
    anode_response_d = cuda.mem_alloc(total_sensors * 4)
    cuda.memset_d32(anode_response_d, 0, total_sensors)

    create_anode = cudaf.get_function('create_anode_response')
    create_anode(anode_response_d,
                 slices_data_d.nsensors,
                 slices_data_d.sensors,
                 slices_data_d.charges,
                 slices_data_d.start,
                 block=(1024, 1, 1), grid=(int(slices_data_d.nslices), 1))

    return anode_response_d

# TODO: Generalize for npmts > 1
def create_cath_response(energies):
    cath_response_d = cuda.to_device(energies)
    return cath_response_d

def compute_probabilites(cudaf, voxels, nsensors, sensors_per_voxel,
                         sensor_dist, xs_d, ys_d, sns_param,
                         slices_start_nc_d,
                         xsize, ysize, sensors_response_d):

    # Reserve memory for probabilities and to compact it
    probs_size = int(voxels.nvoxels * sensors_per_voxel)

    probs_nc_d       = cuda.mem_alloc(probs_size * 4)
    probs_active_d   = cuda.mem_alloc(probs_size)
    probs_addr       = gpuarray.empty(probs_size+1, np.dtype('i4'))
    sensors_ids_nc_d = cuda.mem_alloc(probs_size * 4)

    probs_d       = cuda.mem_alloc(probs_size * 4)
    sensors_ids_d = cuda.mem_alloc(probs_size * 4)
    fwd_num_d     = cuda.mem_alloc(probs_size * 4)

    voxel_starts  = gpuarray.zeros(int(voxels.nvoxels + 1), np.dtype('i4'))

    # Sensor starts to be compacted
    total_sensors = int(nsensors * voxels.nslices)
    sensor_starts_nc       = gpuarray.zeros(total_sensors + 1, np.dtype('i4'))
    sensor_starts_addr     = gpuarray.zeros(total_sensors + 1, np.dtype('i4'))
    sensor_starts_active_d = cuda.mem_alloc(total_sensors+ 1)

    # Launch kernel
    voxels_per_block = 512
    blocks = math.ceil(voxels.nvoxels / voxels_per_block)

    compute_active = cudaf.get_function('compute_active_sensors')
    compute_active(probs_nc_d, probs_active_d, probs_addr.gpudata,
                   sensors_ids_nc_d, voxels.slice_ids,
                   sensor_starts_nc.gpudata, sensor_starts_active_d, sensor_starts_addr.gpudata,
                   voxel_starts.gpudata, np.int32(voxels.nvoxels), nsensors, np.int32(sensors_per_voxel),
                   voxels.voxels,
                   sensor_dist, xs_d, ys_d,
                   sns_param.step, sns_param.nbins, sns_param.xmin, sns_param.ymin, sns_param.params,
                   block=(voxels_per_block, 1, 1), grid=(blocks, 1))

    probs_h = cuda.from_device(probs_nc_d, (probs_size,), np.dtype('f4'))
    probs_active_h = cuda.from_device(probs_active_d, (probs_size,), np.dtype('i1'))

    # Scan everything for compact
    scan = ExclusiveScanKernel(np.int32, "a+b", 0)
    scan(probs_addr)
    scan(sensor_starts_nc)
    scan(sensor_starts_addr)
    scan(voxel_starts)

    #slices_start
    slices_start_probs_d = cuda.mem_alloc(int(voxels.nslices+1) * 4)
    compact_slices = cudaf.get_function('compact_slices')
    compact_slices(slices_start_probs_d, voxels.slice_start,
                   probs_addr.gpudata, np.int32(sensors_per_voxel),
                   block=(int(voxels.nslices+1), 1, 1), grid=(1, 1))

    compact_probs = cudaf.get_function('compact_probs')
    compact_probs(probs_nc_d, probs_d, fwd_num_d, sensors_ids_nc_d,
                  sensors_ids_d, voxels.slice_ids, probs_addr.gpudata,
                  probs_active_d, np.int32(probs_size),
                  nsensors, np.int32(sensors_per_voxel), sensors_response_d,
                  block=(1024, 1, 1), grid=(100, 1))

    sensor_starts = Scan(sensor_starts_nc.gpudata, sensor_starts_active_d, sensor_starts_addr)
    nprobs = voxel_starts.get()[-1]

    probs = ResetProbs2(nprobs, probs_d, sensors_ids_d, voxel_starts.gpudata, sensor_starts, fwd_num_d)
    return probs

def compute_sensor_probs(cudaf, rst_voxels, nslices, nsensors, sensors_per_voxel, voxels_per_sensor,
                         sensor_dist, xs_d, ys_d, sns_param,
                         slices_start_nc_d, voxels_data_d,
                         xsize, ysize, sensor_starts):
    sensor_probs_size = int(nslices * nsensors * voxels_per_sensor)

    sensor_probs_d         = cuda.mem_alloc(sensor_probs_size * 4)
    active_sensor_probs_d  = cuda.mem_alloc(sensor_probs_size)
    voxel_ids_d            = cuda.mem_alloc(sensor_probs_size * 4)
    cuda.memset_d8(active_sensor_probs_d, 0, sensor_probs_size)
    address_sensor_probs   = gpuarray.zeros(sensor_probs_size, np.dtype('i4'))
    address_sensor_probs_d = address_sensor_probs.gpudata

    # sensor probs
    #assumes even nsensors
    block_size = int(nsensors / 4) if nsensors > 1000 else int(nsensors)
    grid_size  = int(nslices  * 4) if nsensors > 1000 else int(nslices)

    sensor_voxel_probs = cudaf.get_function('sensor_voxel_probs')
    sensor_voxel_probs(sensor_probs_d, sensor_starts.data, voxel_ids_d, np.int32(nsensors),
                       np.int32(nslices), xs_d, ys_d, rst_voxels.voxels, slices_start_nc_d,
                       rst_voxels.address.gpudata, sensor_dist,
                       voxels_data_d.xmin,
                       voxels_data_d.xmax,
                       voxels_data_d.ymin,
                       voxels_data_d.ymax,
                       xsize, ysize, sns_param.xmin, sns_param.ymin,
                       sns_param.step, sns_param.nbins, sns_param.params,
                       block=(block_size, 1, 1), grid=(grid_size, 1))

    sensor_starts_d     = cuda.mem_alloc(int(nsensors * nslices + 1) * 4)
    sensor_starts_ids_d = cuda.mem_alloc(int(nsensors * nslices + 1) * 4)

    num_active_sensors = sensor_starts.addr.get()[-1]
    compact_starts = cudaf.get_function('compact_sensor_start')
    compact_starts(sensor_starts.data, sensor_starts_d, sensor_starts_ids_d,
                   sensor_starts.addr.gpudata, sensor_starts.active,
                   np.int32(nslices * nsensors + 1),
                  block=(block_size, 1, 1), grid=(grid_size, 1))

    sensor_probs = ResetSnsProbs(sensor_probs_d, voxel_ids_d,
                                 num_active_sensors, sensor_starts_d, sensor_starts_ids_d)

    return sensor_probs, num_active_sensors


def compute_mlem(cudaf, rst_voxels_d, nslices,
                 npmts, active_pmts, pmt_probs, pmt_sns_probs,
                 nsipms, active_sipms, sipm_probs, sipm_sns_probs):
    block_sipm = 1024
    grid_sipm  = math.ceil(active_sipms / block_sipm)
    print("block sipm: ", block_sipm)
    print("grid sipm: ", grid_sipm)

    block_pmt = int(active_pmts) if active_pmts < 1024 else 1024
    grid_pmt  = math.ceil(active_pmts / block_pmt)
    print("block pmt: ", block_pmt)
    print("grid pmt: ", grid_pmt)

    voxels_per_block = 1024
    voxels_per_block = 512
    blocks = math.ceil(rst_voxels_d.nvoxels / voxels_per_block)

    sipm_denoms   = gpuarray.zeros(int(nsipms * nslices), np.dtype('f4'))
    sipm_denoms_d = sipm_denoms.gpudata
    pmt_denoms    = gpuarray.zeros(int(npmts * nslices), np.dtype('f4'))
    pmt_denoms_d  = pmt_denoms.gpudata

    forward_denom = cudaf.get_function('forward_denom')
    mlem_step = cudaf.get_function('mlem_step')

    iterations = 100
    for i in range(iterations):
        forward_denom(sipm_denoms_d, sipm_sns_probs.sensor_start,
                      sipm_sns_probs.sensor_start_ids, sipm_sns_probs.probs,
                      sipm_sns_probs.voxel_ids, rst_voxels_d.voxels, active_sipms,
                      block=(block_sipm, 1, 1), grid=(grid_sipm, 1))

        forward_denom(pmt_denoms_d, pmt_sns_probs.sensor_start,
                      pmt_sns_probs.sensor_start_ids, pmt_sns_probs.probs,
                      pmt_sns_probs.voxel_ids, rst_voxels_d.voxels, active_pmts,
                      block=(block_pmt, 1, 1), grid=(grid_pmt, 1))

        mlem_step(rst_voxels_d.voxels, pmt_probs.voxel_start, pmt_probs.probs,
                  pmt_probs.sensor_ids, pmt_probs.fwd_nums, pmt_denoms_d,
                  sipm_probs.voxel_start, sipm_probs.probs,
                  sipm_probs.sensor_ids, sipm_probs.fwd_nums, sipm_denoms_d,
                  np.int32(rst_voxels_d.nvoxels),
                  block=(voxels_per_block, 1, 1), grid=(blocks, 1))

    voxels_h = cuda.from_device(rst_voxels_d.voxels, (rst_voxels_d.nvoxels,), voxels_dt)
    energies = np.array(list(map(lambda v: v[2], voxels_h)))

    return voxels_h
