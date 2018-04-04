import numpy as np
import tables as tb
import invisible_cities.database.load_db as dbf
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda.compiler import DynamicSourceModule
from pycuda.tools import make_default_context
from pycuda.scan import InclusiveScanKernel
from pycuda.scan import ExclusiveScanKernel
from pycuda import gpuarray
import pycuda
import math

import time
import pdb

from invisible_cities.evm.ic_containers import SensorsParams
from invisible_cities.evm.ic_containers import ResetProbs
from invisible_cities.evm.ic_containers import ResetProbs2
from invisible_cities.evm.ic_containers import ResetSnsProbs
from invisible_cities.evm.ic_containers import ProbsCompact
from invisible_cities.evm.ic_containers import ProbsCompact2
from invisible_cities.evm.ic_containers import Scan
from invisible_cities.evm.ic_containers import ResetVoxels
from invisible_cities.evm.ic_containers import ResetSlices
from invisible_cities.evm.ic_containers import Voxels

# Define types
# due to packing the c struct has 4 bytes for the boolean (maybe pragma pack...)
voxels_dt      = np.dtype([('x', 'f4'), ('y', 'f4'), ('E', 'f4')])
sensors_dt     = np.dtype([('charge', 'f4')])
corrections_dt = np.dtype([('x', 'f4'), ('y', 'f4'), ('factor', 'f4')])
active_dt      = np.dtype([('id', 'i1')])

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
        kernel_code = open('reset_event.cu').read()
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
        self.pmt_param  = read_corrections_file(pmt_param,  'PMT')
        print(self.pmt_param)
        self.sipm_param = read_corrections_file(sipm_param, 'SiPM')

        self.pmts_corr_d  = cuda.to_device(self.pmt_param .params)
        self.sipms_corr_d = cuda.to_device(self.sipm_param.params)

    # sizes of gpuarrays need to be updated with .shape in order to
    # do scan only where there is real data
    def _mem_allocations(self):
        pass

    def run(self, voxels, slices, energies, slices_start, iterations):
        self.nslices = int(voxels.xmin.shape[0])
        print("nslices: ", self.nslices)
        voxels_data_d = copy_voxels_data_h2d(voxels)
        slices_data_d = copy_slice_data_h2d(slices)

        slices_start_nc_d = cuda.to_device(slices_start)

        rst_voxels, slice_ids_h = create_voxels(self.cudaf,
                      voxels_data_d, self.xsize,
                      self.ysize, self.rmax, self.max_voxels, self.nslices,
                      slices_start_nc_d, int(slices_start[-1]), slices_start)

        anode_d = create_anode_response(self.cudaf, slices_data_d,
                                        self.nsipms, self.nslices,
                                        np.int32(slices.charges.shape[0]))

        cath_d = create_cath_response(self.npmts, self.nslices, energies)

        sipms_per_voxel = int(math.floor(2 * self.sipm_dist / self.pitch) + 1)**2
        voxels_per_sipm = int((2 * self.sipm_dist)**2 / ( self.xsize * self.ysize))
#        sipm_probs_d, sipm_nums_d, active_sipms = compute_active_sensors(self.cudaf,
#                               rst_voxels, self.nslices, self.nsipms, sipms_per_voxel,
#                               voxels_per_sipm, self.sipm_dist, self.xs_sipms_d,
#                               self.ys_sipms_d, self.sipm_param, self.sipms_corr_d,
#                               slices_start_nc_d, voxels_data_d,
#                               self.xsize, self.ysize, anode_d)

        #probs_d, sensors_ids_d, voxel_starts_d, sensor_starts, fwd_num_d = compute_probabilites(self.cudaf,
        sipm_probs, nprobs_sipm = compute_probabilites(self.cudaf,
                               rst_voxels, self.nslices, self.nsipms, sipms_per_voxel,
                               self.sipm_dist, self.xs_sipms_d,
                               self.ys_sipms_d, self.sipm_param, self.sipms_corr_d,
                               slices_start_nc_d,
                               self.xsize, self.ysize, anode_d)


        sipm_sns_probs = compute_sensor_probs(self.cudaf,
                             rst_voxels, self.nslices, self.nsipms, sipms_per_voxel,
                             voxels_per_sipm, self.sipm_dist, self.xs_sipms_d,
                             self.ys_sipms_d, self.sipm_param, self.sipms_corr_d,
                             slices_start_nc_d, voxels_data_d,
                             self.xsize, self.ysize, anode_d,
                             sipm_probs.sensor_start)

        pmts_per_voxel = self.npmts
        voxels_per_pmt = int((2 * self.pmt_dist)**2 / ( self.xsize * self.ysize))
#        pmt_probs_d, pmt_nums_d, active_pmts = compute_active_sensors(self.cudaf,
#                               rst_voxels, self.nslices, self.npmts, pmts_per_voxel,
#                               voxels_per_pmt, self.pmt_dist, self.xs_pmts_d,
#                               self.ys_pmts_d, self.pmt_param, self.pmts_corr_d,
#                               slices_start_nc_d, voxels_data_d,
#                               self.xsize, self.ysize, cath_d)

        voxels_h = []
#        voxels_h = compute_mlem(self.cudaf, voxels_d, rst_voxels.nvoxels, self.nslices,
#                     self.npmts,  active_pmts, pmt_probs_d, pmt_nums_d,
#                     self.nsipms, active_sipms, sipm_probs_d, sipm_nums_d)

        return voxels_h, slice_ids_h

def copy_voxels_data_h2d(voxels):
    xmin_d        = cuda.to_device(voxels.xmin)
    xmax_d        = cuda.to_device(voxels.xmax)
    ymin_d        = cuda.to_device(voxels.ymin)
    ymax_d        = cuda.to_device(voxels.ymax)
    charges_avg_d = cuda.to_device(voxels.charge)

    voxels_data_d = Voxels(xmin_d, xmax_d, ymin_d, ymax_d, charges_avg_d)
    return voxels_data_d

def copy_slice_data_h2d(slices):
    sensors_ids_d  = cuda.to_device(slices.sensors)
    charges_d      = cuda.to_device(slices.charges)
    slices_start_d = cuda.to_device(slices.start)

    slices_data_d = ResetSlices(slices_start_d, sensors_ids_d, charges_d)
    return slices_data_d

def copy_reset_voxels_d2h(rst_voxels_d, nslices):
    nvoxels = rst_voxels_d.nvoxels

    voxels_h  = cuda.from_device(rst_voxels_d.voxels,      (nvoxels,),   voxels_dt)
    ids_h     = cuda.from_device(rst_voxels_d.slice_ids,   (nvoxels,),   np.dtype('i4'))
    start_h   = cuda.from_device(rst_voxels_d.slice_start, (nslices+1,), np.dtype('i4'))
    address_h = rst_voxels_d.address.get()

    rst_voxels = ResetVoxels(nvoxels, voxels_h, ids_h, start_h, address_h)
    return rst_voxels

def copy_reset_probs_d2h(probs_d, probs_size, nvoxels, nslices, nsensors):
    probs_h      = cuda.from_device(probs_d.probs,       (int(probs_size),), np.dtype('f4'))
    sensor_ids_h = cuda.from_device(probs_d.sensor_ids, (int(probs_size),), np.dtype('i4'))
    fwd_num_h    = cuda.from_device(probs_d.fwd_nums,     (int(probs_size),), np.dtype('f4'))

    voxel_start_h  = cuda.from_device(probs_d.voxel_start,  (int(nvoxels+1),), np.dtype('i4'))

    sensor_start_data_h   = cuda.from_device(probs_d.sensor_start.data,   (int(nsensors * nslices + 1),), np.dtype('i4'))
    sensor_start_active_h = cuda.from_device(probs_d.sensor_start.active, (int(nsensors * nslices + 1),), np.dtype('i1'))
    sensor_start_addr_h   = probs_d.sensor_start.addr.get()

    sensor_start_h = Scan(sensor_start_data_h, sensor_start_active_h, sensor_start_addr_h)

    rst_probs = ResetProbs2(probs_h, sensor_ids_h, voxel_start_h, sensor_start_h, fwd_num_h)
    return rst_probs


def create_voxels(cudaf, voxels_data_d,
                  xsize, ysize, rmax, max_voxels, nslices, slices_start_d,
                  nvoxels, slices_start):
    # Conservative approach valid for all events
    max_total_voxels = nslices * max_voxels
    voxels_nc_d    = cuda.mem_alloc(max_total_voxels * voxels_dt.itemsize)
    active_d       = cuda.mem_alloc(max_total_voxels)
    slice_ids_nc_d = cuda.mem_alloc(max_total_voxels * 4)

    address   = pycuda.gpuarray.empty(nslices * max_voxels, np.dtype('i4'))
    address_d = address.gpudata

    #TODO Fine tune this memalloc size
    voxels_d         = cuda.mem_alloc(max_total_voxels * voxels_dt.itemsize)
    slice_ids_d      = cuda.mem_alloc(max_total_voxels * 4)
    slices_start_c_d = cuda.mem_alloc((nslices+1) * 4)

    create_voxels = cudaf.get_function('create_voxels')
    create_voxels(voxels_nc_d, slices_start_d,
                  voxels_data_d.xmin,
                  voxels_data_d.xmax,
                  voxels_data_d.ymin,
                  voxels_data_d.ymax,
                  voxels_data_d.charge,
                  xsize, ysize, rmax,
                  active_d, address_d, slice_ids_nc_d,
                  block=(1024, 1, 1), grid=(nslices, 1))

    scan = ExclusiveScanKernel(np.int32, "a+b", 0)
    address.shape = (nvoxels + 1,)
    scan(address)

    compact_voxels = cudaf.get_function('compact_voxels')
    compact_voxels(voxels_nc_d, voxels_d, slice_ids_nc_d, slice_ids_d,
                   address_d, active_d, slices_start_d, slices_start_c_d,
                  block=(1024, 1, 1), grid=(nslices, 1))

    slices_start_c_h = cuda.from_device(slices_start_c_d, (nslices+1,), np.dtype('i4'))
    nvoxels_compact = int(slices_start_c_h[-1])
    slice_ids_h = cuda.from_device(slice_ids_d, (nvoxels_compact,), np.dtype('i4'))

    rst_voxels = ResetVoxels(nvoxels_compact, voxels_d, slice_ids_d, slices_start_c_d, address)
    return rst_voxels, slice_ids_h

def create_anode_response(cudaf, slices_data_d, nsensors, nslices, ncharges):
    total_sensors = int(nslices * nsensors)
    anode_response_d = cuda.mem_alloc(total_sensors * 4)
    cuda.memset_d32(anode_response_d, 0, total_sensors)

    create_anode = cudaf.get_function('create_anode_response')
    create_anode(anode_response_d, nsensors,
                 slices_data_d.sensors,
                 slices_data_d.charges,
                 slices_data_d.start,
                 block=(1024, 1, 1), grid=(nslices, 1))

    return anode_response_d

# TODO: Generalize for npmts > 1
def create_cath_response(npmts, nslices, energies):
    cath_response_d = cuda.to_device(energies)
    return cath_response_d

def compute_probabilites(cudaf, rst_voxels, nslices, nsensors, sensors_per_voxel,
                         sensor_dist, xs_d, ys_d, sensor_param, params_d,
                         slices_start_nc_d,
                         xsize, ysize, sensors_response_d):

    # Reserve memory for probabilities and to compact it
    probs_size = int(rst_voxels.nvoxels * sensors_per_voxel)

    probs_nc_d       = cuda.mem_alloc(probs_size * 4)
    probs_active_d   = cuda.mem_alloc(probs_size)
    probs_addr       = pycuda.gpuarray.empty(probs_size+1, np.dtype('i4'))
    sensors_ids_nc_d = cuda.mem_alloc(probs_size * 4)

    probs_d       = cuda.mem_alloc(probs_size * 4)
    sensors_ids_d = cuda.mem_alloc(probs_size * 4)
    fwd_num_d     = cuda.mem_alloc(probs_size * 4)

    voxel_starts  = pycuda.gpuarray.zeros(int(rst_voxels.nvoxels + 1), np.dtype('i4'))

    # Sensor starts to be compacted
    total_sensors = int(nsensors * nslices)
    sensor_starts_nc       = pycuda.gpuarray.zeros(total_sensors + 1, np.dtype('i4'))
    sensor_starts_addr     = pycuda.gpuarray.zeros(total_sensors + 1, np.dtype('i4'))
    sensor_starts_active_d = cuda.mem_alloc(total_sensors+ 1)

    # Launch kernel
    voxels_per_block = 512
    blocks = math.ceil(rst_voxels.nvoxels / voxels_per_block)

    compute_active = cudaf.get_function('compute_active_sensors')
    compute_active(probs_nc_d, probs_active_d, probs_addr.gpudata,
                   sensors_ids_nc_d, rst_voxels.slice_ids,
                   sensor_starts_nc.gpudata, sensor_starts_active_d, sensor_starts_addr.gpudata,
                   voxel_starts.gpudata, np.int32(rst_voxels.nvoxels), nsensors, np.int32(sensors_per_voxel),
                   rst_voxels.voxels,
                   sensor_dist, xs_d, ys_d,
                   sensor_param.step, sensor_param.nbins, sensor_param.xmin, sensor_param.ymin, params_d,
                   block=(voxels_per_block, 1, 1), grid=(blocks, 1))

    probs_h = cuda.from_device(probs_nc_d, (probs_size,), np.dtype('f4'))
    probs_active_h = cuda.from_device(probs_active_d, (probs_size,), np.dtype('i1'))
    sensor_ids_nc_h = cuda.from_device(sensors_ids_nc_d, (probs_size,), np.dtype('i4'))
#    slices_start_h = cuda.from_device(slices_start_d, (nslices+1,), np.dtype('i4'))


    # Scan everything for compact
    scan = ExclusiveScanKernel(np.int32, "a+b", 0)
    scan(probs_addr)
#    sensor_starts_nc_h = sensor_starts_nc.get()
#    sensor_starts_addr_h = sensor_starts_addr.get()
#    voxel_starts_h  = voxel_starts.get()
    scan(sensor_starts_nc)
    scan(sensor_starts_addr)
    scan(voxel_starts)

    #slices_start
    slices_start_probs_d = cuda.mem_alloc((nslices+1) * 4)
    compact_slices = cudaf.get_function('compact_slices')
    compact_slices(slices_start_probs_d, rst_voxels.slice_start,
                   probs_addr.gpudata, np.int32(sensors_per_voxel),
                   block=(nslices+1, 1, 1), grid=(1, 1))

#    slices_compact_h = cuda.from_device(slices_start_probs_d, (nslices+1,), np.dtype('i4'))
#    slices_start_nc_h = cuda.from_device(slices_start_nc_d, (nslices+1,), np.dtype('i4'))

    compact_probs = cudaf.get_function('compact_probs')
    compact_probs(probs_nc_d, probs_d, fwd_num_d, sensors_ids_nc_d,
                  sensors_ids_d, rst_voxels.slice_ids, probs_addr.gpudata,
                  probs_active_d, np.int32(probs_size),
                  nsensors, np.int32(sensors_per_voxel), sensors_response_d,
                  block=(1024, 1, 1), grid=(100, 1))


    probs_c_h = cuda.from_device(probs_d, (probs_size,), np.dtype('f4'))
    sensor_ids_c_h  = cuda.from_device(sensors_ids_d, (probs_size,), np.dtype('i4'))
    slice_ids_c_h   = cuda.from_device(rst_voxels.slice_ids, (rst_voxels.nvoxels,), np.dtype('i4'))
    fwd_num_h   = cuda.from_device(fwd_num_d, (probs_size,), np.dtype('f4'))
    sensors_response_h = cuda.from_device(sensors_response_d, (nsensors*nslices,), np.dtype('f4'))

    sensor_starts = Scan(sensor_starts_nc.gpudata, sensor_starts_active_d, sensor_starts_addr)

    nprobs = voxel_starts.get()[-1]

    probs = ResetProbs2(probs_d, sensors_ids_d, voxel_starts.gpudata, sensor_starts, fwd_num_d)
    #return probs_d, sensors_ids_d, voxel_starts.gpudata, sensor_starts, fwd_num_d
    return probs, nprobs

def compute_sensor_probs(cudaf, rst_voxels, nslices, nsensors, sensors_per_voxel, voxels_per_sensor,
                         sensor_dist, xs_d, ys_d, sensor_param, params_d,
                         slices_start_nc_d, voxels_data_d,
                         xsize, ysize, sensors_response_d,
                         sensor_starts):
    sensor_probs_size = int(nslices * nsensors * voxels_per_sensor)

    sensor_probs_d         = cuda.mem_alloc(sensor_probs_size * 4)
    active_sensor_probs_d  = cuda.mem_alloc(sensor_probs_size)
    voxel_ids_d            = cuda.mem_alloc(sensor_probs_size * 4)
    cuda.memset_d8(active_sensor_probs_d, 0, sensor_probs_size)
    address_sensor_probs   = pycuda.gpuarray.zeros(sensor_probs_size, np.dtype('i4'))
    address_sensor_probs_d = address_sensor_probs.gpudata

    # sensor probs
    #assumes even nsensors
    block_size = int(nsensors / 4) if nsensors > 1000 else int(nsensors)
    grid_size  = int(nslices  * 4) if nsensors > 1000 else int(nslices)
#    grid_size  = 4
    print("block: ", block_size)
    print("grid: ", grid_size)

    #TODO remove
    counts = pycuda.gpuarray.zeros(int(nsensors), np.dtype('i4'))

    sensor_voxel_probs = cudaf.get_function('sensor_voxel_probs')
    sensor_voxel_probs(sensor_probs_d, sensor_starts.data, voxel_ids_d, np.int32(nsensors),
                       np.int32(nslices), xs_d, ys_d, rst_voxels.voxels, slices_start_nc_d,
                       rst_voxels.address.gpudata, sensor_dist,
                       voxels_data_d.xmin,
                       voxels_data_d.xmax,
                       voxels_data_d.ymin,
                       voxels_data_d.ymax,
                       xsize, ysize, sensor_param.xmin, sensor_param.ymin,
                       sensor_param.step, sensor_param.nbins, params_d, counts.gpudata,
                       block=(block_size, 1, 1), grid=(grid_size, 1))

#TODO remove
    sensor_probs_h = cuda.from_device(sensor_probs_d, (sensor_probs_size,), np.dtype('f4'))
    voxel_ids_h = cuda.from_device(voxel_ids_d, (sensor_probs_size,), np.dtype('i4'))

    sensor_starts_d     = cuda.mem_alloc(int(nsensors * nslices + 1) * 4)
    sensor_starts_ids_d = cuda.mem_alloc(int(nsensors * nslices + 1) * 4)

    num_active_sensors = sensor_starts.addr.get()[-1]
    compact_starts = cudaf.get_function('compact_sensor_start')
    compact_starts(sensor_starts.data, sensor_starts_d, sensor_starts_ids_d,
                   sensor_starts.addr.gpudata, sensor_starts.active,
                   np.int32(nslices * nsensors + 1),
                  block=(block_size, 1, 1), grid=(grid_size, 1))

#    sensor_starts_h     = cuda.from_device(sensor_starts_d, (num_active_sensors+1,), np.dtype('i4'))
#    sensor_starts_ids_h = cuda.from_device(sensor_starts_ids_d, (num_active_sensors+1,), np.dtype('i4'))

    sensor_probs = ResetSnsProbs(sensor_probs_d, voxel_ids_d, sensor_starts_d, sensor_starts_ids_d)

    return sensor_probs


def compute_active_sensors(cudaf, rst_voxels, nslices, nsensors, sensors_per_voxel, voxels_per_sensor,
                           sensor_dist, xs_d, ys_d, sensor_param, params_d,
                           slices_start_nc_d, voxels_data_d,
                           xsize, ysize, sensors_response_d):
    print("voxels_per_sensor: ", voxels_per_sensor)
    probs_size = int(rst_voxels.nvoxels * sensors_per_voxel)
    sensor_probs_size = int(nslices * nsensors * voxels_per_sensor)
    print("sensor_probs_size: ", sensor_probs_size)
    voxel_probs_d         = cuda.mem_alloc(probs_size * 4)
    active_voxel_probs_d  = cuda.mem_alloc(probs_size)
    sensors_ids_d         = cuda.mem_alloc(probs_size * 4)

    voxel_probs_compact_d = cuda.mem_alloc(probs_size * 4)
    forward_num_d         = cuda.mem_alloc(probs_size * 4)
    sensors_ids_compact_d = cuda.mem_alloc(probs_size * 4)
    voxel_starts   = pycuda.gpuarray.zeros(int(rst_voxels.nvoxels + 1), np.dtype('i4'))
    voxel_starts_d = voxel_starts.gpudata

    # One last element for compact later
    address_voxel_probs   = pycuda.gpuarray.empty(probs_size+1, np.dtype('i4'))
    address_voxel_probs_d = address_voxel_probs.gpudata
    #TEST
##    cuda.memset_d32(voxel_probs_d, 0, probs_size)
##    voxel_probs_h = cuda.from_device(voxel_probs_d, (probs_size,), np.dtype('f4'))
#    pdb.set_trace()

    sensor_probs_d         = cuda.mem_alloc(sensor_probs_size * 4)
    active_sensor_probs_d  = cuda.mem_alloc(sensor_probs_size)
    voxel_ids_d            = cuda.mem_alloc(sensor_probs_size * 4)
    cuda.memset_d8(active_sensor_probs_d, 0, sensor_probs_size)
    address_sensor_probs   = pycuda.gpuarray.zeros(sensor_probs_size, np.dtype('i4'))
    address_sensor_probs_d = address_sensor_probs.gpudata

    #One extra position
    sensor_starts_nc  = pycuda.gpuarray.zeros(int(nsensors * nslices + 1), np.dtype('i4'))
    sensor_starts_nc_d  = sensor_starts_nc.gpudata
    sensor_starts_active_d = cuda.mem_alloc(int(nsensors * nslices + 1))
    sensor_starts_addr = pycuda.gpuarray.zeros(int(nsensors * nslices + 1), np.dtype('i4'))
    sensor_starts_addr_d = sensor_starts_addr.gpudata

    sensor_starts_d     = cuda.mem_alloc(int(nsensors * nslices + 1) * 4)
    sensor_starts_ids_d = cuda.mem_alloc(int(nsensors * nslices + 1) * 4)
#    sensor_starts_d = cuda.mem_alloc(int(nsensors * nslices + 1) * 4)
#    cuda.memset_d32(sensor_starts_d, 0, int(nsensors))

    print("nvoxels: ", rst_voxels.nvoxels)

    voxels_per_block = 1024
    voxels_per_block = 512
    blocks = math.ceil(rst_voxels.nvoxels / voxels_per_block)
#    blocks = 1
    print("blocks: ", blocks)

    compute_active = cudaf.get_function('compute_active_sensors')
    compute_active(voxel_probs_d, active_voxel_probs_d, address_voxel_probs_d, sensors_ids_d,
                   rst_voxels.slice_ids, sensor_starts_nc_d, sensor_starts_active_d, sensor_starts_addr_d,
                   voxel_starts_d, np.int32(rst_voxels.nvoxels), nsensors, np.int32(sensors_per_voxel),
                   rst_voxels.voxels, sensor_dist, xs_d, ys_d, sensor_param.step, sensor_param.nbins,
                   sensor_param.xmin, sensor_param.ymin, params_d,
                   block=(voxels_per_block, 1, 1), grid=(blocks, 1))
                   #block=(voxels_per_block, 1, 1), grid=(1, 1))
                   #block=(2, 1, 1), grid=(1, 1))

##    voxel_probs_h = cuda.from_device(voxel_probs_d, (probs_size,), np.dtype('f4'))
##    active_voxel_probs_h = cuda.from_device(active_voxel_probs_d, (probs_size,), np.dtype('i1'))
##    sensor_ids_h = cuda.from_device(sensors_ids_d, (probs_size,), np.dtype('i4'))
##    slices_start_h = cuda.from_device(slices_start_d, (nslices+1,), np.dtype('i4'))

    scan = InclusiveScanKernel(np.int32, "a+b")
    scan(address_voxel_probs)
    sensor_starts_nc_h = sensor_starts_nc.get()
    sensor_starts_addr_h = sensor_starts_addr.get()
    voxel_starts_h  = voxel_starts.get()
    scan(sensor_starts_nc)
    scan(sensor_starts_addr)
    scan(voxel_starts)

    #slices_start
    slices_start_probs_d = cuda.mem_alloc((nslices+1) * 4)
    compact_slices = cudaf.get_function('compact_slices')
    compact_slices(slices_start_probs_d, rst_voxels.slice_start,
                   address_voxel_probs_d, np.int32(sensors_per_voxel),
                   block=(nslices+1, 1, 1), grid=(1, 1))

##    slices_compact_h = cuda.from_device(slices_start_probs_d, (nslices+1,), np.dtype('i4'))
##    slices_start_nc_h = cuda.from_device(slices_start_nc_d, (nslices+1,), np.dtype('i4'))

    compact_probs = cudaf.get_function('compact_probs')
    compact_probs(voxel_probs_d, voxel_probs_compact_d, forward_num_d, sensors_ids_d,
                  sensors_ids_compact_d, rst_voxels.slice_ids, address_voxel_probs_d,
                  active_voxel_probs_d, np.int32(probs_size),
                  nsensors, np.int32(sensors_per_voxel), sensors_response_d,
                  block=(1024, 1, 1), grid=(100, 1))

#TODO remove
    voxel_probs_c_h = cuda.from_device(voxel_probs_compact_d, (probs_size,), np.dtype('f4'))
    sensor_ids_c_h  = cuda.from_device(sensors_ids_compact_d, (probs_size,), np.dtype('i4'))
    slice_ids_c_h   = cuda.from_device(rst_voxels.slice_ids, (rst_voxels.nvoxels,), np.dtype('i4'))
    forward_num_h   = cuda.from_device(forward_num_d, (probs_size,), np.dtype('f4'))
    sensors_response_h = cuda.from_device(sensors_response_d, (nsensors*nslices,), np.dtype('f4'))


    # sensor probs
    #assumes even nsensors
    block_size = int(nsensors / 4) if nsensors > 1000 else int(nsensors)
    grid_size  = int(nslices  * 4) if nsensors > 1000 else int(nslices)
#    grid_size  = 4
    print("block: ", block_size)
    print("grid: ", grid_size)

    #TODO remove
    counts = pycuda.gpuarray.zeros(int(nsensors), np.dtype('i4'))

    sensor_voxel_probs = cudaf.get_function('sensor_voxel_probs')
    sensor_voxel_probs(sensor_probs_d, sensor_starts_nc_d, voxel_ids_d, nsensors,
                       np.int32(nslices), xs_d, ys_d, rst_voxels.voxels, slices_start_nc_d,
                       rst_voxels.address.gpudata, sensor_dist,
                       voxels_data_d.xmin,
                       voxels_data_d.xmax,
                       voxels_data_d.ymin,
                       voxels_data_d.ymax,
                       xsize, ysize, sensor_param.xmin, sensor_param.ymin,
                       sensor_param.step, sensor_param.nbins, params_d, counts.gpudata,
                       block=(block_size, 1, 1), grid=(grid_size, 1))

#TODO remove
    sensor_probs_h = cuda.from_device(sensor_probs_d, (sensor_probs_size,), np.dtype('f4'))
    voxel_ids_h = cuda.from_device(voxel_ids_d, (sensor_probs_size,), np.dtype('i4'))


##    xmins_h  = cuda.from_device(xmins_d, (nslices,), np.dtype('f4'))
##    xmaxs_h  = cuda.from_device(xmaxs_d, (nslices,), np.dtype('f4'))
##    ymins_h  = cuda.from_device(ymins_d, (nslices,), np.dtype('f4'))
##    ymaxs_h  = cuda.from_device(ymaxs_d, (nslices,), np.dtype('f4'))


    num_active_sensors = sensor_starts_addr.get()[-1]
    compact_starts = cudaf.get_function('compact_sensor_start')
    compact_starts(sensor_starts_nc_d, sensor_starts_d, sensor_starts_ids_d,
                   sensor_starts_addr_d, sensor_starts_active_d,
                   np.int32(nslices * nsensors + 1),
                  block=(block_size, 1, 1), grid=(grid_size, 1))

    sensor_starts_h     = cuda.from_device(sensor_starts_d, (num_active_sensors+1,), np.dtype('i4'))
    sensor_starts_ids_h = cuda.from_device(sensor_starts_ids_d, (num_active_sensors+1,), np.dtype('i4'))

##    sensor_starts_nc_h = sensor_starts_nc.get()
##    for i,idx in enumerate(sensor_starts_ids_h):
##        assert sensor_starts_h[i] == sensor_starts_nc_h[idx]

    #check transpose
#    total_size = sensor_starts.get()[-1]
#    sensors_start = sensor_starts.get()
#    voxels_start  = voxel_starts.get()
#    slice_id = 0
#    voxel_id = 0
#    for i in range(total_size):
#        if i >= slices_compact_h[slice_id + 1]:
#            slice_id = slice_id + 1
##            print(slice_id, i)
#        if i >= voxels_start[voxel_id + 1]:
#            voxel_id = voxel_id + 1
#            print(voxel_id, i)
#        voxel_sensor_p  = voxel_probs_c_h[i]
#        voxel_sensor_id = sensor_ids_c_h[i]
#        assert i == i
#
#        sensor_start = sensors_start[slice_id * nsensors + voxel_sensor_id]
#        sensor_end   = sensors_start[slice_id * nsensors + voxel_sensor_id + 1]
#        for j in range(sensor_start, sensor_end):
#            if voxel_ids_h[j] == voxel_id:
#                try:
#                    assert sensor_probs_h[j] == voxel_sensor_p
#                except AssertionError:
#                    print("{} sid {}, vid {}, p1 {}, p2 {}".format(i, voxel_sensor_id, voxel_id, voxel_sensor_p, sensor_probs_h[j]))


#    addrs = address_voxel_probs.get()
#    for i in range(len(voxel_probs_h)):
#        print(i)
#        if active_voxel_probs_h[i]:
#            addr = addrs[i]-1
#            print (voxel_probs_h[i], voxel_probs_c_h[addr])
#            assert voxel_probs_h[i] == voxel_probs_c_h[addr]
#            assert  sensor_ids_h[i] ==  sensor_ids_c_h[addr]

#    sensors = []
#    counts = []
#    for i in range(len(slices_start_h)-1):
#        print(slices_start_h[i], slices_start_h[i+1])
#        start = slices_start_h[i]
#        end = slices_start_h[i+1]
#        s, c = np.unique(sensor_ids_h[start:end], return_counts=True)
#        sensors.append(s[1:])
#        counts.append(c[1:])
#    print(list(map(max, counts)))

#    pdb.set_trace()

    all_probs = ProbsCompact2(voxel_probs_compact_d, sensors_ids_compact_d,
                              voxel_starts_d, sensor_probs_d, voxel_ids_d,
                              sensor_starts_d, sensor_starts_ids_d)
    return all_probs, forward_num_d, num_active_sensors


def compute_mlem(cudaf, voxels_d, nvoxels, nslices,
                 npmts, active_pmts, pmt_probs, pmt_nums_d,
                 nsipms, active_sipms, sipm_probs, sipm_nums_d):
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
    blocks = math.ceil(nvoxels / voxels_per_block)

    sipm_denoms   = pycuda.gpuarray.zeros(int(nsipms * nslices + 1), np.dtype('f4'))
    sipm_denoms_d = sipm_denoms.gpudata
    pmt_denoms    = pycuda.gpuarray.zeros(int(npmts * nslices + 1), np.dtype('f4'))
    pmt_denoms_d  = pmt_denoms.gpudata

    forward_denom = cudaf.get_function('forward_denom')
    mlem_step = cudaf.get_function('mlem_step')

    iterations = 100
    for i in range(iterations):
        forward_denom(sipm_denoms_d, sipm_probs.sensor_start,
                      sipm_probs.sensor_start_ids, sipm_probs.sensor_probs,
                      sipm_probs.voxel_ids, voxels_d, active_sipms,
                      block=(block_sipm, 1, 1), grid=(grid_sipm, 1))

        forward_denom(pmt_denoms_d, pmt_probs.sensor_start,
                      pmt_probs.sensor_start_ids, pmt_probs.sensor_probs,
                      pmt_probs.voxel_ids, voxels_d, active_pmts,
                      block=(block_pmt, 1, 1), grid=(grid_pmt, 1))

        mlem_step(voxels_d, pmt_probs.voxel_start, pmt_probs.probs,
                  pmt_probs.sensor_ids, pmt_nums_d, pmt_denoms_d,
                  sipm_probs.voxel_start, sipm_probs.probs,
                  sipm_probs.sensor_ids, sipm_nums_d, sipm_denoms_d,
                  np.int32(nvoxels),
                  block=(voxels_per_block, 1, 1), grid=(blocks, 1))

    voxels_h = cuda.from_device(voxels_d, (nvoxels,), voxels_dt)
    energies = np.array(list(map(lambda v: v[2], voxels_h)))
#    pdb.set_trace()

    return voxels_h
