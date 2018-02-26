import numpy as np
import tables as tb
import invisible_cities.database.load_db as dbf
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda.compiler import DynamicSourceModule
from pycuda.tools import make_default_context
from pycuda.scan import InclusiveScanKernel
from pycuda import gpuarray
import pycuda
import math

import time
import pdb

from invisible_cities.evm.ic_containers import SensorsParams
from invisible_cities.evm.ic_containers import ResetProbs
from invisible_cities.evm.ic_containers import ProbsCompact
from invisible_cities.evm.ic_containers import Scan

# Define types
# due to packing the c struct has 4 bytes for the boolean (maybe pragma pack...)
voxels_dt      = np.dtype([('x', 'f4'), ('y', 'f4'), ('E', 'f4')])
#sensors_dt     = np.dtype([('id', 'i4'), ('charge', 'f4'), ('active', 'i4')])
sensors_dt     = np.dtype([('charge', 'f4')])
corrections_dt = np.dtype([('x', 'f4'), ('y', 'f4'), ('factor', 'f4')])
active_dt      = np.dtype([('id', 'i1')])

segmented_scan_dt = np.dtype([('value', 'f4'), ('flag', 'i4')])
mlem_scan_dt = np.dtype([('eff', 'f4'), ('projection', 'f4'),('flag', 'i4')])
pycuda.tools.get_or_register_dtype("scan_t", segmented_scan_dt)
pycuda.tools.get_or_register_dtype("mlem_scan_t", mlem_scan_dt)

header = '''
struct scan_t {
    float value;
    int active;

     __device__ scan_t& operator=(const scan_t& a){
        value = a.value;
        active = a.active;
        return *this;
    }

    __device__ scan_t operator+(const scan_t& a) const{
        scan_t res;
        res.value = a.value + value * !a.active;
        res.active = a.active || active;
        return res;
    }
};

struct mlem_scan_t {
    float eff;
    float projection;
    int active;

     __device__ mlem_scan_t& operator=(const mlem_scan_t& a){
        eff = a.eff;
        projection = a.projection;
        active = a.active;
        return *this;
    }

    __device__ mlem_scan_t operator+(const mlem_scan_t& a) const{
        mlem_scan_t res;
        res.eff = a.eff + eff * !a.active;
        res.projection = a.projection + projection * !a.active;
        res.active = a.active || active;
        return res;
    }
};'''


def mem_allocator():
    dev = cuda.Context.get_device()
    max_groups = 3*dev.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)
    scan_mem = list(map(cuda.mem_alloc, [max_groups * 100] * 2)) #extra size for possible structs
    flip = 0
    def allocator(nbytes):
        nonlocal flip
#        print("mem: ", flip)
        mem = scan_mem[flip]
        flip = (flip + 1) % 2
        return mem
    return allocator

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

        #Need to choose somehow whether to use one or more PMTs
        #self.xs_pmts = DataPMT.X.values.astype('f4')
        #self.ys_pmts = DataPMT.Y.values.astype('f4')
        self.xs_pmts_h = np.array([0.], dtype='f4')
        self.ys_pmts_h = np.array([0.], dtype='f4')
        self.xs_pmts_d = cuda.to_device(self.xs_pmts_h)
        self.ys_pmts_d = cuda.to_device(self.ys_pmts_h)

    def _load_parametrization(self, sipm_param, pmt_param):
        self.pmt_param  = read_corrections_file(pmt_param,  'PMT')
        self.sipm_param = read_corrections_file(sipm_param, 'SiPM')

        self.pmts_corr_d  = cuda.to_device(self.pmt_param .params)
        self.sipms_corr_d = cuda.to_device(self.sipm_param.params)

    # sizes of gpuarrays need to be updated with .shape in order to
    # do scan only where there is real data
    def _mem_allocations(self):
        pass

    #def run(self, sensor_ids, charges, energy, iterations):
    def run(self, xmin, xmax, ymin, ymax, charges_avg, slices_start,
            iterations, sensors_ids, charges, slices_start_charges,
            energies):
        self.nslices = int(xmin.shape[0])
        print("nslices: ", self.nslices)
        xmin_d    = cuda.to_device(xmin)
        xmax_d    = cuda.to_device(xmax)
        ymin_d    = cuda.to_device(ymin)
        ymax_d    = cuda.to_device(ymax)
        charges_avg_d = cuda.to_device(charges_avg)
        slices_start_d = cuda.to_device(slices_start)

        sensors_ids_d = cuda.to_device(sensors_ids)
        charges_d = cuda.to_device(charges)
        slices_start_charges_d = cuda.to_device(slices_start_charges)

        nvoxels, voxels_d, slices_start_d = create_voxels(self.cudaf, xmin_d,
                      xmax_d, ymin_d, ymax_d, charges_avg_d, self.xsize,
                      self.ysize, self.rmax, self.max_voxels, self.nslices,
                      slices_start_d, int(slices_start[-1]), slices_start)

        anode_d = create_anode_response(self.cudaf, self.nsipms, self.nslices,
                              sensors_ids_d, charges_d,
                              np.int32(charges.shape[0]),
                              slices_start_charges_d)

        cath_d = create_cath_response(self.npmts, self.nslices, energies)

        sipms_per_voxel = int(math.floor(2 * self.sipm_dist / self.pitch) + 1)**2
        compute_active_sensors(self.cudaf, self.nslices, nvoxels, self.nsipms,
                               sipms_per_voxel, voxels_d, self.sipm_dist, self.xs_sipms_d,
                               self.ys_sipms_d, self.sipm_param, self.sipms_corr_d, slices_start_d)


def create_voxels(cudaf, xmin_d, xmax_d, ymin_d, ymax_d, charges_d,
                  xsize, ysize, rmax, max_voxels, nslices, slices_start_d,
                  nvoxels, slices_start):
    # Conservative approach valid for all events
    voxels_nc_d = cuda.mem_alloc(nslices * max_voxels * voxels_dt.itemsize)
    active_d = cuda.mem_alloc(nslices * max_voxels)
    address  = pycuda.gpuarray.empty(nslices * max_voxels, np.dtype('i4'))
    address_d = address.gpudata

    #TODO Fine tune this memalloc size
    voxels_d = cuda.mem_alloc(nslices * max_voxels * voxels_dt.itemsize)
    slices_start_c_d = cuda.mem_alloc((nslices+1) * 4)

    create_voxels = cudaf.get_function('create_voxels')
    create_voxels(voxels_nc_d, slices_start_d, xmin_d, xmax_d,
                  ymin_d, ymax_d, charges_d, xsize, ysize, rmax,
                  active_d, address_d,
                  block=(1024, 1, 1), grid=(nslices, 1))
#                  block=(1024, 1, 1), grid=(2, 1))
    voxels_h = cuda.from_device(voxels_nc_d, (nvoxels,), voxels_dt)
    active_h = cuda.from_device(active_d, (nvoxels,), np.dtype('i1'))

    scan = InclusiveScanKernel(np.int32, "a+b")
    address.shape = (nvoxels,)
    scan(address)

    compact_voxels = cudaf.get_function('compact_voxels')
    compact_voxels(voxels_nc_d, voxels_d, address_d, active_d,
                   slices_start_d, slices_start_c_d,
                  block=(1024, 1, 1), grid=(nslices, 1))
    voxels_c_h = cuda.from_device(voxels_d, (nvoxels,), voxels_dt)
    slices_start_c_h = cuda.from_device(slices_start_c_d, (nslices+1,), np.dtype('i4'))
    nvoxels_compact = int(slices_start_c_h[-1])

    return nvoxels_compact, voxels_d, slices_start_c_d

def create_anode_response(cudaf, nsensors, nslices, sensors_ids_d,
                          charges_d, ncharges, slices_start_charges_d):
    total_sensors = int(nslices * nsensors)
    anode_response_d = cuda.mem_alloc(total_sensors * 4)
    cuda.memset_d32(anode_response_d, 0, total_sensors)
    anode_response_h = cuda.from_device(anode_response_d, (total_sensors,), np.dtype('i4'))

    create_anode = cudaf.get_function('create_anode_response')
    create_anode(anode_response_d, nsensors, sensors_ids_d, charges_d,
                 slices_start_charges_d,
                 block=(1024, 1, 1), grid=(nslices, 1))
    anode_response_h = cuda.from_device(anode_response_d, (total_sensors,), np.dtype('f4'))

    sensor_ids_h = cuda.from_device(sensors_ids_d, (ncharges,), np.dtype('i4'))
    charges_h = cuda.from_device(charges_d, (ncharges,), np.dtype('f4'))
    slices_h = cuda.from_device(slices_start_charges_d, (nslices,), np.dtype('i4'))

    #check resulst is correct
    slc = 0
    for i in range(slices_h[-1]):
        if i >= slices_h[slc+1]:
            slc = slc + 1
        idx = slc * nsensors + sensor_ids_h[i]
        assert anode_response_h[idx] == charges_h[i]

    return anode_response_h


# TODO: Generalize for npmts > 1
def create_cath_response(npmts, nslices, energies):
    cath_response_d = cuda.to_device(energies)


def compute_active_sensors(cudaf, nslices, nvoxels, nsensors, sensors_per_voxel,
                           voxels_d, sensor_dist, xs_d, ys_d, sensor_param, params_d, slices_start_d):
    probs_size = int(nvoxels * sensors_per_voxel)
    voxel_probs_d         = cuda.mem_alloc(probs_size * 4)
    active_voxel_probs_d  = cuda.mem_alloc(probs_size)
    sensors_ids_d         = cuda.mem_alloc(probs_size * 4)

    voxel_probs_compact_d = cuda.mem_alloc(probs_size * 4)
    sensors_ids_compact_d = cuda.mem_alloc(probs_size * 4)

    # One last element for compact later
    address_voxel_probs   = pycuda.gpuarray.empty(probs_size+1, np.dtype('i4'))
    address_voxel_probs_d = address_voxel_probs.gpudata
    #TEST
    cuda.memset_d32(voxel_probs_d, 0, probs_size)
    voxel_probs_h = cuda.from_device(voxel_probs_d, (probs_size,), np.dtype('f4'))
#    pdb.set_trace()

    sensor_probs_d         = cuda.mem_alloc(probs_size * 4)
    active_sensor_probs_d  = cuda.mem_alloc(probs_size)
    address_sensor_probs   = pycuda.gpuarray.empty(probs_size, np.dtype('i4'))
    address_sensor_probs_d = address_voxel_probs.gpudata
    print("nvoxels: ", nvoxels)

    voxels_per_block = 1024
    blocks = math.ceil(nvoxels / voxels_per_block)
    print("blocks: ", blocks)

    compute_active = cudaf.get_function('compute_active_sensors')
    compute_active(voxel_probs_d, active_voxel_probs_d, address_voxel_probs_d, sensors_ids_d,
                   np.int32(nvoxels), nsensors, np.int32(sensors_per_voxel), voxels_d,
                   sensor_dist, xs_d, ys_d, sensor_param.step, sensor_param.nbins,
                   sensor_param.xmin, sensor_param.ymin, params_d,
                   block=(voxels_per_block, 1, 1), grid=(blocks, 1))
                   #block=(voxels_per_block, 1, 1), grid=(1, 1))
                   #block=(2, 1, 1), grid=(1, 1))

    voxel_probs_h = cuda.from_device(voxel_probs_d, (probs_size,), np.dtype('f4'))
    active_voxel_probs_h = cuda.from_device(active_voxel_probs_d, (probs_size,), np.dtype('i1'))
    sensor_ids_h = cuda.from_device(sensors_ids_d, (probs_size,), np.dtype('i4'))
    slices_start_h = cuda.from_device(slices_start_d, (nslices+1,), np.dtype('i4'))

    scan = InclusiveScanKernel(np.int32, "a+b")
    scan(address_voxel_probs)

    #slices_start
    slices_start_probs_d = cuda.mem_alloc((nslices+1) * 4)
    compact_slices = cudaf.get_function('compact_slices')
    compact_slices(slices_start_probs_d, slices_start_d,
                   address_voxel_probs_d, np.int32(sensors_per_voxel),
                   block=(nslices+1, 1, 1), grid=(1, 1))

    slices_compact_h = cuda.from_device(slices_start_probs_d, (nslices+1,), np.dtype('i4'))

    compact_probs = cudaf.get_function('compact_probs')
    compact_probs(voxel_probs_d, voxel_probs_compact_d, sensors_ids_d,
                  sensors_ids_compact_d, address_voxel_probs_d,
                  active_voxel_probs_d, np.int32(probs_size),
                  block=(1024, 1, 1), grid=(100, 1))


    voxel_probs_c_h = cuda.from_device(voxel_probs_compact_d, (probs_size,), np.dtype('f4'))
    sensor_ids_c_h  = cuda.from_device(sensors_ids_compact_d, (probs_size,), np.dtype('i4'))


#    addrs = address_voxel_probs.get()
#    for i in range(len(voxel_probs_h)):
#        print(i)
#        if active_voxel_probs_h[i]:
#            addr = addrs[i]-1
#            print (voxel_probs_h[i], voxel_probs_c_h[addr])
#            assert voxel_probs_h[i] == voxel_probs_c_h[addr]
#            assert  sensor_ids_h[i] ==  sensor_ids_c_h[addr]


#    pdb.set_trace()
