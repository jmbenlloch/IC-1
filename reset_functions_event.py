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
    def run(self, xmin, xmax, ymin, ymax, charges, slices_start, iterations):
        self.nslices = int(xmin.shape[0])
        print("nslices: ", self.nslices)
        xmin_d    = cuda.to_device(xmin)
        xmax_d    = cuda.to_device(xmax)
        ymin_d    = cuda.to_device(ymin)
        ymax_d    = cuda.to_device(ymax)
        charges_d      = cuda.to_device(charges)
        slices_start_d = cuda.to_device(slices_start)

        create_voxels(self.cudaf, xmin_d, xmax_d, ymin_d,
                      ymax_d, charges_d, self.xsize, self.ysize,
                      self.rmax, self.max_voxels, self.nslices,
                      slices_start_d, int(slices_start[-1]), slices_start)


def create_voxels(cudaf, xmin_d, xmax_d, ymin_d, ymax_d, charges_d,
                  xsize, ysize, rmax, max_voxels, nslices, slices_start_d,
                  nvoxels, slices_start):
    # Conservative approach valid for all events
    voxels_nc_d = cuda.mem_alloc(nslices * max_voxels * voxels_dt.itemsize)
    active_d = cuda.mem_alloc(nslices * max_voxels)
    address  = pycuda.gpuarray.empty(nslices * max_voxels, np.dtype('i4'))
    address_d = address.gpudata

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

#    pdb.set_trace()

    return 0
