import numpy as np
import tables as tb
import invisible_cities.database.load_db as dbf
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda.tools import make_default_context

import time

from invisible_cities.evm.ic_containers import SensorsParams

# Define types
# due to packing the c struct has 4 bytes for the boolean (maybe pragma pack...)
voxels_dt      = np.dtype([('x', 'f4'), ('y', 'f4'), ('E', 'f4'), ('active', 'i4')])
sensors_dt     = np.dtype([('id', 'i4'), ('charge', 'f4'), ('active', 'i4')])
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

        self._create_context()
        self._compile()
        self._load_xy_positions()
        self._load_parametrization(sipm_param, pmt_param)

    def _create_context(self):
        #create context
        cuda.init()
        self.ctx = make_default_context()

    def _destroy_context(self):
        self.ctx.detach()

    def _compile(self):
        kernel_code = open('reset_non_compact.cu').read()
        self.cudaf = SourceModule(kernel_code)

    def _load_xy_positions(self):
        #Get (x,y) positions
        self.xs_sipms_h = self.data_sipm.X.values.astype('f4')
        self.ys_sipms_h = self.data_pmt .Y.values.astype('f4')
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

    def run(self, sensor_ids, charges, energy, iterations):
        tstart = time.time()
        sensor_ids_d = cuda.to_device(sensor_ids)
        charges_d    = cuda.to_device(charges)
        nsensors     = np.int32(sensor_ids.shape[0])
        tend = time.time()
        print("Copy sensor ids & charge time: {}".format(tend-tstart))

        tstart = time.time()
        xmin, xmax, ymin, ymax, num_voxels, voxels_d, voxels_h = create_voxels(self.cudaf, sensor_ids, charges, self.sipm_thr, self.dist, self.xsize, self.ysize, self.rmax, self.data_sipm)
        tend = time.time()
        print("Create voxels time: {}".format(tend-tstart))

        tstart = time.time()
        sensors_sipms_d = create_anode_response(self.cudaf, sensor_ids_d,
                            charges_d, nsensors, self.nsipms,
                            self.sipm_dist, xmin, xmax, self.xs_sipms_d,
                            ymin, ymax, self.ys_sipms_d)
        tend = time.time()
        print("Create anode response: {}".format(tend-tstart))

        tstart = time.time()
        sensors_pmts_d = create_cath_response(self.npmts, energy)
        tend = time.time()
        print("Create cath response: {}".format(tend-tstart))

        tstart = time.time()
        active_sipms_d, probs_sipms_d = compute_active_sensors(self.cudaf,
                        num_voxels, voxels_d, self.nsipms, self.xs_sipms_d,
                        self.ys_sipms_d, sensors_sipms_d, self.sipm_dist,
                        self.sipm_param, self.sipms_corr_d)
        tend = time.time()
        print("Compute active SiPMs: {}".format(tend-tstart))

        tstart = time.time()
        active_pmts_d, probs_pmts_d = compute_active_sensors(self.cudaf,
                        num_voxels, voxels_d, self.npmts, self.xs_pmts_d,
                        self.ys_pmts_d, sensors_pmts_d, self.pmt_dist,
                        self.pmt_param, self.pmts_corr_d)
        tend = time.time()
        print("Compute active PMTs: {}".format(tend-tstart))

        tstart = time.time()
        run_mlem_step(self.cudaf, iterations, voxels_d, sensors_sipms_d, sensors_pmts_d, probs_sipms_d, probs_pmts_d, active_sipms_d, active_pmts_d, num_voxels, self.nsipms, self.npmts)
        tend = time.time()
        print("Run MLEM step: {}".format(tend-tstart))

def create_voxels(cudaf, sensor_ids, charges, sipm_thr, dist,
                  xsize, ysize, rmax, data_sipm):
    selC = (charges > sipm_thr)
    xmin = np.float32(data_sipm.X[sensor_ids[selC]].values.min()-dist)
    xmax = np.float32(data_sipm.X[sensor_ids[selC]].values.max()+dist)
    ymin = np.float32(data_sipm.Y[sensor_ids[selC]].values.min()-dist)
    ymax = np.float32(data_sipm.Y[sensor_ids[selC]].values.max()+dist)
    charge = np.float32(charges.mean())
    xsize = np.float32(xsize)
    ysize = np.float32(ysize)
    rmax = np.float32(rmax)

    #TODO: Check rounding here
    threads_x = int((xmax - xmin) / xsize)
    threads_y = int((ymax - ymin) / ysize)
    print(threads_x, threads_y)

    num_voxels = threads_x * threads_y

    #allocate memory for result
    voxels_d = cuda.mem_alloc(num_voxels * voxels_dt.itemsize)
    func = cudaf.get_function('create_voxels')
    func(voxels_d, xmin, xmax, ymin, ymax, xsize, ysize, rmax, charge,
         block=(1, 1, 1), grid=(threads_x, threads_y))
    voxels_h = cuda.from_device(voxels_d, (num_voxels,), voxels_dt)

    return xmin, xmax, ymin, ymax, np.int32(num_voxels), voxels_d, voxels_h

def create_anode_response(cudaf, sensor_ids_d, charges_d, nsensors,
                          total_sipms, sipm_dist, xmin, xmax,
                          xs_sipms_d, ymin, ymax, ys_sipms_d):
    # for mallocs has to be a python int not a np.int32...
    sensors_sipms_d = cuda.mem_alloc(int(total_sipms) * sensors_dt.itemsize)
    # Step 1: Initialize sipms with zero charge
    func = cudaf.get_function('initialize_anode')
    func(sensors_sipms_d, xmin, xmax, xs_sipms_d, ymin, ymax,
         ys_sipms_d, sipm_dist,
         block=(1, 1, 1), grid=(int(total_sipms), 1))
    # Step 2: Put the correct charge for active sensors
    func = cudaf.get_function('create_anode_response')
    func(sensors_sipms_d, sensor_ids_d, charges_d,
         block=(1, 1, 1), grid=(int(nsensors), 1))
    return sensors_sipms_d

# Energy already corrected, decide where to do that...
# TODO: Currently only works with one pmt
def create_cath_response(npmts, energy):
    sensors_pmts_d  = cuda.mem_alloc(int(npmts) * sensors_dt.itemsize)
    sensors_pmts = np.array([(1, energy, 1)], dtype=sensors_dt)
    sensors_pmts_d = cuda.to_device(sensors_pmts)
    return sensors_pmts_d

## Select active sensors & compute probabilities
def compute_active_sensors(cudaf, num_voxels, voxels_d, nsensors, xs_d, ys_d,
                           sensors_d, sensor_dist, sensor_param, params_d):
    # TODO: Update after compact
    active_d = cuda.mem_alloc(int(num_voxels * nsensors))
    probs_d  = cuda.mem_alloc(int(num_voxels * nsensors * 4))
    func = cudaf.get_function('compute_active_sensors')
    func(active_d, probs_d, sensors_d, voxels_d, xs_d, ys_d, nsensors,
         sensor_dist, sensor_param.step, sensor_param.nbins,
         sensor_param.xmin, sensor_param.ymin, params_d,
         block=(1, 1, 1), grid=(int(num_voxels), 1))
    return active_d, probs_d

## Run MLEM step
def run_mlem_step(cudaf, iterations, voxels_d, sensors_sipms_d, sensors_pmts_d,
                  probs_sipms_d, probs_pmts_d, active_sipms_d, active_pmts_d,
                  num_voxels, nsipms, npmts):

    tstart = time.time()
    voxels_out_d    = cuda.mem_alloc(int(num_voxels * voxels_dt.itemsize))
    func = cudaf.get_function('mlem_step')
    for i in range(iterations):
        if i > 0:
            voxels_d, voxels_out_d = voxels_out_d, voxels_d
        func(voxels_d, voxels_out_d, sensors_sipms_d, sensors_pmts_d,
             probs_pmts_d, probs_sipms_d, active_sipms_d, active_pmts_d,
             num_voxels, nsipms, npmts,
             block=(1, 1, 1), grid=(int(num_voxels), 1))
    tend = time.time()
    print("MLEM: {}".format(tend-tstart))


    tstart = time.time()
    voxels_out_h = cuda.from_device(voxels_out_d, (num_voxels,), voxels_dt)
    tend = time.time()
    print("Copy voxels from device: {}".format(tend-tstart))
    return voxels_out_h
