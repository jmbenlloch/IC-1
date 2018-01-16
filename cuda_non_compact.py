import numpy as np
import tables as tb
import invisible_cities.database.load_db as dbf
import invisible_cities.reco.corrections as corrf

import pycuda.driver as cuda
from pycuda.compiler import SourceModule
#import pycuda.autoinit

run_number = 4495
DataSiPM = dbf.DataSiPM(run_number)
DataPMT  = dbf.DataPMT(run_number)
nsipms = 1792
npmts = 1

#Define types
voxels_dt      = np.dtype([('x', 'f4'), ('y', 'f4'), ('E', 'f4'), ('active', 'i4')])
sensors_dt     = np.dtype([('id', 'i4'), ('charge', 'f4'), ('active', 'i4')])
corrections_dt = np.dtype([('x', 'f4'), ('y', 'f4'), ('factor', 'f4')])
active_dt      = np.dtype([('id', 'i1')])

#create context
from pycuda.tools import make_default_context
cuda.init()
ctx = make_default_context()

#compile cuda code
kernel_code = open('reset_non_compact.cu').read()
mod = SourceModule(kernel_code)
create_voxels          = mod.get_function("create_voxels")
initiliaze_anode       = mod.get_function("initialize_anode")
create_anode_response  = mod.get_function("create_anode_response")
compute_active_sensors = mod.get_function("compute_active_sensors")
mlem_step              = mod.get_function("mlem_step")

#Get (x,y) positions
xs_sipms_h = DataSiPM.X.values.astype('f4')
ys_sipms_h = DataSiPM.Y.values.astype('f4')
xs_sipms_d = cuda.to_device(xs_sipms_h)
ys_sipms_d = cuda.to_device(ys_sipms_h)

#Need to choose somehow whether to use one or more PMTs
#xs_pmts = DataPMT.X.values.astype('f4')
#ys_pmts = DataPMT.Y.values.astype('f4')
xs_pmts_h = np.array([0.], dtype='f4')
ys_pmts_h = np.array([0.], dtype='f4')
xs_pmts_d = cuda.to_device(xs_pmts_h)
ys_pmts_d = cuda.to_device(ys_pmts_h)



t0 = 578125.0
z = 421.6528125
sensor_ids = np.array([1023, 1601, 1609, 1687, 1403, 1376, 1379, 1384, 1385, 1386, 1387,
                       1392, 1393, 1394, 1395, 1396, 1397, 1400, 1401, 1402, 1431, 1454,
                       1455, 1462, 1471, 1125], dtype='i4')
charges    = np.array([ 2.66531368,  3.73287782,  5.2828832 ,  2.31351022,  4.41027829,
                        2.20720324,  3.98584685, 14.72511416, 18.30488489,  6.26837369,
                        2.96073255, 14.59842807, 23.50427029,  6.8357565 ,  3.92635856,
                        7.65335644,  2.09604995,  3.67059242,  9.99260022, 10.94429901,
                        2.83670564,  2.89119827,  2.57772977,  2.82898736,  2.83071036,
                        2.26349407], dtype='f4')
s2_energy = np.float32(491.47727966) # measured by pmts

#Lifetime correction
ZCorr = corrf.LifetimeCorrection(1093.77, 23.99)


#################
#################
# Create voxels #
#################
#################
dist = 20.
sipm_dist = np.float32(20.)
#pmt_dist = np.float32(205)
pmt_dist = np.float32(10000) # all pmts are included
sipm_thr = 5.
sizeX = 2.
sizeY = 2.
rMax = 198

#Compute min,max x,y
#size x size y, rmax
selC = (charges > sipm_thr)
xmin = np.float32(DataSiPM.X[sensor_ids[selC]].values.min()-dist)
xmax = np.float32(DataSiPM.X[sensor_ids[selC]].values.max()+dist)
ymin = np.float32(DataSiPM.Y[sensor_ids[selC]].values.min()-dist)
ymax = np.float32(DataSiPM.Y[sensor_ids[selC]].values.max()+dist)
charge = np.float32(charges.mean())
xsize = np.float32(sizeX)
ysize = np.float32(sizeY)
rmax = np.float32(rMax)

#TODO: Check rounding here
threads_x = int((xmax - xmin) / xsize)
threads_y = int((ymax - ymin) / ysize)
print(threads_x, threads_y)

#allocate memory for result
num_voxels = threads_x * threads_y
voxels_d = cuda.mem_alloc(num_voxels * voxels_dt.itemsize)

create_voxels(voxels_d, xmin, xmax, ymin, ymax, xsize, ysize, rmax, charge, block=(1, 1, 1), grid=(threads_x, threads_y))

voxels_h = cuda.from_device(voxels_d, (num_voxels,), voxels_dt)
#for h in voxels_h:
#    print (h)


##########################
##########################
## Create anode response #
##########################
##########################
#### Step 1: Initialize sipms with zero charge
#allocate memory for result
# due to packing the c struct is 12 bytes instead of 9.
# Mem layout to be updated, maybe pragma pack
sensors_sipms_d = cuda.mem_alloc(nsipms * sensors_dt.itemsize)

initiliaze_anode(sensors_sipms_d, xmin, xmax, xs_sipms_d, ymin, ymax, ys_sipms_d, sipm_dist, block=(1, 1, 1), grid=(nsipms, 1))

#### Step 2: Put the correct charge for active sensors
sensor_ids_d = cuda.to_device(sensor_ids)
charges_d    = cuda.to_device(charges)

nsensors = sensor_ids.shape[0]
create_anode_response(sensors_sipms_d, sensor_ids_d, charges_d, block=(1, 1, 1), grid=(nsensors, 1))

############################
############################
## Create cathode response #
############################
############################

s2e = s2_energy * ZCorr(z).value
sensors_pmts = np.array([(1, s2e, 1)], dtype=sensors_dt)
print(sensors_pmts)
sensors_pmts_d = cuda.mem_alloc(npmts * sensors_dt.itemsize)

##################################################
##################################################
## Select active sensors & compute probabilities #
##################################################
##################################################

def read_corrections_file(filename, node):
    corr_h5 = tb.open_file(filename)
    corr_table = getattr(corr_h5.root.ResetMap, node)
    corrections_dt = np.dtype([('x', 'f4'), ('y', 'f4'), ('factor', 'f4')])

    # we need to explicitly build it to get into memory only (x,y,factor)
    # to check: struct.unpack('f', bytes(pmts_corr.data)[i*4:(i+1)*4])
    corr = np.array(list(zip(corr_table.col('x'), corr_table.col('y'), corr_table.col('factor'))), dtype=corrections_dt)

    step  =  corr[1][1] - corr[0][1]
    xmin  =  corr[0][0]
    ymin  =  corr[0][1]

    nbins = (corr[-1][0] - corr[0][0]) / step + 1
    nbins = nbins.astype('i4')
    corr_h5.close()

    return xmin, ymin, nbins, step, corr

#Sipms
# Read parametrizations
sipm_corr_file = "/home/jmbenlloch/reset_data/mapas/SiPM_Map_corr_z0.0_keV.h5"
xmin_sipms, ymin_sipms, nbins_sipms, step_sipms, sipms_corr = read_corrections_file(sipm_corr_file, 'SiPM')

active_sipms_d = cuda.mem_alloc(num_voxels * nsipms) # TODO: Update after compact
probs_sipms_d  = cuda.mem_alloc(num_voxels * nsipms * 4) # TODO: Update after compact
sipms_corr_d = cuda.to_device(sipms_corr)

compute_active_sensors(active_sipms_d, probs_sipms_d, sensors_sipms_d, voxels_d, xs_sipms_d, ys_sipms_d, np.int32(nsipms), sipm_dist, step_sipms, nbins_sipms, xmin_sipms, ymin_sipms, sipms_corr_d, block=(1, 1, 1), grid=(num_voxels, 1))

#Pmts
# If using more than 1 pmt, take into account that maximum distance is 205
pmt_corr_file  = "/home/jmbenlloch/reset_data/mapas/PMT_Map_corr_keV.h5"
xmin_pmts, ymin_pmts, nbins_pmts, step_pmts, pmts_corr = read_corrections_file(pmt_corr_file, 'PMT')

active_pmts_d = cuda.mem_alloc(num_voxels * npmts) # TODO: Update after compact
probs_pmts_d  = cuda.mem_alloc(num_voxels * npmts * 4) # TODO: Update after compact
pmts_corr_d  = cuda.to_device(pmts_corr)

compute_active_sensors(active_pmts_d, probs_pmts_d, sensors_pmts_d, voxels_d, xs_pmts_d, ys_pmts_d, np.int32(npmts), pmt_dist, step_pmts, nbins_pmts, xmin_pmts, ymin_pmts, pmts_corr_d, block=(1, 1, 1), grid=(num_voxels, 1))

probs_pmts_h = cuda.from_device(probs_pmts_d, (num_voxels,), np.dtype('f4'))
active_pmts_h = cuda.from_device(active_pmts_d, (num_voxels,), active_dt)
print (probs_pmts_h)
print (active_pmts_h)
#for p in probs_pmts_h:
#    print(p)

##################
##################
## Run MLEM step #
##################
##################

#allocate memory for output
voxels_out_d = cuda.mem_alloc(voxels_h.nbytes)

#run
#mlem_step(voxels_d, voxels_out_d, sensors_sipms_d, sensors_pmts_d, probs_pmts_d, probs_sipms_d, active_sipms_d, active_pmts_d, np.int32(num_voxels), np.int32(nsipms), np.int32(npmts), block=(1, 1, 1), grid=(num_voxels, 1))


iterations = 100
for i in range(iterations):
    if i > 0:
        voxels_d, voxels_out_d = voxels_out_d, voxels_d
    mlem_step(voxels_d, voxels_out_d, sensors_sipms_d, sensors_pmts_d, probs_pmts_d, probs_sipms_d, active_sipms_d, active_pmts_d, np.int32(num_voxels), np.int32(nsipms), np.int32(npmts), block=(1, 1, 1), grid=(num_voxels, 1))

voxels_out_h = cuda.from_device(voxels_out_d, voxels_h.shape, voxels_h.dtype)

##############################
##############################

ctx.detach()
