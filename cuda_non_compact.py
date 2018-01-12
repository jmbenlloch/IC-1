import numpy as np
import invisible_cities.database.load_db as dbf

run_number = 4495
nsipms = 1792

sensor_ids = np.array([1023, 1601, 1609, 1687, 1403, 1376, 1379, 1384, 1385, 1386, 1387,
                       1392, 1393, 1394, 1395, 1396, 1397, 1400, 1401, 1402, 1431, 1454,
                       1455, 1462, 1471, 1125], dtype='i4')
charges    = np.array([ 2.66531368,  3.73287782,  5.2828832 ,  2.31351022,  4.41027829,
                        2.20720324,  3.98584685, 14.72511416, 18.30488489,  6.26837369,
                        2.96073255, 14.59842807, 23.50427029,  6.8357565 ,  3.92635856,
                        7.65335644,  2.09604995,  3.67059242,  9.99260022, 10.94429901,
                        2.83670564,  2.89119827,  2.57772977,  2.82898736,  2.83071036,
                        2.26349407], dtype='f4')

DataSiPM = dbf.DataSiPM(run_number)
DataPMT  = dbf.DataPMT(run_number)

#################
#################
# Create voxels #
#################
#################
dist = 20.
sipm_dist = np.float32(20.)
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

# #### Call CUDA kernel
import pycuda.driver as cuda
#import pycuda.autoinit
import numpy
from pycuda.compiler import SourceModule

#create context
from pycuda.tools import make_default_context
cuda.init()
ctx = make_default_context()

#compile cuda code
kernel_code = open('reset_non_compact.cu').read()
mod = SourceModule(kernel_code)
create_voxels = mod.get_function("create_voxels")

#TODO: Check rounding here
threads_x = int((xmax - xmin) / xsize)
threads_y = int((ymax - ymin) / ysize)
print(threads_x, threads_y)

#allocate memory for result
voxels_dt = np.dtype([('x', 'f4'), ('y', 'f4'), ('E', 'f4'), ('active', 'i4')])
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
kernel_code = open('reset_non_compact.cu').read()
mod = SourceModule(kernel_code)
initiliaze_anode = mod.get_function("initialize_anode")

xs = DataSiPM.X.values.astype('f4')
ys = DataSiPM.Y.values.astype('f4')
x_d = cuda.to_device(xs)
y_d = cuda.to_device(ys)

#allocate memory for result
# due to packing the c struct is 12 bytes instead of 9.
# Mem layout to be updated, maybe pragma pack
sensors_dt = np.dtype([('id', 'i4'), ('charge', 'f4'), ('active', 'i4')])
sensors_d = cuda.mem_alloc(nsipms * sensors_dt.itemsize)

initiliaze_anode(sensors_d, xmin, xmax, x_d, ymin, ymax, y_d, sipm_dist, block=(1, 1, 1), grid=(nsipms, 1))

#### Step 2: Put the correct charge for active sensors
kernel_code = open('reset_non_compact.cu').read()
mod = SourceModule(kernel_code)
create_anode_response = mod.get_function("create_anode_response")

sensor_ids_d = cuda.to_device(sensor_ids)
charges_d    = cuda.to_device(charges)

nsensors = sensor_ids.shape[0]
create_anode_response(sensors_d, sensor_ids_d, charges_d, block=(1, 1, 1), grid=(nsensors, 1))

##########################
##########################
## Select active sensors #
##########################
##########################
active_dt = np.dtype([('id', 'i1')])
active_d = cuda.mem_alloc(num_voxels * nsipms) # TODO: Update after compact
#probabilities = cuda.mem_alloc(nvoxels * nsensors)

kernel_code = open('reset_non_compact.cu').read()
mod = SourceModule(kernel_code)
compute_active_sensors = mod.get_function("compute_active_sensors")

compute_active_sensors(active_d, sensors_d, voxels_d, x_d, y_d, np.int32(nsipms), sipm_dist, block=(1, 1, 1), grid=(num_voxels, 1))


ctx.detach()
