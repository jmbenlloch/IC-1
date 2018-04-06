import pycuda.driver as cuda
import numpy as np

from invisible_cities.evm.ic_containers import ResetSnsProbs
from invisible_cities.evm.ic_containers import ResetProbs2
from invisible_cities.evm.ic_containers import Voxels
from invisible_cities.evm.ic_containers import ResetVoxels
from invisible_cities.evm.ic_containers import ResetSlices


def copy_voxels_data_h2d(voxels):
    nslices       = np.int32      (voxels.nslices)
    xmin_d        = cuda.to_device(voxels.xmin)
    xmax_d        = cuda.to_device(voxels.xmax)
    ymin_d        = cuda.to_device(voxels.ymin)
    ymax_d        = cuda.to_device(voxels.ymax)
    charges_avg_d = cuda.to_device(voxels.charge)

    voxels_data_d = Voxels(nslices, xmin_d, xmax_d, ymin_d, ymax_d, charges_avg_d)
    return voxels_data_d

def copy_slice_data_h2d(slices):
    nslices        = np.int32      (slices.nslices)
    nsensors       = np.int32      (slices.nsensors)
    ncharges       = np.int32      (slices.ncharges)
    sensors_ids_d  = cuda.to_device(slices.sensors)
    charges_d      = cuda.to_device(slices.charges)
    slices_start_d = cuda.to_device(slices.start)

    slices_data_d = ResetSlices(nslices, nsensors, ncharges, slices_start_d, sensors_ids_d, charges_d)
    return slices_data_d

def copy_voxels_d2h(rst_voxels_d, nslices):
    nvoxels = rst_voxels_d.nvoxels

    voxels_h  = cuda.from_device(rst_voxels_d.voxels,      (nvoxels,),   voxels_dt)
    ids_h     = cuda.from_device(rst_voxels_d.slice_ids,   (nvoxels,),   np.dtype('i4'))
    start_h   = cuda.from_device(rst_voxels_d.slice_start, (nslices+1,), np.dtype('i4'))
    address_h = rst_voxels_d.address.get()

    rst_voxels = ResetVoxels(nvoxels, voxels_h, ids_h, start_h, address_h)
    return rst_voxels

def copy_probs_d2h(probs_d, probs_size, nvoxels, nslices, nsensors):
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


def copy_sensor_probs_d2h(sns_probs_d, probs_size, active_sensors):
    sns_probs_h = cuda.from_device(sns_probs_d.probs,     (int(probs_size),), np.dtype('f4'))
    voxel_ids_h = cuda.from_device(sns_probs_d.voxel_ids, (int(probs_size),), np.dtype('i4'))

    sensor_starts_h     = cuda.from_device(sns_probs_d.sensor_start,     (int(active_sensors+1),), np.dtype('i4'))
    sensor_starts_ids_h = cuda.from_device(sns_probs_d.sensor_start_ids, (int(active_sensors),), np.dtype('i4'))

    sensor_probs = ResetSnsProbs(sns_probs_h, voxel_ids_h, sensor_starts_h, sensor_starts_ids_h)
    return sensor_probs

