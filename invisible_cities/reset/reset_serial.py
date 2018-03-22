import invisible_cities.reset.reset_utils as reset

import numpy as np
import math

from invisible_cities.core.system_of_units import pes, mm, mus, ns
import invisible_cities.reco.corrections    as corrf
import invisible_cities.database.load_db as dbf
import reset_functions_event as rstf

from invisible_cities.evm.ic_containers import ResetVoxels

def create_voxels(voxels_data, slices_start, xsize, ysize, rmax):
    max_voxels = slices_start[-1]
    nslices = voxels_data.xmin.shape[0]

    nvoxels = 0

    voxels_dt  = np.dtype([('x', 'f4'), ('y', 'f4'), ('E', 'f4')])
    voxels     = np.empty(max_voxels, dtype=voxels_dt)
    slice_ids  = np.empty(max_voxels, dtype='i4')
    # One extra position to have the correct addresses after cumsum
    address    = np.zeros(max_voxels + 1, dtype='i4')

    slice_start_voxels = np.zeros(nslices+1, dtype='i4')

    for slice_id in range(nslices):
        xmin   = voxels_data.xmin[slice_id];
        xmax   = voxels_data.xmax[slice_id];
        ymin   = voxels_data.ymin[slice_id];
        ymax   = voxels_data.ymax[slice_id];
        charge = voxels_data.charge[slice_id];
        xsteps = (xmax - xmin) / xsize + 1;
        ysteps = (ymax - ymin) / ysize + 1;

        xs = np.linspace(xmin, xmax, xsteps)
        ys = np.linspace(ymin, ymax, ysteps)

        voxels_nc = np.array([(x, y, charge) for x in xs for y in ys], dtype=voxels_dt)
        check_actives = np.vectorize(lambda v: np.sqrt(v[0]**2 + v[1]**2) < rmax, otypes=[bool])
        actives = check_actives(voxels_nc)
        nactive = actives.sum()
        start = slice_start_voxels[slice_id]
        end = start + nactive
        voxels[start:end] = voxels_nc[actives]
        slice_ids[start:end] = slice_id

        slice_start_voxels[slice_id+1] = end
        nvoxels = nvoxels + nactive

        print(slice_id, nactive, end)

        addr_start = slices_start[slice_id]     + 1
        addr_end   = slices_start[slice_id + 1] + 1
        address[addr_start:addr_end] = actives

    address = address.cumsum()

    reset_voxels = ResetVoxels(nvoxels, voxels[:nvoxels],
                               slice_ids[:nvoxels], slice_start_voxels,
                               address)

    return reset_voxels

def create_anode_response(nslices, nsensors, slices):
    total_sensors = nslices * nsensors
    anode_response = np.zeros(total_sensors, dtype='f4')

    slice_id = 0
    ncharges = slices.charges.shape[0]

    for s in range(ncharges):
        if s >= slices.start[slice_id+1]:
            slice_id = slice_id + 1

            position = slices.sensors[s] + nsensors * slice_id
            anode_response[position] = slices.charges[s]

    return anode_response

def compute_probabilities(voxels, nvoxels, xs, ys, nsensors, sensors_per_voxel, sensor_dist, sensor_param, sensor_response):
    probs_size = nvoxels * sensors_per_voxel
    sensor_ids = np.empty(probs_size, dtype='i4')
    probs      = np.empty(probs_size, dtype='f4')
    fwd_num    = np.empty(probs_size, dtype='f4')
    voxel_starts = np.zeros(nvoxels+1, dtype='i4')

    sensor_starts = np.zeros(nsensors * nslices + 1, dtype='i4')

    last_position = 0
    for i, v in enumerate(voxels):
        for s in range(nsensors):
            xdist = v[0] - xs[s]
            ydist = v[1] - ys[s]
            active = ((abs(xdist) <= sensor_dist) and (abs(ydist) <= sensor_dist))
            if active:
                #print(v, s, xdist, ydist)
                sens_idx = slice_ids[i] * nsensors + s
                sensor_starts[sens_idx + 1] = sensor_starts[sens_idx + 1] + 1

                #sensor_ids[last_position] = s
                sensor_ids[last_position] = sens_idx
                prob                      = sensor_param(xdist, ydist).value
                probs     [last_position] = prob
                fwd_num   [last_position] = prob * sensor_response[sens_idx]
                last_position = last_position + 1

            voxel_starts[i+1] = last_position

        sensor_starts = sensor_starts.cumsum()

    return probs[:last_position], sensor_ids[:last_position], voxel_starts, sensor_starts, last_position, fwd_num[:last_position]

def compute_sensor_probs(probs, nprobs, nsensors, voxel_starts, sensor_ids, slice_ids):
    # offset one position to get the ending position of the last element
    sensor_counts = np.zeros(nsensors * nslices + 1, dtype='i4')
    voxel_ids    = np.empty(nprobs, dtype='i4')
    sensor_probs = np.empty(nprobs, dtype='f4')

    vidx = 0
    for i, p in enumerate(probs):
        if i >= voxel_starts[vidx + 1]:
            vidx = vidx + 1

        sidx     = sensor_ids[i]
        slice_id = slice_ids[vidx]

        count = sensor_counts[sidx + 1]
        pos   = sensor_starts[sidx] + count

        sensor_probs[pos] = p
        voxel_ids[pos] = vidx

        sensor_counts[sidx + 1] = count + 1

    active_sensors = sensor_counts > 0
    sensor_starts_c_ids = np.where(active_sensors)[0] - 1
    sensor_starts_c = sensor_counts.cumsum()[np.concatenate((sensor_starts_c_ids, [-1]))]

    return sensor_probs, sensor_starts, sensor_starts_c, sensor_starts_c_ids

def forward_denoms(nsensors, nslices, voxels, sensor_probs, voxel_ids, sensor_starts, sensor_starts_ids):
    nsensor_active = sensor_starts_ids.shape[0]
    denoms = np.zeros(nsensors * nslices, dtype='f4')

    for id in np.arange(nsensor_active):
        start = sensor_starts[id]
        end   = sensor_starts[id+1]

        denom = 0.
        for i in np.arange(start, end):
            vid = voxel_ids[i]
            denom += voxels[vid][2] * sensor_probs[i]

        sid = sensor_starts_c_ids[id]
        denoms[sid] = denom

    return denoms

def mlem_step(voxels, nvoxels, voxel_starts, probs, sensor_ids, fwd_num, fwd_denoms):
    #for vidx in np.arange(nvoxels):
    for vidx in np.arange(1000):
        start = voxel_starts[vidx]
        end   = voxel_starts[vidx+1]

        eff = 0
        fwd = 0
        for i in np.arange(start, end):
            eff += probs[i]
            sidx = sensor_ids[i]

            value = fwd_num[i] / denoms[sidx]

            if(np.isfinite(value)):
                fwd += value

        voxels[vidx][2] = voxels[vidx][2] / eff * fwd
