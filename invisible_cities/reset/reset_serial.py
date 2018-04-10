import numpy as np
import math
import tables as tb
from invisible_cities.evm.ic_containers import SensorsParams
from invisible_cities.evm.ic_containers import ResetProbs
from invisible_cities.evm.ic_containers import ResetSnsProbs

from invisible_cities.core.system_of_units import pes, mm, mus, ns
import invisible_cities.reco.corrections    as corrf
import invisible_cities.database.load_db as dbf

from invisible_cities.evm.ic_containers import ResetVoxels


def create_voxels(voxels_data, slices_start, xsize, ysize, rmax):
    max_voxels = slices_start[-1]

    nvoxels = 0

    voxels_dt  = np.dtype([('x', 'f4'), ('y', 'f4'), ('E', 'f4')])
    voxels     = np.empty(max_voxels, dtype=voxels_dt)
    slice_ids  = np.empty(max_voxels, dtype='i4')
    # One extra position to have the correct addresses after cumsum
    address    = np.zeros(max_voxels + 1, dtype='i4')

    slice_start_voxels = np.zeros(voxels_data.nslices+1, dtype='i4')

    for slice_id in range(voxels_data.nslices):
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

        addr_start = slices_start[slice_id]     + 1
        addr_end   = slices_start[slice_id + 1] + 1
        address[addr_start:addr_end] = actives

    address = address.cumsum()

    reset_voxels = ResetVoxels(voxels_data.nslices, nvoxels, voxels[:nvoxels],
                               slice_ids[:nvoxels], slice_start_voxels,
                               address)

    return reset_voxels


def create_anode_response(slices):
    total_sensors = slices.nslices * slices.nsensors
    anode_response = np.zeros(total_sensors, dtype='f4')

    slice_id = 0

    for s in range(slices.ncharges):
        if s >= slices.start[slice_id+1]:
            slice_id = slice_id + 1

        position = slices.sensors[s] + slices.nsensors * slice_id
        anode_response[position] = slices.charges[s]

    return anode_response

def get_probability(xdist, ydist, sensor_param):
    xindex = math.floor((xdist - sensor_param.xmin) / sensor_param.step + 0.5)
    yindex = math.floor((ydist - sensor_param.ymin) / sensor_param.step + 0.5)
    prob_idx = xindex * sensor_param.nbins + yindex
    return sensor_param.params[prob_idx][2]

def compute_probabilities(voxels, xs, ys, nsensors, sensors_per_voxel, sensor_dist, sensor_param, sensor_response):
    probs_size = voxels.nvoxels * sensors_per_voxel
    sensor_ids = np.empty(probs_size, dtype='i4')
    probs      = np.empty(probs_size, dtype='f4')
    fwd_num    = np.empty(probs_size, dtype='f4')
    voxel_starts = np.zeros(voxels.nvoxels+1, dtype='i4')

    sensor_starts = np.zeros(nsensors * voxels.nslices + 1, dtype='i4')

    last_position = 0
    for i, v in enumerate(voxels.voxels):
        for s in range(nsensors):
            xdist = v[0] - xs[s]
            ydist = v[1] - ys[s]
            active = ((abs(xdist) <= sensor_dist) and (abs(ydist) <= sensor_dist))
            if active:
                #print(v, s, xdist, ydist)
                sens_idx = voxels.slice_ids[i] * nsensors + s
                sensor_starts[sens_idx + 1] = sensor_starts[sens_idx + 1] + 1

                #sensor_ids[last_position] = s
                sensor_ids[last_position] = sens_idx
                prob = get_probability(xdist, ydist, sensor_param)
                probs     [last_position] = prob
                fwd_num   [last_position] = prob * sensor_response[sens_idx]
                last_position = last_position + 1

            voxel_starts[i+1] = last_position

    sensor_starts = sensor_starts.cumsum()

    probs = ResetProbs(last_position, probs[:last_position],
                        sensor_ids[:last_position], voxel_starts,
                        sensor_starts, fwd_num[:last_position])

    return probs


def compute_sensor_probs(probs, nslices, nsensors, slice_ids):
    # offset one position to get the ending position of the last element
    sensor_counts = np.zeros(nsensors * nslices + 1, dtype='i4')
    voxel_ids    = np.empty(probs.nprobs, dtype='i4')
    sensor_probs = np.empty(probs.nprobs, dtype='f4')

    vidx = 0
    for i, p in enumerate(probs.probs):
        if i >= probs.voxel_start[vidx + 1]:
            vidx = vidx + 1

        sidx     = probs.sensor_ids[i]
        slice_id = slice_ids[vidx]

        count = sensor_counts[sidx + 1]
        pos   = probs.sensor_start[sidx] + count

        sensor_probs[pos] = p
        voxel_ids[pos] = vidx

        sensor_counts[sidx + 1] = count + 1

    active_sensors = sensor_counts > 0
    sensor_starts_c_ids = np.where(active_sensors)[0] - 1
    sensor_starts_c = sensor_counts.cumsum()[np.concatenate((sensor_starts_c_ids, [-1]))]

    nsensors = active_sensors.sum()
    sns_probs = ResetSnsProbs(sensor_probs, voxel_ids,
                              nsensors, sensor_starts_c, sensor_starts_c_ids)
    return sns_probs


def forward_denoms(nsensors, nslices, voxels, sns_probs):
    denoms = np.zeros(nsensors * nslices, dtype='f4')

    for id in np.arange(sns_probs.nsensors):
        start = sns_probs.sensor_start[id]
        end   = sns_probs.sensor_start[id+1]

        denom = 0.
        for i in np.arange(start, end):
            vid = sns_probs.voxel_ids[i]
            denom += voxels[vid][2] * sns_probs.probs[i]

        sid = sns_probs.sensor_start_ids[id]
        denoms[sid] = denom

    return denoms


def mlem_step(voxels, sipm_probs, sipm_fwd_denoms, pmt_probs, pmt_fwd_denoms):
    for vidx in np.arange(voxels.nvoxels):
        sipm_start = sipm_probs.voxel_start[vidx]
        sipm_end   = sipm_probs.voxel_start[vidx+1]

        sipm_eff = 0
        sipm_fwd = 0

        for i in np.arange(sipm_start, sipm_end):
            sipm_eff += sipm_probs.probs[i]
            sidx = sipm_probs.sensor_ids[i]

            value = sipm_probs.fwd_nums[i] / sipm_fwd_denoms[sidx]

            if(np.isfinite(value)):
                sipm_fwd += value

        pmt_start = pmt_probs.voxel_start[vidx]
        pmt_end   = pmt_probs.voxel_start[vidx+1]

        pmt_eff = 0
        pmt_fwd = 0

        for i in np.arange(pmt_start, pmt_end):
            pmt_eff += pmt_probs.probs[i]
            sidx = pmt_probs.sensor_ids[i]

            value = pmt_probs.fwd_nums[i] / pmt_fwd_denoms[sidx]

            if(np.isfinite(value)):
                pmt_fwd += value

        voxels.voxels[vidx][2] = voxels.voxels[vidx][2] / (sipm_eff + pmt_eff) * (sipm_fwd + pmt_fwd)

def compute_mlem(iterations, voxels, nsipms, sipm_probs, sipm_sns_probs, npmts, pmt_probs, pmt_sns_probs):
    for i in np.arange(iterations):
        sipm_fwd_denoms = forward_denoms(nsipms, voxels.nslices, voxels.voxels, sipm_sns_probs)
        pmt_fwd_denoms  = forward_denoms(npmts,  voxels.nslices, voxels.voxels, pmt_sns_probs)
        mlem_step(voxels, sipm_probs, sipm_fwd_denoms, pmt_probs, pmt_fwd_denoms)
