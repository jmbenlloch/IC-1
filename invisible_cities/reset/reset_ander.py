import numpy as np
import tables as tb
import numba
import scipy

from invisible_cities.evm.ic_containers import ResetProbs
from invisible_cities.evm.ic_containers import ResetSnsProbs


def CreateVoxels(DataSiPM, sens_id, sens_q, point_dist, sipm_thr, sizeX, sizeY, rMax):
    voxX = []
    voxY = []
    voxE = []
    dist = point_dist
    selC = (sens_q > sipm_thr)
    rangex = np.arange(DataSiPM.X[sens_id[selC]].values.min()-dist, DataSiPM.X[sens_id[selC]].values.max()+dist, sizeX)
    rangey = np.arange(DataSiPM.Y[sens_id[selC]].values.min()-dist, DataSiPM.Y[sens_id[selC]].values.max()+dist, sizeY)
    for x in rangex:
        for y in rangey:
            if(np.sqrt(x*x+y*y) > rMax):
                continue
            voxX.append(x)
            voxY.append(y)
            voxE.append(sens_q.mean())
    return np.array([voxX, voxY, voxE])


def CreateSiPMresponse(DataSiPM, sens_id, sens_q, sipm_dist, sipm_thr, vox):
    sens_response = []
    selDB = (DataSiPM.X.values >= vox[0].min()-sipm_dist) & (DataSiPM.X.values <= vox[0].max()+sipm_dist)
    selDB = selDB & (DataSiPM.Y.values >= vox[1].min()-sipm_dist) & (DataSiPM.Y.values <= vox[1].max()+sipm_dist)
    for ID in DataSiPM[selDB].index.values:
        sel = (sens_id==ID)
        if sens_q[sel] > sipm_thr:
            sens_response.append([ID, sens_q[sel]])
        else:
            sens_response.append([ID,0.])
    return np.array(sens_response)


def computeDiff(DataSiPM, oldVox, anode_response):
    sensx = DataSiPM.X[anode_response[:,0]].values
    sensy = DataSiPM.Y[anode_response[:,0]].values
    voxD = np.array([[oldVox[0][j] - sensx, oldVox[1][j] - sensy] for j in range(len(oldVox[0]))])
    return voxD[:,0], voxD[:,1]


@numba.autojit
def createSel(voxDX, voxDY, anode_response, sipm_dist):
    selVox = []
    selSens = []
    for sensor in range(len(anode_response)):
        selVox.append( (np.abs(voxDX[:,sensor]) <= sipm_dist) & (np.abs(voxDY[:,sensor]) <= sipm_dist) )
    for voxel in range(len(voxDX)):
        selSens.append( (np.abs(voxDX[voxel]) <= sipm_dist) & (np.abs(voxDY[voxel]) <= sipm_dist) )
    return np.array(selVox), np.array(selSens)


def computeProb(pmt_xy_map, sipm_xy_map, voxDX, voxDY, voxX, voxY):
    xyprob = [sipm_xy_map(voxDX[j], voxDY[j]).value for j in range(len(voxDX))]
    pmtprob = []
    for j in range(len(voxDX)):
        pmtprob.append([pmt_xy_map(voxX[j], voxY[j]).value])
    return np.array(xyprob), np.array(pmtprob)


def ComputeCathForward(vox, cath_response, pmt_prob):
    cathForward = []
    for sensor in range(len(cath_response)):
        cathForward.append(np.sum(vox[2]*(pmt_prob[:,sensor])))
    return np.array(cathForward)


@numba.autojit
def ComputeAnodeForward(voxDX, voxDY, vox, anode_response, sipm_dist, xy_prob, selVox):
    dim = len(anode_response)
    anodeForward = np.zeros(dim)
    for sensor in range(dim):
        selV = selVox[sensor]
        anodeForward[sensor] = (np.sum(vox[2][selV]*xy_prob[selV,sensor]))
    return anodeForward


def MLEM_step(voxDX, voxDY, oldVox, selVox, selSens, anode_response, cath_response, pmt_prob, xy_prob, sipm_dist=20., eThres=0., fCathode = True, fAnode = True):
    newVoxE = []
    newVoxX = []
    newVoxY = []

    anodeForward = 0
    cathForward = 0

    if fAnode:
        anodeForward = ComputeAnodeForward(voxDX, voxDY, oldVox, anode_response, sipm_dist, xy_prob, selVox)
    if fCathode:
        cathForward = ComputeCathForward(oldVox, cath_response, pmt_prob)

    likelihood = 0
    if fCathode:
        cath_lhood  = compute_likelihood(cathForward, cath_response)
        likelihood += cath_lhood
    if fAnode:
        anode_lhood = compute_likelihood(anodeForward, anode_response)
        likelihood += anode_lhood

    for j in range(len(oldVox[0])):
        if oldVox[2][j] <= 0:
            print("if1, voxel {} is 0".format(j))
            print("\t {}".format(oldVox[2][j]))
            newVoxE.append(0.)
            newVoxX.append(oldVox[0,j])
            newVoxY.append(oldVox[1,j])
            continue

        efficiency = 0
        anWeight = 0
        cathWeight = 0

        if fAnode:
            selS = selSens[j]
            if np.sum(anode_response[selS,1]) == 0.:
                print("if2, voxel {} is 0, selS.sum: {}".format(j, selS.sum()))
                newVoxE.append(0.)
                newVoxX.append(oldVox[0,j])
                newVoxY.append(oldVox[1,j])
                continue
            sipmCorr = xy_prob[j]
            anWeight += np.sum( (anode_response[:,1]*sipmCorr/anodeForward)[selS] )
            efficiency += np.sum(sipmCorr[selS])
        if fCathode:
            pmtCorr = pmt_prob[j]
            cathWeight += np.sum(cath_response*pmtCorr/cathForward)
            efficiency += np.sum(pmtCorr)

        newValue = oldVox[2][j]*(anWeight+cathWeight)/efficiency

        if(newValue < eThres):
            print("if3, voxel {} is 0".format(j))
            newVoxE.append(0.)
            newVoxX.append(oldVox[0,j])
            newVoxY.append(oldVox[1,j])
            print("Negative weight, revise code and probability model")
            continue

        newVoxE.append(newValue)
        newVoxX.append(oldVox[0,j])
        newVoxY.append(oldVox[1,j])

    return np.array([newVoxX, newVoxY, newVoxE]), likelihood


def compute_likelihood(forward_proj, sns_response):
    if sns_response.ndim > 1:
        sns_response = sns_response[:,1]
    likelihood = -forward_proj + sns_response * np.log(forward_proj) - np.real(scipy.special.loggamma(sns_response + 1))

    nans = np.isinf(likelihood) + np.isnan(likelihood)
    likelihood[nans] = 0
    return likelihood.sum()


def build_sipm_probs_serial(selSens, selVox, probs_serial, anode):
    nprobs = selSens.sum().sum()

    voxel_start = np.zeros(selSens.shape[0]+1, dtype=np.int32)
    voxel_start[1:] = selSens.sum(axis=1).cumsum()

    sensor_ids = np.zeros(nprobs, dtype=np.int32)
    for i in range(selSens.shape[0]):
        vstart = voxel_start[i  ]
        vend   = voxel_start[i+1]
        sensor_ids[vstart:vend] = np.where(selSens[i])[0]

    sensor_start = np.zeros(selVox.shape[0]+1, dtype=np.int32)
    sensor_start[1:] = selVox.sum(axis=1).cumsum()

    probs = np.zeros(nprobs, dtype=np.float64)
    for i in range(selSens.shape[0]):
        start = voxel_start[i  ]
        end   = voxel_start[i+1]
        probs[start:end] = probs_serial[i, sensor_ids[start:end]]

    fwd_nums = (anode[sensor_ids][:,1] * probs).astype(np.float64)
    probs_serial_h = ResetProbs(nprobs=np.int32(probs.shape[0]),
                                 probs=probs,
                                 sensor_ids=sensor_ids,
                                 voxel_start=voxel_start,
                                 sensor_start=sensor_start,
                                 fwd_nums=fwd_nums)

    return probs_serial_h


def build_pmt_probs_serial(probs_serial, energies):
    nprobs = probs_serial.shape[0]
    probs = probs_serial[:,0].astype(np.float64)
    sensor_ids = np.zeros(nprobs, dtype=np.int32)
    voxel_start = np.arange(0, nprobs+1, dtype=np.int32)
    sensor_start = np.array([0, nprobs], dtype=np.int32)
    fwd_nums = energies * probs

    probs_serial_h = ResetProbs(nprobs=np.int32(nprobs),
                                 probs=probs,
                                 sensor_ids=sensor_ids,
                                 voxel_start=voxel_start,
                                 sensor_start=sensor_start,
                                 fwd_nums=fwd_nums)
    return probs_serial_h


def build_pmt_sns_probs_serial(probs_h, nvoxels):
    probs = probs_h.probs

    voxel_ids = np.zeros(nvoxels)
    for i in np.arange(nvoxels):
        start = probs_h.voxel_start[i  ]
        end   = probs_h.voxel_start[i+1]
        if end > start:
            voxel_ids[i] = True
    voxel_ids = np.where(voxel_ids)[0]

    nsensors = 1
    sensor_start = np.array([0, len(probs)], dtype=np.int32)
    sensor_start_ids = np.array([0], dtype=np.int32)

    sns_probs_h = ResetSnsProbs(probs            = probs,
                                voxel_ids        = voxel_ids,
                                nsensors         = np.int32(nsensors),
                                sensor_start     = sensor_start,
                                sensor_start_ids = sensor_start_ids)

    return sns_probs_h


def build_sipm_sns_probs_serial(probs_h):
    probs     = np.zeros(probs_h.nprobs, dtype=np.float64)
    voxel_ids = np.zeros(probs_h.nprobs, dtype=np.int32)
    sensor_start = probs_h.sensor_start.copy()
    nsensors = sensor_start.shape[0] -1

    sensor_offset    = np.zeros_like(probs_h.sensor_start)
    sensor_start_ids = np.zeros_like(probs_h.sensor_start)

    voxel_idx = 0

    for idx, p in enumerate(probs_h.probs):
        if idx >= probs_h.voxel_start[voxel_idx+1]:
            voxel_idx = voxel_idx + 1

        sid = probs_h.sensor_ids[idx]

        sensor_idx = sensor_start[sid] + sensor_offset[sid]
        sensor_offset[sid] = sensor_offset[sid] + 1

        probs    [sensor_idx] = p
        voxel_ids[sensor_idx] = voxel_idx
        sensor_start_ids[sid] = sid

    sns_probs_h = ResetSnsProbs(probs            = probs,
                                voxel_ids        = voxel_ids,
                                nsensors         = np.int32(nsensors),
                                sensor_start     = sensor_start,
                                sensor_start_ids = sensor_start_ids)

    return sns_probs_h

