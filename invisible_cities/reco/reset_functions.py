import numpy  as np
import invisible_cities.reco.corrections    as corrf
import numba
#from invisible_cities.reco.corrections.corrections import Correction

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
            voxE.append(sens_q.mean()) # Why mean?
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

def ComputeCathForward(vox, cath_response, pmt_prob):
    cathForward = []
    for sensor in range(len(cath_response)):
        cathForward.append(np.sum(vox[2]*(pmt_prob[:,sensor])))
    return np.array(cathForward)

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

    for j in range(len(oldVox[0])):
        if oldVox[2][j] <= 0: # Why this? Can E be less than 0?
            newVoxE.append(0.)
            newVoxX.append(oldVox[0,j])
            newVoxY.append(oldVox[1,j])
            continue

        efficiency = 0
        anWeight = 0
        cathWeight = 0

        if fAnode:
            selS = selSens[j]
            if np.sum(anode_response[selS,1]) == 0.: # Check when can this happen
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

        if(newValue < eThres): #What is this for? Is eThres always zero?
            newVoxE.append(0.)
            newVoxX.append(oldVox[0,j])
            newVoxY.append(oldVox[1,j])
            print("Negative weight, revise code and probability model")
            continue

        newVoxE.append(newValue)
        newVoxX.append(oldVox[0,j])
        newVoxY.append(oldVox[1,j])

    return np.array([newVoxX, newVoxY, newVoxE])

def computeDiff(DataSiPM, oldVox, anode_response):
    sensx = DataSiPM.X[anode_response[:,0]].values
    sensy =  DataSiPM.Y[anode_response[:,0]].values
    voxD = np.array([[oldVox[0][j] - sensx, oldVox[1][j] - sensy] for j in range(len(oldVox[0]))])
    return voxD[:,0], voxD[:,1]

@numba.autojit
def ComputeAnodeForward(voxDX, voxDY, vox, anode_response, sipm_dist, xy_prob, selVox):
    dim = len(anode_response)
    anodeForward = np.zeros(dim)
    for sensor in range(dim):
        selV = selVox[sensor]
        anodeForward[sensor] = (np.sum(vox[2][selV]*xy_prob[selV,sensor]))
    return anodeForward

def computeProb(pmt_xy_map, sipm_xy_map, voxDX, voxDY, voxX, voxY):
    xyprob = [sipm_xy_map(voxDX[j], voxDY[j]).value for j in range(len(voxDX))]
    pmtprob = []
    for j in range(len(voxDX)):
        pmtprob.append([pmt_xy_map[i](voxX[j], voxY[j]).value[0] for i in range(len(pmt_xy_map))])
    return np.array(xyprob), np.array(pmtprob)

@numba.autojit
def createSel(voxDX, voxDY, anode_response, sipm_dist):
    selVox = []
    selSens = []
    for sensor in range(len(anode_response)):
        selVox.append( (np.abs(voxDX[:,sensor]) <= sipm_dist) & (np.abs(voxDY[:,sensor]) <= sipm_dist) )
    for voxel in range(len(voxDX)):
        selSens.append( (np.abs(voxDX[voxel]) <= sipm_dist) & (np.abs(voxDY[voxel]) <= sipm_dist) )
    return np.array(selVox), np.array(selSens)

def pmtCorrections(j, oldVox, pmt_xy_map, n_pmts):
    pmtcorrection = np.zeros(n_pmts)
    for i in range(n_pmts):
        pmtcorrection[i] = pmt_xy_map[i](oldVox[0][j],oldVox[1][j]).value
    return pmtcorrection
