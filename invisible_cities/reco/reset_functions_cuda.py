import numpy as np
import numba

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
    #TODO try to avoid using lists before creating the numpy array
    data = np.empty(len(voxX), dtype={'names':('x', 'y', 'E'),
                          'formats':('f4', 'f4', 'f4')})
    data['x'] = voxX
    data['y'] = voxY
    data['E'] = voxE
    return data


def CreateSiPMresponse(DataSiPM, sens_id, sens_q, sipm_dist, sipm_thr, vox):
    ids     = []
    charges = []
    selDB = (DataSiPM.X.values >= vox['x'].min()-sipm_dist) & (DataSiPM.X.values <= vox['x'].max()+sipm_dist)
    selDB = selDB & (DataSiPM.Y.values >= vox['y'].min()-sipm_dist) & (DataSiPM.Y.values <= vox['y'].max()+sipm_dist)
    for ID in DataSiPM[selDB].index.values:
        sel = (sens_id==ID)
        ids.append(ID)
        if sens_q[sel] > sipm_thr:
            charges.append(sens_q[sel])
        else:
            charges.append(0)

    data = np.empty(len(ids), dtype={'names':('id', 'charge'),
                          'formats':('i4', 'f4')})
    data['id']     = ids
    data['charge'] = charges
    return data


def computeDiff(DataSiPM, oldVox, anode_response):
    sensx = DataSiPM.X[anode_response['id']].values
    sensy =  DataSiPM.Y[anode_response['id']].values
    voxD = np.array([[oldVox[0][j] - sensx, oldVox[1][j] - sensy] for j in range(len(oldVox[0]))], dtype='f4')
    return voxD[:,0], voxD[:,1]


def computeProb(pmt_xy_map, sipm_xy_map, voxDX, voxDY, voxX, voxY):
    xyprob = [sipm_xy_map(voxDX[j], voxDY[j]).value for j in range(len(voxDX))]
    pmtprob = []
    for j in range(len(voxDX)):
        pmtprob.append([pmt_xy_map[i](voxX[j], voxY[j]).value[0] for i in range(len(pmt_xy_map))])
    return np.array(xyprob, dtype='f4'), np.array(pmtprob, dtype='f4')


@numba.autojit
def createSel(voxDX, voxDY, anode_response, sipm_dist):
    selSens = []
    for voxel in range(len(voxDX)):
        selSens.append( (np.abs(voxDX[voxel]) <= sipm_dist) & (np.abs(voxDY[voxel]) <= sipm_dist) )
    return np.array(selSens)
