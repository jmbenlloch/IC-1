import glob

#data files
run_number = 4495#"MC/Kr"
map_run_number = 4446
outputfolder = '/home/jmbenlloch/reset_data/4495/dsts'
inputfiles = glob.glob("/home/jmbenlloch/reset_data/4495/pmaps/pmaps*.root.h5")


#  ic imports
import os
import functools
import itertools
import time
print("Running on ", time.asctime())

import numpy             as np
import tables            as tb
import matplotlib.pyplot as plt

import invisible_cities.database.load_db as dbf
import invisible_cities.filters.s1s2_filter as s1s2filt
import invisible_cities.reco.pmaps_functions  as pmapsf
import invisible_cities.reco.pmaps_functions_c  as pmapsfc
import invisible_cities.reco.corrections    as corrf
import invisible_cities.core.core_functions as coref
import invisible_cities.core.core_functions_c as corefc
import invisible_cities.core.fit_functions  as fitf
import invisible_cities.reco.dst_functions as dstf
import invisible_cities.reco.reset_functions as rstf
#import invisible_cities.reco.reset_functions_old as rstfOld
import invisible_cities.io.table_io as tbiof
import invisible_cities.io.pmap_io as pmapio
import invisible_cities.io.dst_io as dstio
import invisible_cities.reco.xy_algorithms as xyf

from invisible_cities.icaro.hst_functions import hist
from invisible_cities.icaro.hst_functions import hist2d
from invisible_cities.icaro.hst_functions import hist2d_profile
from invisible_cities.icaro.hst_functions import labels
from invisible_cities.icaro.hst_functions import gausstext

from invisible_cities.reco.tbl_functions     import get_event_numbers_and_timestamps_from_file_name
from invisible_cities.core.system_of_units import pes, mm, mus, ns

from invisible_cities.evm.pmaps import S2, S2Si, Peak

# sensors
DataSiPM = dbf.DataSiPM(run_number)
DataPMT = dbf.DataPMT(run_number)


# Function to calculate lifetime
def lifetime_calculation(z, zrange, energy, erange):
    hist(energy, 50, erange)
    labels("S2 energy (pes)", "Entries", "Fidualized energy spectrum")

    low_cut   = 11000 * np.exp(-0.0009*z)
    high_cut  = 14000 * np.exp(-0.0009*z)
    sel       = coref.in_range(energy, low_cut, high_cut) # remove low and high E background

    hist2d(z, energy, (100, 50), range=(zrange, erange))
    x, y, u_y = fitf.profileX(z[sel], energy[sel], 100, xrange=zrange, yrange=erange)

    plt.plot(x, y, profOpt)
    plt.plot(z,  low_cut, "k.")
    plt.plot(z, high_cut, "k.")

    Zrange_LT = 50, 350

    seed = np.max(y), (x[15] - x[5])/np.log(y[15]/y[5])
    f    = fitf.fit(fitf.expo, x, y, seed, fit_range=Zrange_LT, sigma=u_y)

    plt.plot(x, f.fn(x), fitOpt)
    print("Energy at z=0 = {:.1f} +- {:.1f}".format( f.values[0], f.errors[0]))
    print("Lifetime      = {:.1f} +- {:.1f}".format(-f.values[1], f.errors[1]))
    print("Chi2          = {:.2f}          ".format(f.chi2))

    labels("Drift time ($\mu$s)", "S2 energy (pes)", "Fiducialized energy spectrum vs Z")

    return corrf.LifetimeCorrection(-f.values[1], f.errors[1])


# write reset dst
class ResetDST(tb.IsDescription):
    X        = tb.Float64Col(pos=0)
    Y        = tb.Float64Col(pos=1)
    Z        = tb.Float64Col(pos=2)
    E        = tb.Float64Col(pos=3)
    EventID  = tb.UInt32Col(pos=4)
    PeakNo = tb.UInt32Col(pos=5)
    Iteration = tb.UInt32Col(pos=6)

def rebin_s2si(s2, s2si, rf):
    """given an s2 and a corresponding s2si, rebin them by a factor rf"""
    assert rf >= 1 and rf % 1 == 0
    t, e, sipms = pmapsf.rebin_s2si_peak(s2[0], s2[1], s2si, rf)
    s2d_rebin = [t, e]
    s2sid_rebin = sipms

    return s2d_rebin, s2sid_rebin

def rebin_s2pmt(s2pmt, stride):
    """rebin: s2 times (taking mean), s2 energies, and s2 sipm qs, by stride"""
    # cython rebin_array is returning memoryview so we need to cast as np array
    return   [corefc.rebin_array(s2pmt.t , stride, remainder=True, mean=True),               corefc.rebin_array(s2pmt.E , stride, remainder=True)]


# ##  Read data
# ### PMap parameter selection
init_val = {}

init_val['s1_emin'] = 54 * pes
init_val['s1_emax'] = 2400. * pes
init_val['s1_wmin'] = 7*25 * ns
init_val['s1_wmax'] = 16*25 * ns
init_val['s1_hmin'] = 12. * pes
init_val['s1_hmax'] = 400. * pes
init_val['s1_num'] = 1
init_val['s1_ethr'] = 0.5 * pes

init_val['s2_emin'] = 200000. * pes
init_val['s2_emax'] = 2.e6 * pes
init_val['s2_wmin'] = 4.* mus
init_val['s2_wmax'] = 500. * mus
init_val['s2_hmin'] = 600. * pes
init_val['s2_hmax'] = 60000. * pes
init_val['s2_numMin'] = 1
init_val['s2_numMax'] = 3
init_val['s2_ethr'] = 0. * pes
init_val['s2_nsipmmin'] = 1
init_val['s2_nsipmmax'] = 1792

def refresh_selector(param_val):
    selector = s1s2filt.S12Selector(s1_nmin = param_val['s1_num'], s1_nmax = param_val['s1_num'], s1_emin = param_val['s1_emin'], s1_emax = param_val['s1_emax'], s1_ethr = param_val['s1_ethr'],
                                    s1_wmin = param_val['s1_wmin'], s1_wmax = param_val['s1_wmax'], s1_hmin = param_val['s1_hmin'], s1_hmax = param_val['s1_hmax'],
                                    s2_nmin = param_val['s2_numMin'], s2_nmax = param_val['s2_numMax'], s2_emin = param_val['s2_emin'], s2_emax = param_val['s2_emax'], 
                                    s2_wmin = param_val['s2_wmin'], s2_wmax = param_val['s2_wmax'], s2_hmin = param_val['s2_hmin'], s2_hmax = param_val['s2_hmax'], 
                                    s2_nsipmmin = param_val['s2_nsipmmin'], s2_nsipmmax = param_val['s2_nsipmmax'], s2_ethr = param_val['s2_ethr'])

    return selector


# ### Compute lifetime correction
ZCorr = corrf.LifetimeCorrection(1093.77, 23.99)


# ### Load correction maps
pmt_xy_map = []
sipm_xy_map = []
modePMT = "Sum"
#modePMT = "Individual"
nPMT = 11
pmt_sens = np.arange(0,nPMT,1)

pmtcorr = "/home/jmbenlloch/reset_data/mapas/PMT_Map_corr_keV.h5"
pmtcorr_i = "/home/jmbenlloch/reset_data/mapas/PMT_Map_corr_keV_{}_4734.h5"
sipmcorr = "/home/jmbenlloch/reset_data/mapas/SiPM_Map_corr_z{}.0_keV.h5"

zstep = 100
z_scan = np.arange(0,600,zstep)

for zmap in z_scan:
    sipmcorr_file = sipmcorr.format(zmap)
    sipm_xy_map.append(dstf.load_xy_corrections(sipmcorr_file, group="ResetMap", node="SiPM"))

if modePMT == "Sum":
    pmt_xy_map.append(dstf.load_xy_corrections(pmtcorr, group="ResetMap", node="PMT"))
elif modePMT == "Individual":
    for nsens in pmt_sens:
        pmtcorr_file = pmtcorr_i.format(nsens)
        pmt_xy_map.append(dstf.load_xy_corrections(pmtcorr_file, group="ResetMap", node="PMT"))
#        pmt_xy_map.append(dstf.load_xy_corrections(pmtcorr_file+".h5"))
else:
    print("Invalid mode")

# # CUDA
import invisible_cities.reco.reset_functions_cuda as rst_cuda
import pycuda.driver as cuda
#import pycuda.autoinit
import numpy
from pycuda.compiler import SourceModule

from pycuda.tools import make_default_context
cuda.init()
ctx = make_default_context()

def reset_algorithm_cuda(DataSiPM, slice_, sens_q, point_dist, sipm_thr, sizeX, sizeY, rMax, sipm_dist, pmt_xy_map, sipm_xy_map):
    print("reset running")

    voxels_h = rst_cuda.CreateVoxels(DataSiPM, np.array(slice_[0]), sens_q, point_dist, sipm_thr, sizeX, sizeY, rMax)
    anode_response_h = rst_cuda.CreateSiPMresponse(DataSiPM, np.array(slice_[0]), sens_q, sipm_dist, sipm_thr, voxels_h)
    voxDX_h, voxDY_h = rst_cuda.computeDiff(DataSiPM, voxels_h, anode_response_h)
    z_index = int(z//zstep )
    sipm_prob_h, pmt_prob_h = rst_cuda.computeProb(pmt_xy_map, sipm_xy_map[z_index], voxDX_h, voxDY_h, voxels_h['x'], voxels_h['y'])
    selSens_h = rst_cuda.createSel(voxDX_h, voxDY_h, anode_response_h, sipm_dist)
    cath_response_h = np.array([e*ZCorr(z).value], dtype='f4')

    # Copy data to device
#    voxels_d         = cuda.to_device(voxels_h)
#    anode_response_d = cuda.to_device(anode_response_h)
#    cath_response_d  = cuda.to_device(cath_response_h)
#    pmt_prob_d      = cuda.to_device(pmt_prob_h)
#    sipm_prob_d      = cuda.to_device(sipm_prob_h)
#    selSens_d        = cuda.to_device(selSens_h)

    #allocate memory for result
#    voxels_out_d = cuda.mem_alloc(voxels_h.nbytes)

    #compile cuda code
#    kernel_code = open('mlem_step.cu').read()
#    mod = SourceModule(kernel_code)
#    func = mod.get_function("mlem_step")

    #run algorithm
#    iterations = 100
#    for i in range(iterations):
#        if i > 0:
#            voxels_d, voxels_out_d = voxels_out_d, voxels_d
#        func(voxels_d, voxels_out_d, anode_response_d, cath_response_d, pmt_prob_d, sipm_prob_d, selSens_d, np.int32(voxels_h.shape[0]), np.int32(anode_response_h.shape[0]), np.int32(1), block=(1, 1, 1), grid=(voxels_h.shape[0], 1))

#    func(voxels_d, voxels_out_d, anode_response_d, cath_response_d, pmt_prob_d, sipm_prob_d, selSens_d, np.int32(voxels_h.shape[0]), np.int32(anode_response_h.shape[0]), np.int32(1), block=(1, 1, 1), grid=(voxels_h.shape[0], 1))

    #copy back the result
    #voxels_out_h = cuda.from_device(voxels_out_d, voxels_h.shape, voxels_h.dtype)
    #TODO fix
    voxels_out_h = voxels_h

#    voxels_d.free()
#    anode_response_d.free()
#    cath_response_d.free()
#    pmt_prob_d.free()
#    sipm_prob_d.free()
#    selSens_d.free()
#    voxels_out_d.free()

    return voxels_out_h




slice_width = 2.

sipm_dist = 20.
point_dist = 20.
sizeX = 2.
sizeY = 2.
#sizeX = 10.
#sizeY = 10.
sipm_thr = 5.
iterations = 100
fCathode = True
fAnode = True
e_thres = 0.
verbose = False
verboseMLEM = False
rMax = 198
#zstep = 100
#modePMT = "Individual"
#nPMT = 11

#outputfolder = "/Users/ansimez/Documents/InvCit/hdf5/{}/reset_dst/".format(run_number)
outputfolder = '/home/jmbenlloch/reset_data/4495/reset_dst'
if not os.path.exists(outputfolder):
    os.mkdir(outputfolder)

in_limit = 189
filestep = 1
s2pmt_file = {}

#for fnumber, file in enumerate(inputfiles[in_limit:in_limit+filestep]):
for fnumber, file in enumerate(['/home/jmbenlloch/reset_data/4495/pmaps/pmaps.gdcsnext.189_4495.root.h5']):
    outputfile = outputfolder + "reset_dst_"+modePMT+"_{}_select.h5".format(fnumber+in_limit)
    print(outputfile)
    h5out = tb.open_file(outputfile, 'w')
    dst_table = tbiof.make_table(h5out, 
                            group = "Reset_DST",
                            name = "Reset voxels",
                            fformat = ResetDST,
                            description = 'Reset voxels',
                            compression = "ZLIB4")
    row = dst_table.row 
    print("Running on ", time.asctime())
    try: 
#        s1_file, s2_file, s2si_file, _, s2pmt_file = pmapio.load_pmaps_with_ipmt(file)
        s1_file, s2_file, s2si_file, = pmapio.load_pmaps(file)
    except: 
        print("File {} could not be opened".format(file))
    else:
        common_events = set(s1_file.keys()) & set(s2_file.keys()) & set(s2si_file.keys())
        s1_all = dict({k:v for k,v in s1_file.items() if k in common_events})
        s2_all = dict({k:v for k,v in s2_file.items() if k in common_events})
        s2si_all =  dict({k:v for k,v in s2si_file.items() if k in common_events})
        s2pmt_all = dict({k:v for k,v in s2pmt_file.items() if k in common_events})
        
        select = refresh_selector(init_val)

#        for evt in range(1,2):
        for evt in s1_all:
            if(evt != 21215):
                continue
            else:
                print("file: {}".format(fnumber))

            if not s1s2filt.s1s2_filter(select, s1_all[evt], s2_all[evt], s2si_all[evt]):
                continue
            s1_cut = select.select_valid_peaks(s1_all[evt], select.s1_ethr, select.s1e, select.s1w, select.s1h)
            s2_cut = select.select_valid_peaks(s2_all[evt], select.s2_ethr, select.s2e, select.s2w, select.s2h)
            s2si_cut = select.select_s2si(s2si_all[evt], select.nsi)
    
            print("evt: ", evt)
            
            s1 = s1_all[evt].peak_waveform(0)
            t0 = s1.t[np.argmax(s1.E)]

            s2_cut = [peakno for peakno, v in s2_cut.items() if v == True]
            s2si_cut = [peakno for peakno, v in s2si_cut.items() if v == True]
            
            valid_peaks = set(s2_cut) & set(s2si_cut)

            for no in valid_peaks:
                s2 = s2si_all[evt].s2d[no]
                s2si = s2si_all[evt].s2sid[no]
                if modePMT == "Individual":
                    s2pmt = []
                    for nsens in pmt_sens:
                        s2pmt.append(rebin_s2pmt(s2pmt_all[evt].pmt_waveform(no, nsens), slice_width))
                s2, s2si = rebin_s2si(s2, s2si, slice_width)
                    
                print("peak ", no)
                
                for tbin, e in enumerate(s2[1]):
                    print("tbin ", tbin)
                    slice_ = pmapsfc.sipm_ids_and_charges_in_slice(s2si, tbin)
                    if len(slice_[0]) <= 0:
                        continue
                    z = (np.average(s2[0][tbin], weights=s2[1][tbin]) - t0)/1000.
                    if(z>550):
                        continue
                    #Create voxels
                    sens_q = np.array(slice_[1]*ZCorr(z).value)
                    if len(sens_q[sens_q>sipm_thr]) <= 0:
                        continue     
                    
                    #break
                    imageIter = reset_algorithm_cuda(DataSiPM, slice_, sens_q, point_dist, sipm_thr, sizeX, sizeY, rMax, sipm_dist, pmt_xy_map, sipm_xy_map)
                        
    #Save Event
                    for iv in range(len(imageIter[0])):
                        if imageIter[2][iv] <= 0.1:
                            continue
                        row['X'] = imageIter[0][iv]
                        row['Y'] = imageIter[1][iv]
                        row['Z'] = z
                        row['E'] = imageIter[2][iv]
                        row['PeakNo'] = no
                        row['EventID'] = evt
                        row['Iteration'] = 100
                        row.append() 
                    
    h5out.close()                    

h5out.close()



ctx.detach()
