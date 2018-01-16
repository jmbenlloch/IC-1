import numpy as np
import invisible_cities.reco.corrections as corrf
import reset_functions as rstf

# Initialize RESET
run_number = 4495
nsipms     = 1792
npmts      = 1
dist       = 20.
sipm_dist  = 20.
pmt_dist   = 10000 # all pmts are included
sipm_thr   = 5.
x_size     = 2.
y_size     = 2.
rmax       = 198
pmt_param  = "/home/jmbenlloch/reset_data/mapas/PMT_Map_corr_keV.h5"
sipm_param = "/home/jmbenlloch/reset_data/mapas/SiPM_Map_corr_z0.0_keV.h5"

reset = rstf.RESET(run_number, nsipms, npmts, dist, sipm_dist, pmt_dist, sipm_thr, x_size, y_size, rmax, sipm_param, pmt_param)

###############
## Read data ##
###############
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
s2e = s2_energy * ZCorr(z).value

iterations = 100
reset.run(sensor_ids, charges, s2e, iterations)

reset._destroy_context()
