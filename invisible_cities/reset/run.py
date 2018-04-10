import invisible_cities.reset.utils as rst_utils
import os

from invisible_cities.core.system_of_units import pes, mm, mus, ns
import invisible_cities.reco.corrections    as corrf
import invisible_cities.database.load_db as dbf
import invisible_cities.reset.reset_gpu as rstf

# Initialize RESET
iterations = 100
run_number = 4495
nsipms     = 1792
npmts      = 1
dist       = 20.
sipm_dist  = 20.
pmt_dist   = 200 # max distance included in the param file
sipm_thr   = 5.
#x_size     = 2.
#y_size     = 2.
x_size     = 10.
y_size     = 10.
rmax       = 198
slice_width = 2
pmt_param  = "/home/jmbenlloch/reset_data/mapas/PMT_Map_corr_keV.h5"
sipm_param = "/home/jmbenlloch/reset_data/mapas/SiPM_Map_corr_z0.0_keV.h5"
pmap_file = '4495_21215.h5'
pmap_file = os.path.expandvars('$ICDIR/database/test_data/reset_4495_21215.h5')

pmap_conf = {}
pmap_conf['s1_emin'] = 54 * pes
pmap_conf['s1_emax'] = 2400. * pes
pmap_conf['s1_wmin'] = 7*25 * ns
pmap_conf['s1_wmax'] = 16*25 * ns
pmap_conf['s1_hmin'] = 12. * pes
pmap_conf['s1_hmax'] = 400. * pes
pmap_conf['s1_num'] = 1
pmap_conf['s1_ethr'] = 0.5 * pes

pmap_conf['s2_emin'] = 200000. * pes
pmap_conf['s2_emax'] = 2.e6 * pes
pmap_conf['s2_wmin'] = 4.* mus
pmap_conf['s2_wmax'] = 500. * mus
pmap_conf['s2_hmin'] = 600. * pes
pmap_conf['s2_hmax'] = 60000. * pes
pmap_conf['s2_numMin'] = 1
pmap_conf['s2_numMax'] = 3
pmap_conf['s2_ethr'] = 0. * pes
pmap_conf['s2_nsipmmin'] = 1
pmap_conf['s2_nsipmmax'] = 1792

# Read file and select peaks
evt = 21215
selector = rst_utils.refresh_selector(pmap_conf)
s1, s2 = rst_utils.load_and_select_peaks(pmap_file, evt, selector)

# Lifetime correction
ZCorr = corrf.LifetimeCorrection(1093.77, 23.99)

# Sensors info
data_sipm = dbf.DataSiPM(run_number)

# Prepare data
reset_data   = rst_utils.prepare_data(s1, s2, slice_width, evt, data_sipm, nsipms, sipm_thr, dist, ZCorr)
slices_start = rst_utils.slices_start(reset_data.voxels_data, x_size, y_size)


rst = rstf.RESET(run_number, nsipms, npmts, dist, sipm_dist, pmt_dist, x_size, y_size, rmax, sipm_param, pmt_param)
voxels, slices = rst.run(reset_data.voxels_data, reset_data.slices, reset_data.energies, slices_start, iterations)
rst_utils.write_hdf5(voxels, slices, reset_data.zs)
rst._destroy_context()
