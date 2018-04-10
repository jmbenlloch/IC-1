"""
code: ic_cotainers.py
description: namedtuples describing miscellaenous containers to
pass info around.

credits: see ic_authors_and_legal.rst in /doc

last revised: JJGC, 10-July-2017
"""

import sys

from collections import namedtuple


this_module = sys.modules[__name__]

def _add_namedtuple_in_this_module(name, attribute_names):
    new_nametuple = namedtuple(name, attribute_names)
    setattr(this_module, name, new_nametuple)

for name, attrs in (
        ('DataVectors'    , 'pmt sipm mc events trg_type trg_channels'),
        ('PmapVectors'    , 'pmaps events timestamps mc'),
        ('RawVectors'     , 'event pmtrwf sipmrwf pmt_active sipm_active'),
        ('CalibParams'    , 'coeff_c, coeff_blr, adc_to_pes_pmt adc_to_pes_sipm'),
        ('DeconvParams'   , 'n_baseline thr_trigger'),
        ('CalibVectors'   , 'channel_id coeff_blr coeff_c adc_to_pes adc_to_pes_sipm pmt_active'),
        ('S12Params'      , 'time stride length rebin_stride'),
        ('PmapParams'     , 's1_params s2_params s1p_params s1_PMT_params s1p_PMT_params'),
        ('ThresholdParams', 'thr_s1 thr_s2 thr_MAU thr_sipm thr_SIPM'),
        ('CSum'           , 'csum csum_mau'),
        ('CCWf'           , 'ccwf, ccwf_mau'),
        ('S12Sum'         , 's1_ene, s1_indx, s2_ene, s2_indx'),
        ('CalibratedPMT'  , 'CPMT CPMT_mau'),
        ('S1PMaps'        , 'S1 S1_PMT S1p S1p_PMT'),
        ('PMaps'          , 'S1 S2 S2Si'),
        ('Peak'           , 't E'),
        ('FitFunction'    , 'fn values errors chi2 pvalue cov'),
        ('TriggerParams'  , 'trigger_channels min_number_channels charge height width'),
        ('PeakData'       , 'charge height width'),
        ('Measurement'    , 'value uncertainty'),
        ('SensorsParams'  , 'xmin ymin step nbins params'),
        ('GPUScan'        , 'data active addr'),
        ('ResetProbs'     , 'nprobs probs sensor_ids voxel_start sensor_start fwd_nums'),
        ('ResetSnsProbs'  , 'probs voxel_ids nsensors sensor_start sensor_start_ids'),
        ('VoxelsLimits'   , 'nslices xmin xmax ymin ymax charge'),
        ('ResetRatios'    , 'sns_per_voxel voxel_per_sns'),
        ('ResetTest'      , 'voxels_ini anode cathode sipm_probs sipm_sns_probs pmt_probs pmt_sns_probs voxels'),
        ('ResetVoxels'    , 'nslices nvoxels voxels slice_ids slice_start address'),
        ('ResetData'      , 'voxels_data slices energies zs'),
        ('ResetSlices'    , 'nslices nsensors ncharges start sensors charges')):
    _add_namedtuple_in_this_module(name, attrs)

# Leave nothing but the namedtuple types in the namespace of this module
del name, namedtuple, sys, this_module, _add_namedtuple_in_this_module
