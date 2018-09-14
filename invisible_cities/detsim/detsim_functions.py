"""
detsim_functions.py
Defines key functions used in Detsim.
"""

import numpy as np

from invisible_cities.evm.pmaps              import S1
from invisible_cities.evm.pmaps              import S2
from invisible_cities.evm.pmaps              import PMTResponses
from invisible_cities.evm.pmaps              import SiPMResponses
from invisible_cities.evm.pmaps import PMap
from invisible_cities.reco import peak_functions as pkf

from invisible_cities.core.system_of_units_c import units


def pmt_lcone(ze):
    """
    Approximate PMT light cone function.
    ze: the distance from the EL region to the SiPM sipm_plane
    """

    def pmt_lcone_r(r):
        return np.abs(ze) / (2 * np.pi) / (r**2 + ze**2)**1.5

    return pmt_lcone_r


def sipm_lcone(A, d, ze):
    """
    Approximate SiPM light cone function.
    A:  the area of a single SiPM
    d:  the length of the EL gap
    ze: the distance from the EL region to the SiPM sipm_plane
    """

    def sipm_lcone_r(r):
        return (A/(4*np.pi*d*np.sqrt(r**2 + ze**2)))*(1 - np.sqrt((r**2 + ze**2)/(r**2 + (ze+d)**2)))

    return sipm_lcone_r


def make_s2(pmt_map, sipm_map, s2_threshold_sipm, slice_width,
            ids_pmt, islice_lastpk, islice, zdrift):
    pk_wf_pmt = pmt_map[islice_lastpk:islice,:].transpose()
    ids_sipm, pk_wf_sipm = pkf.select_wfs_above_time_integrated_thr(
            sipm_map[islice_lastpk:islice,:].transpose(),
            s2_threshold_sipm)
    return S2([(t+int(zdrift))*slice_width*units.mus for t in range(islice_lastpk,islice)],
                PMTResponses(ids_pmt,pk_wf_pmt),
                SiPMResponses(ids_sipm,pk_wf_sipm))

def simulate_sensors(hits, data_sipm, slice_width_sipm, light_function_sipm,
                     E_to_Q_sipm, s2_threshold_sipm,
                     data_pmt, slice_width_pmt, light_function_pmt,
                     E_to_Q_pmt, s2_threshold_pmt, peak_space):

    nsipm = len(data_sipm.X)
    npmt  = len(data_pmt.X)

    zmin = np.min([hit.pos[2] for hit in hits])
    zmax = np.max([hit.pos[2] for hit in hits])

    nslices_sipm = int(np.ceil((zmax - zmin)/slice_width_sipm))
    nslices_pmt  = int(np.ceil((zmax - zmin)/slice_width_pmt))
    #print("nslices_sipms: {}, slices_pmts: {}".format(nslices_sipm, nslices_pmt))

    sipm_map      = np.zeros([nslices_sipm,nsipm])
    sipm_energies = np.zeros(nslices_sipm)
    pmt_map       = np.zeros([nslices_pmt,npmt])
    pmt_energies  = np.zeros(nslices_pmt)

    for hit in hits:
        islice_sipm = int((hit.pos[2] - zmin)/slice_width_sipm)
        r_sipm = np.array([np.sqrt((xi - hit.pos[0])**2 + (yi - hit.pos[1])**2) for xi,yi in zip(data_sipm.X,data_sipm.Y)])
        probs_sipm = light_function_sipm(r_sipm)
        sipm_map[islice_sipm,:] += probs_sipm * hit.E * E_to_Q_sipm
        sipm_energies[islice_sipm] += hit.E

        islice_pmt = int((hit.pos[2] - zmin)/slice_width_pmt)
        r_pmt = np.array([np.sqrt((xi - hit.pos[0])**2 + (yi - hit.pos[1])**2) for xi,yi in zip(data_pmt.X,data_pmt.Y)])
        probs_pmt = light_function_pmt(r_pmt)
        pmt_map[islice_pmt,:] += probs_pmt * hit.E * E_to_Q_pmt
        pmt_energies[islice_pmt] += hit.E


    pmap = get_detsim_pmaps(sipm_map, s2_threshold_sipm, pmt_map,
                            s2_threshold_pmt, slice_width_pmt,
                            peak_space, zmin) #Think if zmin is the best option
    return pmap


def get_detsim_pmaps(sipm_map, s2_threshold_sipm,
                     pmt_map, s2_threshold_pmt, slice_width,
                     peak_space, zdrift):

    ids_pmt = [ipmt for ipmt in range(0,12)]

    # S1: for now, a default S1
    s1s = [ S1([slice_width*units.mus],
            PMTResponses(ids_pmt, 10*np.ones([12,1])),
            SiPMResponses.build_empty_instance())]

    # S2
    s2s = []
    islice_lastpk = 0
    for islice in range(len(pmt_map)):

        signals_sum = np.sum(pmt_map[islice,:])
        if(signals_sum > s2_threshold_pmt):

            if((islice - islice_lastpk)*slice_width >= peak_space):

                # create a new S2 peak beginning where the last one left off
                s2s.append(make_s2(pmt_map, sipm_map, s2_threshold_sipm, slice_width,
                                   ids_pmt, islice_lastpk, islice, zdrift))

                islice_lastpk  = islice

    # create the final S2 peak
    s2s.append(make_s2(pmt_map, sipm_map, s2_threshold_sipm, slice_width,
                       ids_pmt, islice_lastpk, len(pmt_map), zdrift))

    return PMap(s1s,s2s)
