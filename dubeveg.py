"""__________________________________________________________________________________________________________________________________

Main Model Script for PyDUBEVEG

Python version of the DUne, BEach, and VEGetation (DUBEVEG) model from Keijsers et al. (2016) and Galiforni Silva et al. (2018, 2019)

Translated from Matlab by IRB Reeves

Last update: 9 January 2023

__________________________________________________________________________________________________________________________________"""

import numpy as np
import scipy.io
import math
import dill
import matplotlib.pyplot as plt
import time
import imageio
import os
import copy
from mpl_toolkits.mplot3d import Axes3D

import routines_dubeveg as routine

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


class DUBEVEG:
    def __init__(
            self,

            # GENERAL
            name="default",
            simnum=1,  # Reference number of the simulation. Used for personal reference.
            MHT=0,  # [m] Mean high tide
            RSLR=0.000,  # [m/yr] Relative sea-level rise rate
            qpotseries=2,  # Number reference to calculate how many iterations represent one year. 4 is standard year of 100 iterations, corresponds to qpot (#1 = 25 it, #2 = 50 it, #3 = 75 it, #4 = 100 it, #5 = 125 it) (*25 = qpot)
            writeyear=1,  # Write results to disc every n years
            simulation_time_yr=15,  # [yr] Length of the simulation time
            cellsize=1,  # [m] Interpreted cell size
            slabheight=0.1,  # Ratio of cell dimension 0.1 (0.077 - 0.13 (Nield and Baas, 2007))
            inputloc="Input/",  # Input file directory
            outputloc="Output/",  # Output file directory
            topo_filename="topo_small_dune.npy",
            eqtopo_filename="eqtopo_small_dune.npy",
            waterlevel_filename="wl_max_texel.mat",
            veg_spec1_filename="spec1_small_dune.npy",
            veg_spec2_filename="spec2_small_dune.npy",
            seeded_random_numbers=True,
            save_data=False,

            # AEOLIAN
            groundwater_depth=0.8,  # Proportion of the equilibrium profile used to set groundwater profile.
            direction1=1,  # Direction 1 (from) of the slab movement. 1 = west, 2 = north, 3 = east and 4 = south
            direction2=1,  # Direction 2 (from) of the slab movement. 1 = west, 2 = north, 3 = east and 4 = south
            direction3=1,  # Direction 3 (from) of the slab movement. 1 = west, 2 = north, 3 = east and 4 = south
            direction4=1,  # Direction 4 (from) of the slab movement. 1 = west, 2 = north, 3 = east and 4 = south
            direction5=1,  # Direction 5 (from) of the slab movement. 1 = west, 2 = north, 3 = east and 4 = south
            p_dep_sand=0.1,  # [0-1] Probability of deposition of sandy cells
            p_dep_bare=0.1,  # [0-1] Probability of deposition of base cells
            p_ero_bare=0.5,  # [0-1] Probability of erosion of bare/sandy cells
            shadowangle=15,  # [deg]
            repose_bare=20,  # [deg] - orig:30
            repose_veg=30,  # [deg] - orig:35
            repose_threshold=0.3,  # Vegetation threshold for applying repose_veg
            jumplength=1,  # [slabs] Hop length for slabs
            clim=0.5,  # Vegetation cover that limits erosion
            n_contour=10,  # Number of contours to be used to calculate fluxes. Multiples of 10

            # HYDRODYNAMIC
            m26=0.012,  # Parameter for dissipation strength ~[0.01 - 0.02], AKA Fdiss
            wave_energy=1.0,  # Parameter for initial wave strength ~[1 - 10]
            depth_limit=0.01,  # Depth limit up to where dissipation is calculated. For depths smaller than "depth_limit", the program sets the value as "pwaveminf"
            pcurr=0,  # Probability of erosion due to any hydrodynamic forcing rather that waves
            m28f=0.8,  # Resistance of vegetation: 1 = full, 0 = none
            pwavemaxf=1.0,  # Maximum erosive strenght of waves (if >1: overrules vegetation)
            pwaveminf=0.1,  # In area with waves always potential for action (this can never be 0, otherwise the beachupdate is shut down)
            shelterf=1.0,  # Exposure of sheltered cells: 0 = no shelter, 1 = full shelter

            # SHOREFACE & SHORELINE CHANGE
            shoreface_flux_rate=5000,  # [m3/m/yr] Shoreface flux rate coefficient
            shoreface_equilibrium_slope=0.02,  # Equilibrium slope of the shoreface
            shoreface_depth=10,  # [m] Depth to shoreface toe (i.e. depth of ‘closure’)
            shoreface_length_init=500,  # [m] Initial length of shoreface
            shoreface_toe_init=0,  # [m] [m] Start location of shoreface toe

            # VEGETATION
            sp1_a=-1.4,  # Vertice a, spec1. vegetation growth based on Nield and Baas (2008)
            sp1_b=0.2,  # Vertice b, spec1. vegetation growth based on Nield and Baas (2008)
            sp1_c=0.6,  # Vertice c, spec1. vegetation growth based on Nield and Baas (2008)
            sp1_d=2.0,  # Vertice d, spec1. vegetation growth based on Nield and Baas (2008)
            sp1_e=2.2,  # Vertice e, spec1. vegetation growth based on Nield and Baas (2008)

            sp2_a=-1.4,  # Vertice a, spec2. vegetation growth based on Nield and Baas (2008)
            sp2_b=-0.65,  # Vertice b, spec2. vegetation growth based on Nield and Baas (2008)
            sp2_c=0.0,  # Vertice c, spec2. vegetation growth based on Nield and Baas (2008)
            sp2_d=0.2,  # Vertice d, spec2. vegetation growth based on Nield and Baas (2008)
            sp2_e=2.8,  # Vertice e, spec2. vegetation growth based on Nield and Baas (2008)

            sp1_peak=0.2,  # Growth peak, spec1
            sp2_peak=0.05,  # Growth peak, spec2

            VGR=0,  # [%] Growth reduction by end of period
            lateral_probability=0.2,  # Probability of lateral expansion of existing vegetation
            pioneer_probability=0.05,  # Probability of occurrence of new pioneering vegetation

            maxvegeff=1.0,  # [0-1] Value of maximum vegetation effectiveness allowed

            Spec1_elev_min=0.25,  # [m MHT] Minimum elevation for species 1 (1 m MHW for A. brevigulata from Young et al., 2011)
            Spec2_elev_min=0.25,  # [m MHT] Minimum elevation for species 2

            # STORM OVERWASH AND DUNE EROSION
            storm_list_filename="VCRStormList.npy",
            threshold_in=0.25,  # [%] Threshold percentage of overtopped dune cells exceeded by Rlow needed to be in inundation overwash regime
            Rin_in=5,  # [m^3/hr] Flow infiltration and drag parameter, inundation overwash regime
            Rin_ru=40,  # [m^3/hr] Flow infiltration and drag parameter, run-up overwash regime
            Cx=10,  # Constant for representing flow momentum for sediment transport in inundation overwash regime
            nn=0.5,  # Flow routing constant
            MaxUpSlope=2,  # Maximum slope water can flow uphill
            Qs_min=1.0,  # [m^3/hr] Minimum discharge out of cell needed to transport sediment
            K_ru=7.5e-05,  # Sediment transport coefficient for run-up overwash regime
            K_in=7.5e-06,  # Sediment transport coefficient for inundation overwash regime
            mm=2.0,  # Inundation overwash constant
            Cbb_in=0.85,  # [%] Coefficient for exponential decay of sediment load entering back-barrier bay, inundation regime
            Cbb_ru=0.7,  # [%] Coefficient for exponential decay of sediment load entering back-barrier bay, run-up regime
            Qs_bb_min=1,  # [m^3/hr] Minimum discharge out of subaqueous back-barrier cell needed to transport sediment
            substep_in=2,  # Number of substeps to run for each hour in inundation overwash regime (e.g., 3 substeps means discharge/elevation updated every 20 minutes)
            substep_ru=2,  # Number of substeps to run for each hour in run-up overwash regime (e.g., 3 substeps means discharge/elevation updated every 20 minutes)

    ):
        """Python version of the DUne, BEach, and VEGetation model"""

        self._name = name
        self._simnum = simnum
        self._MHT = MHT * slabheight  # [slabs]
        self._RSLR = RSLR
        self._qpotseries = qpotseries
        self._writeyear = writeyear
        self._simulation_time_yr = simulation_time_yr
        self._cellsize = cellsize
        self._slabheight = slabheight
        self._slabheight_m = cellsize * slabheight  # [m] Slab height
        self._inputloc = inputloc
        self._outputloc = outputloc
        self._save_data = save_data
        self._groundwater_depth = groundwater_depth
        self._p_dep_sand = p_dep_sand
        self._p_dep_bare = p_dep_bare
        self._p_ero_bare = p_ero_bare
        self._shadowangle = shadowangle
        self._repose_bare = repose_bare
        self._repose_veg = repose_veg
        self._repose_threshold = repose_threshold
        self._jumplength = jumplength
        self._clim = clim
        self._n_contour = n_contour
        self._m26 = m26
        self._wave_energy = wave_energy
        self._depth_limit = depth_limit
        self._pcurr = pcurr
        self._m28f = m28f
        self._pwavemaxf = pwavemaxf
        self._pwaveminf = pwaveminf
        self._shelterf = shelterf
        self._k_sf = shoreface_flux_rate
        self._s_sf_eq = shoreface_equilibrium_slope
        self._DShoreface = shoreface_depth
        self._LShoreface = shoreface_length_init
        self._sp1_a = sp1_a
        self._sp1_b = sp1_b
        self._sp1_c = sp1_c
        self._sp1_d = sp1_d
        self._sp1_e = sp1_e
        self._sp2_a = sp2_a
        self._sp2_b = sp2_b
        self._sp2_c = sp2_c
        self._sp2_d = sp2_d
        self._sp2_e = sp2_e
        self._sp1_peak = sp1_peak
        self._sp2_peak = sp2_peak
        self._VGR = VGR
        self._lateral_probability = lateral_probability
        self._pioneer_probability = pioneer_probability
        self._maxvegeff = maxvegeff
        self._Spec1_elev_min = Spec1_elev_min
        self._Spec2_elev_min = Spec2_elev_min
        self._threshold_in = threshold_in
        self._Rin_in = Rin_in
        self._Rin_ru = Rin_ru
        self._Cx = Cx
        self._nn = nn
        self._MaxUpSlope = MaxUpSlope
        self._Qs_min = Qs_min
        self._K_ru = K_ru
        self._K_in = K_in
        self._mm = mm
        self._Cbb_in = Cbb_in
        self._Cbb_ru = Cbb_ru
        self._Qs_bb_min = Qs_bb_min
        self._substep_in = substep_in
        self._substep_ru = substep_ru

        # __________________________________________________________________________________________________________________________________
        # SET INITIAL CONDITIONS

        # SEEDED RANDOM NUMBER GENERATOR
        if seeded_random_numbers:
            self._RNG = np.random.default_rng(seed=13)  # Seeded random numbers for reproducibility (e.g., model development/testing)
        else:
            self._RNG = np.random.default_rng()  # Non-seeded random numbers (e.g., model simulations)

        # TIME
        self._vegetationupdate = round(self._qpotseries * 25)
        self._iterations_per_cycle = round(self._qpotseries * 25)  # Number of iterations that is regarded as 1 year (was 50) [iterations/year]
        self._beachreset = round(self._qpotseries * 1)
        self._iterations = self._iterations_per_cycle * self._simulation_time_yr  # Number of iterations

        # TOPOGRAPHY
        # self._topo_initial = scipy.io.loadmat(inputloc + topo_filename)["topo_final"]  # [m] 2D-matrix with initial topography
        # self._eqtopo_initial = scipy.io.loadmat(inputloc + eqtopo_filename)["topo_final"]  # [m] 2D-matrix or 3D-matrix with equilibrium profile. For 3D-matrix, the third matrix relates to time
        self._topo_initial = np.load(inputloc + topo_filename)  # [m] 2D-matrix with initial topography
        self._eqtopo_initial = np.load(inputloc + eqtopo_filename)  # [m] 2D-matrix or 3D-matrix with equilibrium profile. For 3D-matrix, the third matrix relates to time

        topo0 = self._topo_initial / self._slabheight_m  # [slabs] Transform from m into number of slabs
        self._topo = np.round(topo0)  # [slabs] Initialise the topography map

        self._longshore, self._crossshore = topo0.shape * self._cellsize  # [m] Cross-shore/alongshore size of topography

        if self._eqtopo_initial.ndim == 3:
            self._eqtopo = np.round(np.squeeze(self._eqtopo_initial[:, :, 0]) / self._slabheight_m)  # [slabs] Transform from m into number of slabs
        else:
            self._eqtopo = np.round(self._eqtopo_initial / self._slabheight_m)  # [slabs] Transform from m into number of slabs

        eqtopo_i = copy.deepcopy(self._eqtopo)

        self._gw = np.round(eqtopo_i * self._groundwater_depth)  # GW lies under beach with less steep angle
        self._gw[self._gw >= self._topo] = self._topo[self._gw >= self._topo]

        self._beachslopeslabs = (eqtopo_i[0, -1] - eqtopo_i[0, 0]) / self._crossshore  # [slabs/m] Slope of equilibrium beach
        self._offbeachslabs = eqtopo_i[0, 0] - self._beachslopeslabs * self._cellsize  # [slabs] Offset for calculating moving equilibirum beach

        self._x_t = shoreface_toe_init * np.ones([self._longshore])  # Start locations of shoreface toe
        self._x_s = (self._x_t + self._LShoreface) * np.ones([self._longshore])  # [m] Start locations of shoreline

        # HYDRODYNAMIC
        self._wl_timeseries = scipy.io.loadmat(inputloc + waterlevel_filename)["wl_max_texel"]  # [m] Waterlevel time-series. Length and frequency in relation to "simulation_time_yr" and "qpotseries"

        # TEMP extend WL time series
        self._waterlevels = np.concatenate([self._wl_timeseries[:, 0], self._wl_timeseries[:, 0], self._wl_timeseries[:, 0], self._wl_timeseries[:, 0], self._wl_timeseries[:, 0], self._wl_timeseries[:, 0]])

        # VEGETATION
        # self._spec1 = scipy.io.loadmat(inputloc + veg_spec1_filename)["vegf"]  # [0-1] 2D-matrix of vegetation effectiveness for spec1
        # self._spec2 = scipy.io.loadmat(inputloc + veg_spec2_filename)["vegf"]  # [0-1] 2D-matrix of vegetation effectiveness for spec2
        self._spec1 = np.load(inputloc + veg_spec1_filename)  # [0-1] 2D-matrix of vegetation effectiveness for spec1
        self._spec2 = np.load(inputloc + veg_spec2_filename)  # [0-1] 2D-matrix of vegetation effectiveness for spec2

        self._veg = self._spec1 + self._spec2  # Determine the initial cumulative vegetation effectiveness
        self._veg[self._veg > self._maxvegeff] = self._maxvegeff  # Cumulative vegetation effectiveness cannot be negative or larger than one
        self._veg[self._veg < 0] = 0

        self._growth_reduction_timeseries = np.linspace(0, self._VGR / 100, self._simulation_time_yr)

        # STORMS
        self._StormList = np.load(inputloc + storm_list_filename)
        # self._pstorm = [0.393939393939394, 0.212121212121212, 0.181818181818182, 0.181818181818182, 0.212121212121212, 0.242424242424242, 0.212121212121212, 0.333333333333333, 0.363636363636364, 0.272727272727273, 0.303030303030303, 0.303030303030303, 0.181818181818182, 0.151515151515152,
        #                 0.212121212121212, 0.151515151515152, 0.212121212121212, 0.0606060606060606, 0.0909090909090909, 0.0303030303030303, 0.0303030303030303, 0.121212121212121, 0.0606060606060606, 0, 0.0909090909090909, 0.0303030303030303, 0, 0, 0, 0.0303030303030303, 0.0303030303030303,
        #                 0.0909090909090909, 0.0909090909090909, 0.303030303030303, 0.151515151515152, 0.121212121212121, 0.303030303030303, 0.121212121212121, 0.151515151515152, 0.272727272727273, 0.242424242424242, 0.303030303030303, 0.181818181818182, 0.242424242424242, 0.0909090909090909,
        #                 0.181818181818182, 0.333333333333333, 0.151515151515152, 0.212121212121212, 0.272727272727273]  # Empirical probability of storm occurance for each 1/50th (~weekly) iteration of the year, from 1980-2013 VCR storm record
        self._pstorm = [0.787878787878788, 0.393939393939394, 0.454545454545455, 0.727272727272727, 0.575757575757576, 0.484848484848485, 0.363636363636364, 0.363636363636364, 0.151515151515152, 0.0606060606060606, 0.181818181818182,
                        0.0606060606060606, 0.0606060606060606, 0, 0.0303030303030303, 0.121212121212121, 0.393939393939394, 0.272727272727273, 0.424242424242424, 0.424242424242424, 0.545454545454545, 0.424242424242424,
                        0.272727272727273, 0.484848484848485, 0.484848484848485]  # Empirical probability of storm occurance for each 1/25th (~biweekly) iteration of the year, from 1980-2013 VCR storm record
        # self._pstorm = [0.232558139534884, 0.0697674418604651, 0.116279069767442, 0.186046511627907, 0.0930232558139535, 0.0465116279069767, 0.139534883720930, 0.0697674418604651, 0.0232558139534884, 0, 0, 0, 0.0232558139534884, 0,
        #                 0.0232558139534884, 0.0465116279069767, 0.255813953488372, 0.255813953488372, 0.116279069767442, 0.139534883720930, 0.0697674418604651, 0.0465116279069767, 0.0697674418604651, 0.0697674418604651, 0.0930232558139535]  # Empirical probability of storm occurance for each 1/25th (~biweekly) iteration of the year, from 1979-2021 NCB storm record

        # TEMP! Adjust beach towards equilibrium profile before sim begins
        dune_crest = routine.foredune_crest(self._topo, self._eqtopo_initial, self._veg)
        self._topo, self._inundated, pbeachupdate, diss, cumdiss, pwave, crestline_change = routine.marine_processes_Rhigh(
            2.5 / self._slabheight_m,  # Convert to slabs
            self._slabheight_m,
            self._cellsize,
            self._topo,
            self._eqtopo,
            self._veg,
            self._m26,
            self._wave_energy,
            self._m28f,
            self._pwavemaxf,
            self._pwaveminf,
            self._depth_limit,
            self._shelterf,
            dune_crest,
        )

        # MODEL PARAMETERS
        self._direction = self._RNG.choice(np.tile([direction1, direction2, direction3, direction4, direction5], (1, 2000))[0, :], 10000, replace=False)
        self._timewaterlev = np.linspace(self._beachreset / self._iterations_per_cycle, len(self._waterlevels) * self._beachreset / self._iterations_per_cycle, num=len(self._waterlevels))
        self._waterlevels = (self._timewaterlev * self._RSLR + self._waterlevels) / self._slabheight_m  # [slabs]
        self._slabheight = round(self._slabheight_m * 100) / 100
        self._balance = self._topo * 0  # Initialise the sedimentation balance map [slabs]
        self._stability = self._topo * 0  # Initialise the stability map [slabs]
        self._x_s_TS = [self._x_s]  # Initialize storage array for shoreline position
        self._x_t_TS = [self._x_t]  # Initialize storage array for shoreface toe position
        self._sp1_peak_at0 = copy.deepcopy(self._sp1_peak)  # Store initial peak growth of sp. 1
        self._sp2_peak_at0 = copy.deepcopy(self._sp2_peak)  # Store initial peak growth of sp. 2
        self._inundated = np.zeros([self._longshore, self._crossshore])  # Initial area of wave/current action
        self._inundatedcum = np.zeros([self._longshore, self._crossshore])  # Initial area of sea action
        self._pbeachupdatecum = np.zeros([self._longshore, self._crossshore])  # Matrix for cumulative effect of hydrodynamics
        self._beachcount = 0
        self._vegcount = 0
        self._shoreline_change_aggregate = np.zeros([self._longshore])
        self._Qat = np.zeros([self._longshore])  # Need to convert from slabs to m
        self._OWflux = np.zeros([self._longshore])  # [m^3]
        self._DuneLoss = np.zeros([self._longshore])  # Need to convert from slabs to m
        self._StormRecord = np.empty([5])  # Record of each storm that occurs in model: Year, iteration, Rhigh, Rlow, duration
        self._topo_change_leftover = np.zeros(self._topo.shape)
        self._TWLexcursion = np.zeros([self._longshore, int(self._iterations / self._beachreset)])

        # __________________________________________________________________________________________________________________________________
        # MODEL OUPUT CONFIGURATION

        self._timeits = np.linspace(1, self._iterations, self._iterations)  # Time vector for budget calculations
        self._topo_TS = np.empty([self._longshore, self._crossshore, self._simulation_time_yr + 1])  # Array for saving each topo map for each simulation year
        self._topo_TS[:, :, 0] = self._topo
        self._spec1_TS = np.empty([self._longshore, self._crossshore, self._simulation_time_yr + 1])  # Array for saving each spec1 map for each simulation year
        self._spec1_TS[:, :, 0] = self._spec1
        self._spec2_TS = np.empty([self._longshore, self._crossshore, self._simulation_time_yr + 1])  # Array for saving each spec2 map for each simulation year
        self._spec2_TS[:, :, 0] = self._spec2
        self._veg_TS = np.empty([self._longshore, self._crossshore, self._simulation_time_yr + 1])  # Array for saving each veg map for each simulation year
        self._veg_TS[:, :, 0] = self._veg
        self._erosmap_sum = np.zeros([self._longshore, self._crossshore])  # Sum of all erosmaps
        self._deposmap_sum = np.zeros([self._longshore, self._crossshore])  # Sum of all deposmaps
        self._seainput_sum = np.zeros([self._longshore, self._crossshore])  # Sum of all seainput maps
        self._balancea_sum = np.zeros([self._longshore, self._crossshore])  # Sum of all balancea maps
        self._balanceb_sum = np.zeros([self._longshore, self._crossshore])  # Sum of all balanceb maps
        self._stabilitya_sum = np.zeros([self._longshore, self._crossshore])  # Sum of all stabilitya maps
        self._stabilityb_sum = np.zeros([self._longshore, self._crossshore])  # Sum of all stabilityb maps
        self._windtransp_slabs = np.zeros([len(self._timeits)])
        self._avalanches = np.zeros([len(self._timeits)])

    # __________________________________________________________________________________________________________________________________
    # MAIN ITERATION LOOP

    def update(self, it):
        """Update PyDUBEVEG by a single time step"""

        year = math.ceil(it / self._iterations_per_cycle)

        # Update sea level
        self._MHT += self._RSLR / self._iterations_per_cycle / self._slabheight_m  # [slabs]

        # --------------------------------------
        # SAND TRANSPORT

        before = copy.deepcopy(self._topo)
        self._gw = self._eqtopo * self._groundwater_depth
        sandmap = self._topo > self._MHT  # Boolean array, Returns True (1) for sandy cells

        shadowmap = routine.shadowzones2(self._topo, self._slabheight, self._shadowangle, self._longshore, self._crossshore, direction=self._direction[it])  # Returns map of True (1) for in shadow, False (2) not in shadow

        erosmap = routine.erosprobs2(self._veg, shadowmap, sandmap, self._topo, self._gw, self._p_ero_bare)  # Returns map of erosion probabilities
        deposmap = routine.depprobs(self._veg, shadowmap, sandmap, self._p_dep_bare, self._p_dep_sand)  # Returns map of deposition probabilities

        if self._direction[it] == 1 or self._direction[it] == 3:  # East or west wind direction
            contour = np.linspace(0, round(self._crossshore) - 1, self._n_contour + 1)  # Contours to account for transport
            changemap, slabtransp, sum_contour = routine.shiftslabs3_open3(erosmap, deposmap, self._jumplength, contour, self._longshore, self._crossshore, self._direction[it], self._RNG)  # Returns map of height changes
        else:  # North or south wind direction
            contour = np.linspace(0, round(self._longshore) - 1, self._n_contour + 1)  # Contours to account for transport  #  IRBR 21Oct22: This may produce slightly different results than Matlab version - need to verify
            changemap, slabtransp, sum_contour = routine.shiftslabs3_open3(erosmap, deposmap, self._jumplength, contour, self._longshore, self._crossshore, self._direction[it], self._RNG)  # Returns map of height changes

        self._topo = self._topo + changemap  # Changes applied to the topography
        self._topo, aval = routine.enforceslopes2(self._topo, self._veg, self._slabheight, self._repose_bare, self._repose_veg, self._repose_threshold, self._RNG)  # Enforce angles of repose: avalanching
        self._balance = self._balance + (self._topo - before)  # Update the sedimentation balance map
        balance_init = self._balance + (self._topo - before)
        self._stability = self._stability + abs(self._topo - before)
        stability_init = self._stability + abs(self._topo - before)


        # --------------------------------------
        # STORMS - UPDATE BEACH, DUNE, AND INTERIOR

        if it % self._beachreset == 0:  # Update the inundated part of the beach
            veg_elev_limit = np.argmax(min(self._Spec1_elev_min, self._Spec2_elev_min) / self._slabheight_m + self._MHT < self._topo, axis=1)
            dune_crest = routine.foredune_crest(self._topo, self._eqtopo_initial, self._veg)
            slopes = routine.beach_slopes(self._eqtopo, self._MHT, dune_crest, self._slabheight_m)

            # # TEMP: DEBUGGING DUNE CREST LOCATION
            # if it % 50 == 0:
            #     tempfig = plt.figure(figsize=(14, 4.5))
            #     tempfig.suptitle(dubeveg.name, fontsize=13)
            #     ax_1 = tempfig.add_subplot(111)
            #     cax_1 = ax_1.matshow(dubeveg.topo * dubeveg.slabheight, cmap='terrain', vmin=-1.1, vmax=4.0)
            #     ax_1.plot(dune_crest, np.arange(len(dune_crest)))

            iteration_year = np.floor(it % self._iterations_per_cycle / 2).astype(int)  # Iteration of the year (e.g., if there's 50 iterations per year, this represents the week of the year)

            # Generate storm stats stochastically
            storm, Rhigh, Rlow, dur = routine.stochastic_storm(self._pstorm, iteration_year, self._StormList, slopes, self._RNG)  # [m MSL]
            Rhigh += self._MHT * self._slabheight_m  # Convert storm water elevation datum from MSL to datum of topo grid by adding present MSL
            Rlow += self._MHT * self._slabheight_m  # Convert storm water elevation datum from MSL to datum of topo grid by adding present MSL

            if storm:
                inundatedold = copy.deepcopy(self._inundated)  # Make a copy for later use
                before1 = copy.deepcopy(self._topo)  # Copy of topo before it is changed

                #  Update Beach
                self._topo, self._inundated, pbeachupdate, diss, cumdiss, pwave, crestline_change = routine.marine_processes_Rhigh(
                    Rhigh / self._slabheight_m,  # Convert to slabs
                    self._slabheight_m,
                    self._cellsize,
                    self._topo,
                    self._eqtopo,
                    self._veg,
                    self._m26,
                    self._wave_energy,
                    self._m28f,
                    self._pwavemaxf,
                    self._pwaveminf,
                    self._depth_limit,
                    self._shelterf,
                    dune_crest,
                )

                # Overwash
                self._StormRecord = np.vstack((self._StormRecord, [year, iteration_year, Rhigh, Rlow, dur]))
                self._topo, topo_change_overwash, self._topo_change_leftover, self._OWflux = routine.overwash_processes(
                    self._topo,
                    self._topo_change_leftover,
                    dune_crest,
                    Rhigh,
                    Rlow,
                    dur,
                    self._slabheight_m,
                    self._threshold_in,
                    self._Rin_in,
                    self._Rin_ru,
                    self._Cx,
                    np.max(self._eqtopo_initial) * self._slabheight_m / 200,  # Representative average slope of interior (made static - representative of 200-m-wide barrier interior)
                    self._nn,
                    self._MaxUpSlope,  # was 0.25
                    self._Qs_min,
                    self._K_ru,
                    self._K_in,
                    self._mm,
                    self._MHT,
                    self._Cbb_in,
                    self._Cbb_ru,
                    self._Qs_bb_min,
                    self._substep_in,
                    self._substep_ru,
                )

                self._topo = routine.enforceslopes2(self._topo, self._veg, self._slabheight, self._repose_bare, self._repose_veg, self._repose_threshold, self._RNG)[0]  # Enforce angles of repose again after overwash; TODO: Investigate whether slope enforcement after overwash is necessary; remove if not

            else:
                self._OWflux = np.zeros([self._longshore])  # [m^3] No overwash if no storm
                inundatedold = copy.deepcopy(self._inundated)
                self._inundated = np.zeros(self._topo.shape)  # Reset
                self._inundated[self._topo <= self._MHT] = 1  # Cells below MHW are always considered inundated TODO: problem if bay cells are marked as inundated here?
                pbeachupdate = np.zeros(self._topo.shape)  # Reset
                before1 = copy.deepcopy(self._topo)
                topo_change_overwash = np.zeros(self._topo.shape)

            seainput = self._topo - before1  # Sand added to the beach by the sea

            self._inundated = np.logical_or(self._inundated, inundatedold)  # Combine updated area from individual loops
            self._pbeachupdatecum = pbeachupdate + self._pbeachupdatecum  # Cumulative beachupdate probabilities
            self._topo = routine.enforceslopes2(self._topo, self._veg, self._slabheight, self._repose_bare, self._repose_veg, self._repose_threshold, self._RNG)[0]  # Enforce angles of repose again
            balance_ts = self._topo - before1
            self._balance = self._balance + balance_ts + topo_change_overwash
            self._stability = self._stability + abs(self._topo - before1)
            self._TWLexcursion[:, int(it / self._beachreset)] = np.argmax(self._inundated < 1, axis=1)  # Will return 0 if whole column is inundated

            self._beachcount += 1  # Update counter

            # --------------------------------------
            # SHORELINE CHANGE & EQUILIBRIUM BEACH PROFILE

            Qbe = np.sum(seainput, axis=1) * self._slabheight_m / self._cellsize  # [m^3/m/ts] Volume of sediment imported from (+) or exported to (-) the upper shoreface by beach change

            self._x_s, self._x_t = routine.shoreline_change2(
                self.topo,
                self._DShoreface,
                self._k_sf,
                self._s_sf_eq,
                self._RSLR,
                self._Qat,
                Qbe,
                self._OWflux,
                self._DuneLoss,
                self._x_s,
                self._x_t,
                self._MHT,
                self._cellsize,
                self._slabheight_m,
            )

            # LTA14 Row-by-Row
            shoreline_change = (self._x_s - self._x_s_TS[-1]) * self._cellsize  # [cellsize] Shoreline change from last time step

            self._shoreline_change_aggregate += shoreline_change

            shoreline_change[self._shoreline_change_aggregate >= 1] = np.floor(self._shoreline_change_aggregate[self._shoreline_change_aggregate >= 1]).astype(int)
            self._shoreline_change_aggregate[self._shoreline_change_aggregate >= 1] -= shoreline_change[self._shoreline_change_aggregate >= 1]

            shoreline_change[self._shoreline_change_aggregate <= -1] = np.ceil(self._shoreline_change_aggregate[self._shoreline_change_aggregate <= -1]).astype(int)
            self._shoreline_change_aggregate[self._shoreline_change_aggregate <= -1] -= shoreline_change[self._shoreline_change_aggregate <= -1]

            shoreline_change[np.logical_and(-1 < shoreline_change, shoreline_change < 1)] = 0

            self._x_s_TS = np.vstack((self._x_s_TS, self._x_s))  # Store
            self._x_t_TS = np.vstack((self._x_t_TS, self._x_t))  # Store

            # Adjust equilibrium beach profile upward (downward) acording to sea-level rise (fall), and landward (seaward) and according to net loss (gain) of sediment at the upper shoreface
            self._eqtopo += self._RSLR * self._beachreset / self._iterations_per_cycle / self._slabheight_m  # [slabs] Raise vertically by amount of SLR over this substep
            for ls in range(self._longshore):
                sc_ls = int(shoreline_change[ls])
                self._eqtopo[ls, :] = np.roll(self._eqtopo[ls, :], sc_ls)  # Shift laterally
                if sc_ls >= 0:
                    shoreline_ls = routine.ocean_shoreline(self._topo, self._MHT)[ls]  # Ocean shoreline location
                    shoreface = np.floor(np.arange(-shoreline_ls, 0, 1) * self._s_sf_eq / self._slabheight_m)
                    self._eqtopo[ls, :shoreline_ls] = shoreface
                else:
                    self._eqtopo[ls, sc_ls:] = self._eqtopo[ls, sc_ls * 2: sc_ls]
        else:
            seainput = np.zeros([self._longshore, self._crossshore])

        # --------------------------------------
        # VEGETATION

        if it % self._iterations_per_cycle == 0 and it > 0:  # Update the vegetation

            veg_multiplier = (1 + self._growth_reduction_timeseries[self._vegcount])  # For the long term reduction.
            self._sp1_peak = self._sp1_peak_at0 * veg_multiplier
            self._sp2_peak = self._sp2_peak_at0 * veg_multiplier
            spec1_old = copy.deepcopy(self._spec1)
            spec2_old = copy.deepcopy(self._spec2)
            self._spec1 = routine.growthfunction1_sens(self._spec1, self._balance * self._slabheight_m, self._sp1_a, self._sp1_b, self._sp1_c, self._sp1_d, self._sp1_e, self._sp1_peak)
            self._spec2 = routine.growthfunction2_sens(self._spec2, self._balance * self._slabheight_m, self._sp2_a, self._sp2_b, self._sp2_d, self._sp1_e, self._sp2_peak)

            # Lateral Expansion & Veg Establishment
            lateral1 = routine.lateral_expansion(spec1_old, 1, self._lateral_probability * veg_multiplier, self._RNG)  # Species 1
            lateral2 = routine.lateral_expansion(spec2_old, 1, self._lateral_probability * veg_multiplier, self._RNG)  # Species 1
            pioneer1 = routine.establish_new_vegetation(self._topo * self._slabheight_m, self._MHT, self._pioneer_probability * veg_multiplier, self._RNG) * (spec1_old <= 0)
            pioneer2 = routine.establish_new_vegetation(self._topo * self._slabheight_m, self._MHT, self._pioneer_probability * veg_multiplier, self._RNG) * (spec2_old <= 0) * (self._stability == 0)

            lateral1[self._topo <= self._MHT] = False
            lateral2[self._topo <= self._MHT] = False
            pioneer1[self._topo <= self._MHT] = False
            pioneer2[self._topo <= self._MHT] = False

            spec1_diff = self._spec1 - spec1_old  # Determine changes in vegetation cover
            spec2_diff = self._spec2 - spec2_old  # Determine changes in vegetation cover
            spec1_growth = spec1_diff * (spec1_diff > 0)  # Split cover changes into into gain and loss
            spec1_loss = spec1_diff * (spec1_diff < 0)  # Split cover changes into into gain and loss
            spec2_growth = spec2_diff * (spec2_diff > 0)
            spec2_loss = spec2_diff * (spec2_diff < 0)

            spec1_change_allowed = np.minimum(1 - self._veg, spec1_growth) * np.logical_or(lateral1, pioneer1)  # Only allow growth in adjacent or pioneer cells
            spec2_change_allowed = np.minimum(1 - self._veg, spec2_growth) * np.logical_or(lateral2, pioneer2)  # Only allow growth in adjacent or pioneer cells
            self._spec1 = spec1_old + spec1_change_allowed + spec1_loss  # Re-assemble gain and loss and add to original vegetation cover
            self._spec2 = spec2_old + spec2_change_allowed + spec2_loss  # Re-assemble gain and loss and add to original vegetation cover

            self._pbeachupdatecum[self._pbeachupdatecum < 0] = 0
            self._pbeachupdatecum[self._pbeachupdatecum > 1] = 1

            self._spec1 = self._spec1 * (1 - self._pbeachupdatecum)  # Remove species where beach is reset
            self._spec2 = self._spec2 * (1 - self._pbeachupdatecum)  # Remove species where beach is reset

            Spec1_elev_min_mht = self._Spec1_elev_min / self._slabheight_m + self._MHT  # [m MHT]
            Spec2_elev_min_mht = self._Spec1_elev_min / self._slabheight_m + self._MHT  # [m MHT]
            self._spec1[self._topo <= Spec1_elev_min_mht] = 0  # Remove species where below elevation minimum
            self._spec2[self._topo <= Spec2_elev_min_mht] = 0  # Remove species where below elevation minimum

            # Limit to geomorphological range
            spec1_geom = copy.deepcopy(self._spec1)
            spec1_geom[self._spec1 < 0] = 0
            spec1_geom[self._spec1 > 1] = 1
            spec2_geom = copy.deepcopy(self._spec2)
            spec2_geom[self._spec2 < 0] = 0
            spec2_geom[self._spec2 > 1] = 1
            self._veg = spec1_geom + spec2_geom  # Update vegmap
            self._veg[self._veg > self._maxvegeff] = self._maxvegeff  # Limit to effective range
            self._veg[self._veg < 0] = 0

            self._vegcount = self._vegcount + 1  # Update counter

        # --------------------------------------
        # RECORD VARIABLES ANNUALLY

        if it % (self._writeyear * self._iterations_per_cycle) == 0:
            moment = int(it / self._iterations_per_cycle) + 1
            self._topo_TS[:, :, moment] = self._topo
            self._spec1_TS[:, :, moment] = self._spec1
            self._spec2_TS[:, :, moment] = self._spec2
            self._veg_TS[:, :, moment] = self._veg

        self._erosmap_sum = self._erosmap_sum + erosmap
        self._deposmap_sum = self._deposmap_sum + deposmap
        self._balancea_sum = self._balancea_sum + balance_init
        self._stabilitya_sum = self._stabilitya_sum + stability_init
        self._stabilityb_sum = self._stabilityb_sum + self._stability
        self._windtransp_slabs[it] = (slabtransp * self._slabheight_m * self._cellsize ** 2) / self._longshore
        self._avalanches[it] = aval
        self._seainput_sum = self._seainput_sum + seainput

        # --------------------------------------
        # RESET DOMAINS

        self._balance[:] = 0  # Reset the balance map
        self._inundated[:] = 0  # Reset part of beach with wave/current action
        self._pbeachupdatecum[:] = 0
        self._stability[:] = 0  # Reset the balance map

    @property
    def name(self):
        return self._name

    @property
    def iterations(self):
        return self._iterations

    @property
    def iterations_per_cycle(self):
        return self._iterations_per_cycle

    @property
    def topo(self):
        return self._topo

    @property
    def topo_TS(self):
        return self._topo_TS

    @property
    def eqtopo(self):
        return self._eqtopo

    @property
    def slabheight(self):
        return self._slabheight

    @property
    def veg(self):
        return self._veg

    @property
    def veg_TS(self):
        return self._veg_TS

    @property
    def spec1_TS(self):
        return self._spec1_TS

    @property
    def spec2_TS(self):
        return self._spec2_TS

    @property
    def save_data(self):
        return self._save_data

    @property
    def outputloc(self):
        return self._outputloc

    @property
    def simnum(self):
        return self._simnum

    @property
    def simulation_time_yr(self):
        return self._simulation_time_yr

    @property
    def StormRecord(self):
        return self._StormRecord


# __________________________________________________________________________________________________________________________________
# RUN MODEL

start_time = time.time()  # Record time at start of simulation

# Create an instance of the BMI class
dubeveg = DUBEVEG(
    name="10 yr, SLR 0 mm/yr",
    simulation_time_yr=10,
    RSLR=0.000,
    seeded_random_numbers=True,
    p_dep_sand=0.5,  # 0.25 = 10 m^3/m/yr, 0.5 = 5 m^m/3/yr
    p_ero_bare=0.5,
    direction2=2,
    direction4=4,
    substep_ru=1,
    # storm_list_filename="NCB_SimStorms.npy",
    topo_filename="Topo_NCB_20190830_100m_23980.npy",
    eqtopo_filename="EQTopo_NCB_20190830_100m_23980.npy",
    veg_spec1_filename="Spec1_NCB_20190830_100m_23980.npy",
    veg_spec2_filename="Spec2_NCB_20190830_100m_23980.npy",
)

print(dubeveg.name)

# Loop through time
for time_step in range(int(dubeveg.iterations)):
    # Print time step to screen
    print("\r", "Time Step: ", time_step / dubeveg.iterations_per_cycle, "years", end="")

    # Run time step
    dubeveg.update(time_step)

# Print elapsed time of simulation
print()
SimDuration = time.time() - start_time
print()
print("Elapsed Time: ", SimDuration, "sec")

# Save Results
if dubeveg.save_data:
    filename = dubeveg.outputloc + "Sim_" + str(dubeveg.simnum)
    dill.dump_module(filename)  # To re-load data: dill.load_session(filename)

# Plot
Fig = plt.figure(figsize=(14, 9.5))
Fig.suptitle(dubeveg.name, fontsize=13)
ax1 = Fig.add_subplot(211)
cax1 = ax1.matshow(dubeveg.topo * dubeveg.slabheight, cmap='terrain', vmin=-1.2, vmax=5.0)
cbar = Fig.colorbar(cax1)
cbar.set_label('Elevation [m]', rotation=270, labelpad=20)
ax2 = Fig.add_subplot(212)
cax2 = ax2.matshow(dubeveg.veg, cmap='YlGn', vmin=0, vmax=1)
cbar = Fig.colorbar(cax2)
cbar.set_label('Vegetation [%]', rotation=270, labelpad=20)
plt.tight_layout()

# plt.figure()
# plt.hist(dubeveg.StormRecord[1:, 1], bins=50, range=(0, dubeveg.iterations_per_cycle))
# plt.title("Model Storm Occurance by Week")

# # Plot 3D Elevation
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection="3d")
# scale_x = 1
# L, C = dubeveg.topo.shape
# scale_y = L / C
# scale_z = 10 / L
# ax.get_proj = lambda: np.dot(
#     Axes3D.get_proj(ax), np.diag([scale_x, scale_y, scale_z, 1])
# )
# X, Y = np.meshgrid(np.arange(C), np.arange(L))
# ax.plot_surface(
#     X,
#     Y,
#     dubeveg.topo * dubeveg.slabheight,
#     cmap="terrain",
#     alpha=1,
#     linewidth=0,
#     shade=True,
#     vmin=0,
#     vmax=4.5,
# )

# Elevation Animation
for t in range(0, dubeveg.simulation_time_yr + 1):
    Fig = plt.figure(figsize=(14, 9.5))
    ax1 = Fig.add_subplot(211)
    cax1 = ax1.matshow(dubeveg.topo_TS[:, :, t] * dubeveg.slabheight, cmap='terrain', vmin=-1.2, vmax=5.0)  # TODO: Plot relative to sea level
    cbar = Fig.colorbar(cax1)
    cbar.set_label('Elevation [m]', rotation=270, labelpad=20)
    timestr = "Year " + str(t)
    plt.text(2, dubeveg.topo.shape[0] - 2, timestr, c='white')
    ax2 = Fig.add_subplot(212)
    cax2 = ax2.matshow(dubeveg.veg_TS[:, :, t], cmap='YlGn', vmin=0, vmax=1)
    cbar = Fig.colorbar(cax2)
    cbar.set_label('Vegetation [%]', rotation=270, labelpad=20)
    timestr = "Year " + str(t)
    plt.text(2, dubeveg.veg.shape[0] - 2, timestr, c='darkblue')
    if not os.path.exists("Output/SimFrames/"):
        os.makedirs("Output/SimFrames/")
    plt.tight_layout()
    name = "Output/SimFrames/dubeveg_elev_" + str(t)
    plt.savefig(name)  # dpi=200
    plt.close()

frames = []
for filenum in range(0, dubeveg.simulation_time_yr + 1):
    filename = "Output/SimFrames/dubeveg_elev_" + str(filenum) + ".png"
    frames.append(imageio.imread(filename))
imageio.mimwrite("Output/SimFrames/dubeveg_elev.gif", frames, fps=3)
print()
print("[ * GIF successfully generated * ]")

plt.show()
