"""__________________________________________________________________________________________________________________________________

Main Model Script for PyDUBEVEG

Python version of the DUne, BEach, and VEGetation (DUBEVEG) model from Keijsers et al. (2016) and Galiforni Silva et al. (2018, 2019)

Translated from Matlab by IRB Reeves

Last update: 31 October 2022

__________________________________________________________________________________________________________________________________"""

import numpy as np
import scipy.io
import math
import dill
import matplotlib.pyplot as plt
import time

import routines_dubeveg as routine


class DUBEVEG:
    def __init__(
            self,

            # GENERAL
            name="default",
            simnum=1,  # Reference number of the simulation. Used for personal reference.
            MHT=0,  # [m] Sea-level reference
            MHTrise=0.000,  # [m/yr] Sea-level rise rate
            qpotseries=2,  # Number reference to calculate how many iterations represent one year. 4 is standard year of 100 iterations, corresponds to qpot (#1 = 25 it, #2 = 50 it, #3 = 75 it, #4 = 100 it, #5 = 125 it) (*25 = qpot)
            writeyear=1,  # Write results to disc every n years
            simulation_time_y=15,  # [yr] Length of the simulation time
            cellsize=1,  # [m] Interpreted cell size
            slabheight=0.1,  # Ratio of cell dimension 0.1 (0.077 - 0.13 (Nield and Baas, 2007))
            inputloc="Input/",  # Input file directory
            outputloc="Output/",  # Output file directory
            topo_filename="topo_west.mat",
            eqtopo_filename="eqtopo_west.mat",
            waterlevel_filename="wl_max_texel.mat",
            veg_spec1_filename="spec1.mat",
            veg_spec2_filename="spec2.mat",
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
            m26=0.012,  # Parameter for dissipation strength ~[0.01 - 0.02]
            wave_energy=1,  # Parameter for initial wave strength ~[1 - 10]
            depth_limit=0.01,  # Depth limit up to where dissipation is calculated. For depths smaller than "depth_limit", the program sets the value as "pwaveminf"
            pcurr=0,  # Probability of erosion due to any hydrodynamic forcing rather that waves
            m28f=0.8,  # Resistance of vegetation: 1 = full, 0 = none
            pwavemaxf=1,  # Maximum erosive strenght of waves (if >1: overrules vegetation)
            pwaveminf=0.1,  # In area with waves always potential for action (this can never be 0, otherwise the beachupdate is shut down)
            shelterf=1,  # Exposure of sheltered cells: 0 = no shelter, 1 = full shelter

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
            no_timeseries=0,
    ):
        """Python version of the DUne, BEach, and VEGetation model"""

        self._name = name
        self._simnum = simnum
        self._MHT = MHT
        self._MHTrise = MHTrise
        self._qpotseries = qpotseries
        self._writeyear = writeyear
        self._simulation_time_y = simulation_time_y
        self._cellsize = cellsize
        self._slabheight = slabheight
        self._slabheight_m = cellsize * slabheight  # [m] Slab height
        self._inputloc = inputloc
        self._outputloc = outputloc
        self._save_data = save_data
        self._groundwater_depth = groundwater_depth
        self._direction = np.random.choice(np.tile([direction1, direction2, direction3, direction4, direction5], (1, 2000))[0, :], 10000, replace=False)
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
        self._no_timeseries = no_timeseries
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

        # __________________________________________________________________________________________________________________________________
        # SET INITIAL CONDITIONS

        # TIME
        self._vegetationupdate = round(self._qpotseries * 25)
        self._iterations_per_cycle = round(self._qpotseries * 25)  # Number of iterations that is regarded as 1 year (was 50) [iterations/year]
        self._beachreset = round(self._qpotseries * 1)

        # TOPOGRAPHY
        self._topo_initial = scipy.io.loadmat(inputloc + topo_filename)["topo_final"]  # [m] 2D-matrix with initial topography
        self._eqtopo_initial = scipy.io.loadmat(inputloc + eqtopo_filename)["topo_final"]  # [m] 2D-matrix or 3D-matrix with equilibrium profile. For 3D-matrix, the third matrix relates to time

        topo0 = self._topo_initial / self._slabheight_m  # [slabs] Transform from m into number of slabs
        self._topo = np.round(topo0)  # [slabs] Initialise the topography map

        if self._eqtopo_initial.ndim == 3:
            self._eqtopo = np.round(np.squeeze(self._eqtopo_initial[:, :, 0]) / self._slabheight_m)  # [slabs] Transform from m into number of slabs
        else:
            self._eqtopo = np.round(self._eqtopo_initial / self._slabheight_m)  # [slabs] Transform from m into number of slabs
        eqtopo_i = self._eqtopo

        self._longshore, self._crossshore = topo0.shape * self._cellsize  # [m] Cross-shore/alongshore size of topography

        self._gw = np.round(eqtopo_i * self._groundwater_depth)  # GW lies under beach with less steep angle
        self._gw[self._gw >= self._topo] = self._topo[self._gw >= self._topo]

        self._beachslopeslabs = (eqtopo_i[0, -1] - eqtopo_i[0, 0]) / self._crossshore  # [slabs/m] Slope of equilibrium beach
        self._offbeachslabs = eqtopo_i[0, 0] - self._beachslopeslabs * self._cellsize  # [slabs] Offset for calculating moving equilibirum beach

        # HYDRODYNAMIC
        self._wl_timeseries = scipy.io.loadmat(inputloc + waterlevel_filename)["wl_max_texel"]  # [m] Only when "no_timeseries" = 0. Waterlevel time-series. Length and frequency in relation to "simulation_time_y" and "qpotseries"
        # self._wl_probcum                  = scipy.io.loadmat('prob_wl.mat')  # Only when "no_timeseries" = 1. Waterlevel probabilities values. To be used only when no_timeseries = 1

        if self._no_timeseries == 0:
            self._waterlevels = np.concatenate([self._wl_timeseries[:, 0], self._wl_timeseries[:, 0], self._wl_timeseries[:, 0], self._wl_timeseries[:, 0], self._wl_timeseries[:, 0], self._wl_timeseries[:, 0]])  # <------------- TEMP!!!!!!!!!!!!!!
        else:
            raise ValueError('No water level time-series has been loaded. Functionality to automatically build a WL time-series from cumulative probabilities has not yet been incorporated into this model version.')

        # VEGETATION
        self._spec1 = scipy.io.loadmat(inputloc + veg_spec1_filename)["vegf"]  # [0-1] 2D-matrix of vegetation effectiveness for spec1
        self._spec2 = scipy.io.loadmat(inputloc + veg_spec2_filename)["vegf"]  # [0-1] 2D-matrix of vegetation effectiveness for spec2

        self._veg = self._spec1 + self._spec2  # Determine the initial cumulative vegetation effectiveness
        self._veg[self._veg > self._maxvegeff] = self._maxvegeff  # Cumulative vegetation effectiveness cannot be negative or larger than one
        self._veg[self._veg < 0] = 0

        self._growth_reduction_timeseries = np.linspace(0, self._VGR / 100, self._simulation_time_y)

        # MODEL PARAMETERS
        self._timewaterlev = np.linspace(self._beachreset / self._iterations_per_cycle, len(self._waterlevels) * self._beachreset / self._iterations_per_cycle, num=len(self._waterlevels))
        self._waterlevels = ((self._timewaterlev * self._MHTrise) + (self._waterlevels + self._MHT)) / self._slabheight_m  # [slabs]
        self._slabheight = round(self._slabheight_m * 100) / 100
        self._balance = self._topo * 0  # Initialise the sedimentation balance map [slabs]
        self._stability = self._topo * 0  # Initialise the stability map [slabs]
        self._sp1_peak_at0 = self._sp1_peak  # Store initial peak growth of sp. 1
        self._sp2_peak_at0 = self._sp2_peak  # Store initial peak growth of sp. 2
        self._inundated = np.zeros([self._longshore, self._crossshore])  # Initial area of wave/current action
        self._inundatedcum = np.zeros([self._longshore, self._crossshore])  # Initial area of sea action
        self._pbeachupdatecum = np.zeros([self._longshore, self._crossshore])  # Matrix for cumulative effect of hydrodynamics
        self._beachcount = 0
        self._vegcount = 0

        # __________________________________________________________________________________________________________________________________
        # MODEL OUPUT CONFIGURATION
        """Select matrices to be calculated during the simulation. CAUTION WITH THE SIZE OF YOUR OUTPUT, YOU MAY GET AN OUT-OF-MEMORY ERROR."""

        # MANDATORY
        self._iterations = self._iterations_per_cycle * self._simulation_time_y  # Number of iterations
        self._timeits = np.linspace(1, self._iterations, self._iterations)  # Time vector for budget calculations
        self._seainput_slabs = np.zeros([self._longshore, self._crossshore])  # Inititalise vector for sea-transported slab
        self._topo_TS = np.empty([self._longshore, self._crossshore, self._simulation_time_y + 1])  # Array for saving each topo map for each simulation year
        self._topo_TS[:, :, 0] = self._topo
        self._spec1_TS = np.empty([self._longshore, self._crossshore, self._simulation_time_y + 1])  # Array for saving each spec1 map for each simulation year
        self._spec1_TS[:, :, 0] = self._spec1
        self._spec2_TS = np.empty([self._longshore, self._crossshore, self._simulation_time_y + 1])  # Array for saving each spec2 map for each simulation year
        self._spec2_TS[:, :, 0] = self._spec2

        # OPTIONAL
        # self._flux_contour = np.zeros([len(self._timeits), self._n_contour + 1])  # Inititalise vector for sea-transported slabs
        # self._seainput_tot = np.empty([self._longshore, self._crossshore, len(self._timeits)])  # 3-D seainput matrices
        # self._diss_tot = np.empty([self._longshore, self._crossshore, len(self._timeits)])  # 3-D dissipation matrices
        # self._cumdiss_tot = np.empty([self._longshore, self._crossshore, len(self._timeits)])  # 3-D cummulative dissipation matrices
        # pself._wave_tot = np.empty([self._longshore, self._crossshore, len(self._timeits)])  # 3-D pwave matrices
        # self._pbeachupdate_tot = np.empty([self._longshore, self._crossshore, len(self._timeits)])  # 3-D pbeachupdate matrices
        # bself._alancea_tot = np.empty([self._longshore, self._crossshore, len(self._timeits)])  # 3-D balancea matrices
        # self._balanceb_tot = np.empty([self._longshore, self._crossshore, len(self._timeits)])  # 3-D balanceb matrices
        # self._stabilitya_tot = np.empty([self._longshore, self._crossshore, len(self._timeits)])  # 3-D stabilitya matrices
        # self._stabilityb_tot = np.empty([self._longshore, self._crossshore, len(self._timeits)])  # 3-D stabilityb matrices
        # self._erosmap_tot = np.empty([self._longshore, self._crossshore, len(self._timeits)])  # 3-D erosmap matrices
        # self._deposmap_tot = np.empty([self._longshore, self._crossshore, len(self._timeits)])  # 3-D deposmap matrices
        # self._shadowmap_tot = np.zeros([self._longshore, self._crossshore])  # 3-D shadowmap matrice
        self._erosmap_sum = np.zeros([self._longshore, self._crossshore])  # Sum of all erosmaps
        self._deposmap_sum = np.zeros([self._longshore, self._crossshore])  # Sum of all deposmaps
        self._seainput_sum = np.zeros([self._longshore, self._crossshore])  # Sum of all seainput maps
        self._balancea_sum = np.zeros([self._longshore, self._crossshore])  # Sum of all balancea maps
        self._balanceb_sum = np.zeros([self._longshore, self._crossshore])  # Sum of all balanceb maps
        self._stabilitya_sum = np.zeros([self._longshore, self._crossshore])  # Sum of all stabilitya maps
        self._stabilityb_sum = np.zeros([self._longshore, self._crossshore])  # Sum of all stabilityb maps

        # Inititalise vectors for transport activity
        self._windtransp_slabs = np.zeros([len(self._timeits)])
        self._landward_transport = np.zeros([len(self._timeits)])
        self._avalanches = np.zeros([len(self._timeits)])

    # __________________________________________________________________________________________________________________________________
    # MAIN ITERATION LOOP

    def update(self, it):
        """Update PyDUBEVEG by a single time step"""

        year = math.ceil(it / self._iterations_per_cycle)

        if self._eqtopo_initial.ndim == 3:
            self._eqtopo = np.squeeze(self._eqtopo_initial[:, :, it]) / self._slabheight_m

        # --------------------------------------
        # SAND TRANSPORT

        before = self._topo
        self._gw = self._eqtopo * self._groundwater_depth
        sandmap = self._topo > self._MHT  # Boolean array, Returns True (1) for sandy cells

        shadowmap = routine.shadowzones2(self._topo, self._slabheight, self._shadowangle, self._longshore, self._crossshore, direction=self._direction[it])  # Returns map of True (1) for in shadow, False (2) not in shadow

        erosmap = routine.erosprobs2(self._veg, shadowmap, sandmap, self._topo, self._gw, self._p_ero_bare)  # Returns map of erosion probabilities
        deposmap = routine.depprobs(self._veg, shadowmap, sandmap, self._p_dep_bare, self._p_dep_sand)  # Returns map of deposition probabilities

        if self._direction[it] == 1 or self._direction[it] == 3:  # East or west wind direction
            contour = np.linspace(0, round(self._crossshore) - 1, self._n_contour + 1)  # Contours to account for transport
            changemap, slabtransp, sum_contour = routine.shiftslabs3_open3(erosmap, deposmap, self._jumplength, contour, self._longshore, self._crossshore, direction=self._direction[it])  # Returns map of height changes
        else:  # North or south wind direction
            contour = np.linspace(0, round(self._longshore) - 1, self._n_contour + 1)  # Contours to account for transport  #  IRBR 21Oct22: This may produce slightly different results than Matlab version - need to verify
            changemap, slabtransp, sum_contour = routine.shiftslabs3_open3(erosmap, deposmap, self._jumplength, contour, self._longshore, self._crossshore, direction=self._direction[it])  # Returns map of height changes

        self._topo = self._topo + changemap  # Changes applied to the topography
        self._topo, aval = routine.enforceslopes2(self._topo, self._veg, self._slabheight, self._repose_bare, self._repose_veg, self._repose_threshold)  # Enforce angles of repose: avalanching
        self._balance = self._balance + (self._topo - before)  # Update the sedimentation balance map
        balance_init = self._balance + (self._topo - before)
        self._stability = self._stability + abs(self._topo - before)
        stability_init = self._stability + abs(self._topo - before)

        # --------------------------------------
        # BEACH UPDATE

        if it % self._beachreset == 0:  # Update the inundated part of the beach

            inundatedold = self._inundated  # Make a copy for later use
            before1 = self._topo  # Copy of topo before it is changed

            # self._topo, self._inundated, pbeachupdate, diss, cumdiss, pwave = routine.marine_processes3_diss3e(
            #     self._waterlevels[self._beachcount],
            #     self._MHTrise,
            #     self._slabheight_m,
            #     self._cellsize,
            #     self._topo,
            #     self._eqtopo,
            #     self._veg,
            #     self._m26,
            #     self._wave_energy,
            #     self._m28f,
            #     self._pwavemaxf,
            #     self._pwaveminf,
            #     self._depth_limit,
            #     self._shelterf,
            #     self._pcurr,
            # )

            self._topo, self._inundated, pbeachupdate, diss, cumdiss, pwave = routine.marine_processes(
                self._waterlevels[self._beachcount],
                self._MHTrise,
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
            )

            seainput = self._topo - before1  # Sand added to the beach by the sea
            # seainput_slabs[it] = np.nansum(seainput)  # IRBR 24OCt22: Bug here. Why is this even needed?

            self._inundated = np.logical_or(self._inundated, inundatedold)  # Combine updated area from individual loops
            self._pbeachupdatecum = pbeachupdate + self._pbeachupdatecum  # Cumulative beachupdate probabilities
            self._topo = routine.enforceslopes2(self._topo, self._veg, self._slabheight, self._repose_bare, self._repose_veg, self._repose_threshold)[0]  # Enforce angles of repose again
            self._balance = self._balance + (self._topo - before1)
            self._stability = self._stability + abs(self._topo - before1)

            self._beachcount += 1  # Update counter

        # --------------------------------------
        # VEGETATION

        if it % self._iterations_per_cycle == 0 and it > 0:  # Update the vegetation

            veg_multiplier = (1 + self._growth_reduction_timeseries[self._vegcount])  # For the long term reduction.
            self._sp1_peak = self._sp1_peak_at0 * veg_multiplier
            self._sp2_peak = self._sp2_peak_at0 * veg_multiplier
            spec1_old = self._spec1
            spec2_old = self._spec2
            self._spec1 = routine.growthfunction1_sens(self._spec1, self._balance * self._slabheight_m, self._sp1_a, self._sp1_b, self._sp1_c, self._sp1_d, self._sp1_e, self._sp1_peak)
            self._spec2 = routine.growthfunction2_sens(self._spec2, self._balance * self._slabheight_m, self._sp2_a, self._sp2_b, self._sp2_d, self._sp1_e, self._sp2_peak)

            # Lateral Expansion & Veg Establishment
            lateral1 = routine.lateral_expansion(spec1_old, 1, self._lateral_probability * veg_multiplier)  # Species 1
            lateral2 = routine.lateral_expansion(spec2_old, 1, self._lateral_probability * veg_multiplier)  # Species 1
            pioneer1 = routine.establish_new_vegetation(self._topo * self._slabheight_m, self._MHT, self._pioneer_probability * veg_multiplier) * (spec1_old <= 0)
            pioneer2 = routine.establish_new_vegetation(self._topo * self._slabheight_m, self._MHT, self._pioneer_probability * veg_multiplier) * (spec2_old <= 0) * (self._stability == 0)

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

            # Limit to geomorphological range
            spec1_geom = self._spec1
            spec1_geom[self._spec1 < 0] = 0
            spec1_geom[self._spec1 > 1] = 1
            spec2_geom = self._spec2
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

        if 'erosmap_sum' in locals():
            self._erosmap_sum = self._erosmap_sum + erosmap

        if 'deposmap_sum' in locals():
            self._deposmap_sum = self._deposmap_sum + deposmap

        if 'erosmap_tot' in locals():
            self._erosmap_tot[:, :, it] = erosmap

        if 'deposmap_tot' in locals():
            self._deposmap_tot[:, :, it] = deposmap

        if 'shadowmap_tot' in locals():
            self._shadowmap_tot[:, :, it] = shadowmap

        if 'balancea_sum' in locals():
            self._balancea_sum = self._balancea_sum + balance_init

        # if 'balancea_tot' in locals():
        #     balancea_tot[:, :, it] = balance_init

        if 'stabilitya_sum' in locals():
            self._stabilitya_sum = self._stabilitya_sum + stability_init

        # if 'stabilitya_tot' in locals():
        #     stabilitya_tot[:, :, it] = stabilitya

        if 'windtransp_slabs' in locals():
            self._windtransp_slabs[it] = (slabtransp * self._slabheight_m * self._cellsize ** 2) / self._longshore

        if 'avalanches' in locals():
            self._avalanches[it] = aval

        if 'flux_contour' in locals():
            self._flux_contour[it, :] = (sum_contour * self._slabheight_m * self._cellsize ** 2)

        # if 'seainput_total' in locals():
        #     seainput_total[:, :, it] = seainput

        # if 'diss_total' in locals():
        #     diss_total[:, :, it] = diss

        # if 'cumdiss_total' in locals():
        #     cumdiss_total[:, :, it] = cumdiss

        # if 'pwave_total' in locals():
        #     pwave_total[:, :, it] = pwave

        # if 'pbeachupdate_total' in locals():
        #     pbeachupdate_total[:, :, it] = pbeachupdate

        if 'seainput_sum' in locals():
            self._seainput_sum = self._seainput_sum + seainput

        if 'balanceb_sum' in locals():
            balanceb_sum = self._balanceb_sum + self._balance

        if 'balanceb_tot' in locals():
            self._balanceb_tot[:, :, it] = self._balance

        if 'stabilityb_sum' in locals():
            self._stabilityb_sum = self._stabilityb_sum + self._stability

        # if 'stabilityb' in locals():
        #     stabilityb[:, :, it] = self._balance

        # --------------------------------------
        # RESET DOMAINS

        balance_copy = self._balance
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
    def slabheight(self):
        return self._slabheight

    @property
    def veg(self):
        return self._veg

    @property
    def save_data(self):
        return self._save_data

    @property
    def outputloc(self):
        return self._outputloc

    @property
    def simnum(self):
        return self._simnum


# __________________________________________________________________________________________________________________________________
# RUN MODEL

start_time = time.time()  # Record time at start of simulation

# Create an instance of the BMI class
dubeveg = DUBEVEG(
    name="30 yr, SLR 0",
    simulation_time_y=30,
    MHTrise=0.00,
    save_data=False,
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
    dill.dump_module(filename)  # To re-load saved data, use dill.load_session(filename)

# Temp Plot
plt.matshow(dubeveg.topo * dubeveg.slabheight,
            cmap='terrain',
            vmin=0,
            vmax=4.5
            )
plt.title("Elev TMAX, " + dubeveg.name)

plt.matshow(dubeveg.veg,
            cmap='YlGn',
            )
plt.title("Veg TMAX, " + dubeveg.name)
plt.show()
