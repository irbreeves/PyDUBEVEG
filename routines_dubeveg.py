"""__________________________________________________________________________________________________________________________________

Main Model Script for PyDUBEVEG

Python version of the DUne, BEach, and VEGetation (DUBEVEG) model from Keijsers et al. (2016) and Galiforni Silva et al. (2018, 2019)

Translated from Matlab by IRB Reeves

Last update: 5 November 2022

__________________________________________________________________________________________________________________________________"""

import numpy as np
import math
import copy


def shadowzones2(topof, sh, lee, longshore, crossshore, direction):
    """Returns a logical map with all shadowzones identified as ones
    Wind from left to right, along the +2 dimension
    - topof: topography map [topo]
    - sh: slab height [slabheight]
    - lee: shadow angle [shadowangle]
    - longshore:
    - crossshore:
    - direction: wind direction (1 east, 2 north, 3, west, 4 south)
    Returns a logical map of shaded cells, now with open boundaries"""

    # steplimit = math.floor(math.tan(lee * math.pi / 180) / sh)  # The maximum step difference allowed given the slabheight and shadowangle
    steplimit = math.tan(lee * math.pi / 180) / sh  # The maximum step difference allowed given the slabheight and shadowangle, not rounded yet

    # range = min(crossshore, max(topof) / steplimit)
    search_range = int(math.floor(np.max(topof) / steplimit))  # Identifies highest elevation and uses that to determine what the largest search distance needs to be

    inshade = np.zeros([longshore, crossshore]).astype(bool)  # Define the zeroed logical map

    for i in range(1, search_range + 1):
        if direction == 1:
            step = topof - np.roll(topof, i, axis=1)  # Shift across columns (2nd dimension; along a row)
            tempinshade = step < -math.floor(steplimit * i)  # Identify cells with too great a stepheight
            tempinshade[:, 0:i] = 0  # Part that is circshifted back into beginning of space is ignored
        elif direction == 2:
            step = topof - np.roll(topof, i, axis=0)  # Shift across columns (2nd dimension; along a row)
            tempinshade = step < -math.floor(steplimit * i)  # Identify cells with too great a stepheight
            tempinshade[0:i, :] = 0  # Part that is circshifted back into beginning of space is ignored
        elif direction == 3:
            step = topof - np.roll(topof, -i, axis=1)  # Shift across columns (2nd dimension; along a row)
            tempinshade = step < -math.floor(steplimit * i)  # Identify cells with too great a stepheight
            tempinshade[:, -1 - i:-1] = 0  # Part that is circshifted back into beginning of space is ignored
        elif direction == 4:
            step = topof - np.roll(topof, -i, axis=0)  # Shift across columns (2nd dimension; along a row)
            tempinshade = step < -math.floor(steplimit * i)  # Identify cells with too great a stepheight
            tempinshade[-1 - i:-1, :] = 0  # Part that is circshifted back into beginning of space is ignored
        inshade = np.bitwise_or(inshade, tempinshade)  # Merge with previous inshade zones

    return inshade


def erosprobs2(vegf, shade, sand, topof, groundw, p_er):
    """% Returns a map with erosion probabilities
    - vegf: map of combined vegetation effectiveness [veg] [0,1]
    - shade: logical map of shadowzones [shadowmap]
    - sand: logical map of sandy cells [sandmap]
    - topof: topography map [topo]
    - groundw: groundwater map [gw]
    - p_er: probability of erosion of bare cell"""

    r = np.logical_not(shade) * sand * (1 - vegf) * (topof > groundw) * p_er

    return r


def depprobs(vegf, shade, sand, dep0, dep1):
    """Returns a map of deposition probabilities that can then be used to implement transport
    - vegf: map of combined vegetation effectiveness [veg] [0,1]
    - shade: logical map of shadowzones [shadowmap]
    - sand: logical map of sanded sites [sandmap]
    - dep0: deposition probability on bare cell [p_dep_bare]
    - dep1: deposition probability on sandy cell [p_dep_sand]"""

    # For bare cells
    temp1 = vegf * (1 - dep0) + dep0  # Deposition probabilities on bare cells
    temp2 = np.logical_not(sand) * np.logical_not(shade) * temp1  # Apply to bare cells outside shadows only

    # For sandy cells
    temp3 = vegf * (1 - dep1) + dep1  # Deposition probabilities on sandy cells
    temp4 = sand * np.logical_not(shade) * temp3  # Apply to sandy cells outside shadows only

    r = temp2 + temp4 + shade  # Combine both types of cells + shadowzones

    return r


def shiftslabs3_open3(erosprobs, deposprobs, hop, contour, longshore, crossshore, direction):
    """Shifts the sand. Movement is from left to right, along +2 dimension, across columns, along rows. Returns a map of height changes [-,+].
    Open boundaries (modification by Alma), no feeding from the sea side.
    - erosprobs: map of erosion probabilities [erosmap]
    - deposprobs: map of deposition probabilities [deposmap]
    - hop: jumplength
    - direction: wind direction (1 east, 2 north, 3, west, 4 south)"""

    pickedup = np.random.rand(longshore, crossshore) < erosprobs  # True where slab is picked up
    # pickedup[:, -1 - hop: -1] = 0  # Do not pick any slab that are on the ladward boundary -- east only?

    totaldeposit = np.zeros([longshore, crossshore])
    inmotion = copy.deepcopy(pickedup)  # Make copy of original erosion map
    numshifted = 0  # [slabs] Number of shifted cells weighted for transport distance
    transportdist = 0  # Transport distance [slab lengths] or [hop length]
    sum_contour = np.zeros([len(contour)])

    while np.sum(inmotion) > 0:  # While still any slabs moving
        transportdist += 1  # Every time in the loop the slaps are transported one length further to the right
        if direction == 1:
            inmotion = np.roll(inmotion, hop, axis=1)  # Shift the moving slabs one hop length to the right
            transp_contour = np.nansum(inmotion[:, contour.astype(np.int64)], axis=0)  # Account ammount of slabs that are in motion in specific contours
            sum_contour = sum_contour + transp_contour  # Sum the slabs to the others accounted before
            depocells = np.random.rand(longshore, crossshore) < deposprobs  # True where slab should be deposited
            # depocells[:, -1 - hop: -1] = 1  # All slabs are deposited if they are transported over the landward edge
            deposited = inmotion * depocells  # True where a slab is available and should be deposited
            deposited[:, 0: hop] = 0  # Remove all slabs that are transported from the landward side to the seaward side (this changes the periodic boundaries into open ones)
        elif direction == 2:
            inmotion = np.roll(inmotion, hop, axis=0)  # Shift the moving slabs one hop length to the right
            transp_contour = np.nansum(inmotion[:, contour.astype(np.int64)], axis=0)  # Account ammount of slabs that are in motion in specific contours
            sum_contour = sum_contour + transp_contour  # Sum the slabs to the others accounted before
            depocells = np.random.rand(longshore, crossshore) < deposprobs  # True where slab should be deposited
            # depocells[0 : hop, :] = 1  # All slabs are deposited if they are transported over the landward edge
            deposited = inmotion * depocells  # True where a slab is available and should be deposited
            deposited[0: hop, :] = 0  # Remove all slabs that are transported from the landward side to the seaward side (this changes the periodic boundaries into open ones)
        elif direction == 3:
            inmotion = np.roll(inmotion, -hop, axis=1)  # Shift the moving slabs one hop length to the right
            transp_contour = np.nansum(inmotion[:, contour.astype(np.int64)], axis=0)  # Account ammount of slabs that are in motion in specific contours
            sum_contour = sum_contour + transp_contour  # Sum the slabs to the others accounted before
            depocells = np.random.rand(longshore, crossshore) < deposprobs  # True where slab should be deposited
            # depocells[:, -1 - hop: -1] = 1  # All slabs are deposited if they are transported over the landward edge
            deposited = inmotion * depocells  # True where a slab is available and should be deposited
            deposited[:, -1 - hop: -1] = 0  # Remove all slabs that are transported from the landward side to the seaward side (this changes the periodic boundaries into open ones)
        elif direction == 4:
            inmotion = np.roll(inmotion, -hop, axis=0)  # Shift the moving slabs one hop length to the right
            transp_contour = np.nansum(inmotion[contour.astype(np.int64), :], axis=1)  # Account ammount of slabs that are in motion in specific contours
            sum_contour = sum_contour + transp_contour  # Sum the slabs to the others accounted before
            depocells = np.random.rand(longshore, crossshore) < deposprobs  # True where slab should be deposited
            # depocells[0 : hop + 1, :] = 1  # All slabs are deposited if they are transported over the landward edge
            deposited = inmotion * depocells  # True where a slab is available and should be deposited
            deposited[-1 - hop: -1, :] = 0  # Remove all slabs that are transported from the landward side to the seaward side (this changes the periodic boundaries into open ones)

        inmotion[deposited] = False  # Left over in transport after this round of deposition
        numshifted = numshifted + np.sum(deposited) * transportdist  # Number of slabs deposited, weighted for transport distance
        totaldeposit = totaldeposit + deposited  # Total slabs deposited so far

    diff = totaldeposit - pickedup  # deposition - erosion

    return diff, numshifted, sum_contour


def enforceslopes2(topof, vegf, sh, anglesand, angleveg, th):
    """Function to enforce the angle of repose; open boundaries (18 oct 2010). Returns an updated topography.
    - topof         : topography map [topo]
    - vegf          : map of combined vegetation effectiveness [veg] [0,1]
    - sh            : slab height [slabheight]
    - anglesand     : angle of repose for bare sand [repose_bare]
    - angleveg      : angle of repose for vegetated cells [repose_veg]
    - th            : switching threshold [repose_threshold]"""

    steplimitsand = math.floor(math.tan(anglesand * math.pi / 180) / sh)  # Maximum allowed height difference for sandy cells
    steplimitsanddiagonal = math.floor(math.sqrt(2) * math.tan(anglesand * math.pi / 180) / sh)  # Maximum allowed height difference for sandy cells along diagonal
    steplimitveg = math.floor(math.tan(angleveg * math.pi / 180) / sh)  # Maximum allowed height difference for cells vegetated > threshold
    steplimitvegdiagonal = math.floor(math.sqrt(2) * math.tan(angleveg * math.pi / 180) / sh)  # Maximum allowed height difference for cells vegetated  along diagonal > threshold

    steplimit = (vegf < th) * steplimitsand + (vegf >= th) * steplimitveg  # Map with max height diff seen from each central cell
    steplimitdiagonal = (vegf < th) * steplimitsanddiagonal + (vegf >= th) * steplimitvegdiagonal  # Idem for diagonal

    M, N = topof.shape  # Retrieve dimensions of area
    slopes = np.zeros([M, N, 8])  # Pre-allocating matrix to improve calculation speed
    exceeds = np.zeros([M, N, 8])  # Pre-allocating matrix to improve calculation speed

    total = 0  # Number of avalanches
    slabsmoved = 1  # Initial number to get the loop going
    avcount = 0

    while slabsmoved > 0:
        # Padding with same cell to create open boundaries
        topof = np.column_stack((topof[:, 0], topof, topof[:, -1]))
        topof = np.row_stack((topof[0, :], topof, topof[-1, :]))

        # All height differences relative to centre cell (positive is sloping upward away, negative is sloping downward away)
        # Directions are clockwise starting from upward/North [-1,0]=1, and so-on
        central = topof[1: -1, 1: -1]  # (2: -1 -1) --> (1 : -1)
        slopes[:, :, 0] = topof[0: -2, 1: -1] - central  # dir 1 [-1,0]
        slopes[:, :, 4] = topof[2:, 1: -1] - central  # dir 5 [+1,0]
        slopes[:, :, 2] = topof[1: -1, 2:] - central  # dir 3 [0,+1]
        slopes[:, :, 6] = topof[1: -1, 0:-2] - central  # dir 7 [0,-1]
        slopes[:, :, 1] = topof[0:-2, 2:] - central  # dir 2 [-1,+1]
        slopes[:, :, 3] = topof[2:, 2:] - central  # dir 4 [+1,+1]
        slopes[:, :, 5] = topof[2:, 0:-2] - central  # dir 6 [+1,-1]
        slopes[:, :, 7] = topof[0:-2, 0:-2] - central  # dir 8 [-1,-1]

        minima = np.amin(slopes, axis=2)  # Identify for each cell the value of the lowest (negative) slope of all 8 directions

        # True (1) if steepest slope is in this direction & is more negative than the repose limit
        exceeds[:, :, 0] = (minima == slopes[:, :, 0]) * (slopes[:, :, 0] < -steplimit)
        exceeds[:, :, 1] = (minima == slopes[:, :, 1]) * (slopes[:, :, 1] < -steplimitdiagonal)
        exceeds[:, :, 2] = (minima == slopes[:, :, 2]) * (slopes[:, :, 2] < -steplimit)
        exceeds[:, :, 3] = (minima == slopes[:, :, 3]) * (slopes[:, :, 3] < -steplimitdiagonal)
        exceeds[:, :, 4] = (minima == slopes[:, :, 4]) * (slopes[:, :, 4] < -steplimit)
        exceeds[:, :, 5] = (minima == slopes[:, :, 5]) * (slopes[:, :, 5] < -steplimitdiagonal)
        exceeds[:, :, 6] = (minima == slopes[:, :, 6]) * (slopes[:, :, 6] < -steplimit)
        exceeds[:, :, 7] = (minima == slopes[:, :, 7]) * (slopes[:, :, 7] < -steplimitdiagonal)

        # If there are multiple equally steepest slopes that exceed the angle of repose, one of them needs to be assigned and the rest set to 0
        k = np.argwhere(np.sum(exceeds, axis=2) > 1)  # Identify cells with multiple equally steepest minima in different directions that all exceed repose angle
        if k.size != 0:
            for i in range(len(k)):
                row, col = k[i]  # Recover row and col #s from k
                a1 = np.random.rand(1, 1, 8) * exceeds[row, col, :]  # Give all equally steepest slopes in this cell a random number
                exceeds[row, col, :] = (a1 == np.max(a1))  # Pick the largest random number and set the rest to zero

        # Begin avalanching
        topof[1: -1, 1: -1] = topof[1: -1, 1: -1] \
                              - exceeds[:, :, 0] + np.roll(exceeds[:, :, 0], -1, axis=0) \
                              - exceeds[:, :, 4] + np.roll(exceeds[:, :, 4], 1, axis=0) \
                              - exceeds[:, :, 2] + np.roll(exceeds[:, :, 2], 1, axis=1) \
                              - exceeds[:, :, 6] + np.roll(exceeds[:, :, 6], -1, axis=1) \
                              - exceeds[:, :, 1] + np.roll(exceeds[:, :, 1], (-1, 1), axis=(0, 1)) \
                              - exceeds[:, :, 3] + np.roll(exceeds[:, :, 3], (1, 1), axis=(0, 1)) \
                              - exceeds[:, :, 5] + np.roll(exceeds[:, :, 5], (1, -1), axis=(0, 1)) \
                              - exceeds[:, :, 7] + np.roll(exceeds[:, :, 7], (-1, -1), axis=(0, 1))

        topof = topof[1: -1, 1: -1]  # Remove padding
        slabsmoved = np.sum(exceeds)  # Count moved slabs during this loop
        total = total + slabsmoved  # Total number of moved slabs since beginning of m-file

    return topof, total


def marine_processes3_diss3e(total_tide, msl, slabheight, cellsizef, topof, eqtopof, vegf, m26f, m27af, m28f, pwavemaxf, pwaveminf, depthlimitf, shelterf, pcurr):
    """Calculates the effects of high tide levels on the beach and foredune (with stochastic element)
    % Beachupdate in cellular automata fashion
    % Sets back a certain length of the profile to the equilibrium;
    % if waterlevel exceeds dunefoot, this lead to dune erosion;
    % otherwise, only the beach is reset.
    %
    % tide          : height of storm surge [slabs]
    % slabheight    : slabheight in m [slabheightm] [m]
    % cellsizef     : interpreted cell size [cellsize] [m]
    % topof         : topography map [topo] [slabs]
    % eqtopof       : equilibrium beach topography map [eqtopo] [slabs]
    % vegf          : map of combined vegetation effectiveness [veg] [0-1]
    % m26f          : parameter for dissipation strength ~[0.01 - 0.02]
    % m27af         : wave energy factor
    % m28f          : resistance of vegetation: 1 = full, 0 = none.
    % pwavemaxf     : maximum erosive strenght of waves (if >1: overrules vegetation)
    % pwaveminf     : in area with waves always potential for action
    % depthlimitf   : no erosive strength in very shallow water
    % shelterf      : exposure of sheltered cells: 1 = full shelter, 0 = no shelter.
    % phydro         : probability of erosion due to any other hydrodynamic process rather than waves"""

    # --------------------------------------
    # WAVE RUNUP
    # Offshore measured tide (sealevf) has to be converted to effective tide
    # level at the shoreline. The vertical limit of wave runup (R) is a function of
    # tide, slope and wave conditions. Since wave conditions are correlated
    # with tide level (higher waves associated with higher storms), we use a
    # simplified expression where R = f(tide, slope).
    #
    # Original expression of wave runup height controlled by foreshore slope
    # (Stockdon et al, 2006):
    #   Irribarren number = b/sqrt(H/L)
    #   for dissipative beaches (Irribarren number < 0.3):
    #       runup_m = 0.043 * sqrt(H*L);
    #   for intermediate or reflective beaches (Irribarren number >= 0.3):
    #       runup = 1.1*(.35*tan(b)*sqrt(H*L) + sqrt(H*L*(0.563*tan(b^2)+0.004))/2);

    # Derive gradient of eqtopo as an approximation of foreshore slope
    loc_upper_slope = np.argwhere(eqtopof[0, :] > 15)
    loc_lower_slope = np.argwhere(eqtopof[0, :] > -15)

    if len(loc_upper_slope) == 0 or len(loc_lower_slope) == 0:
        b = 0.01
    else:
        b = np.nanmean(np.gradient(eqtopof[0, loc_lower_slope[0]: loc_upper_slope[0]] * slabheight))

    # Tidal elevation above MSL
    tide = total_tide - msl
    tide_m = tide * slabheight
    msl_m = msl * slabheight

    # Calculate wave height and length based on empirical relationships between tide level and wave conditions
    H = max(0, -2.637 + 2.931 * tide_m)
    L = max(0, -30.59 + 46.74 * tide_m)

    # Runup as a function of wave conditions (H, L) and foreshore slope (b)
    if L == 0 and H > 0:  # Dissipative beach
        runup_m = 0.043 * math.sqrt(H * L)
    elif (L > 0 and H > 0) and b / math.sqrt(H / L) < 0.3:  # Dissipative beach
        runup_m = 0.043 * math.sqrt(H * L)
    else:  # Intermediate or reflective beach
        runup_m = 1.1 * (0.35 * math.tan(b) * math.sqrt(H * L) + math.sqrt(H * L * (0.563 * math.tan(b ** 2) + 0.004)) / 2)

    # Add runup to tide to arrive at total water level (=tide + runup + msl)
    totalwater_m = tide_m + runup_m + msl_m
    totalwater = totalwater_m / slabheight

    # --------------------------------------
    # IDENTIFY CELLS EXPOSED TO WAVES
    # by dunes and embryodunes, analogous to shadowzones but no angle

    toolow = topof < totalwater  # [0 1] Give matrix with cells that are potentially under water
    pexposed = np.ones(topof.shape)  # Initialise matrix of cells exposed [T] and not exposed [F] to high water
    for m20 in range(len(topof[:, 0])):  # Run along all the rows
        twlloc = np.argwhere(topof[m20, :] >= totalwater)
        if len(twlloc) > 0:  # If there are any sheltered cells
            m21 = twlloc[0][0]  # Finds for every row the first instance where the topography exceeds the sea level
            pexposed[m20, m21: -1] = 1 - shelterf  # Subtract shelter from exposedness: sheltered cells are less exposed

    # --------------------------------------
    # FILL TOPOGRAPHY TO EQUILIBRIUM

    inundatedf = copy.deepcopy(pexposed)  # Bool matrix of cells that exposed to sea water

    # --------------------------------------
    # WAVES

    waterdepth = (totalwater - topof) * pexposed * slabheight  # [m] Exposed is used to avoid negative waterdepths
    waterdepth[waterdepth <= depthlimitf] = depthlimitf  # This limits high dissipitation when depths are limited; remainder of energy is transferred landward

    # Initialise dissiptation matrices
    diss = np.zeros(topof.shape)
    cumdiss = np.zeros(topof.shape)

    # Calculate dissipation
    loc = np.argwhere(topof[0, :] > -10)  # Find location to start dissipation; limits dissipation to cells greater than -1 m in elevation (assuming 0.1 m slabheight)

    if len(loc) == 0:
        loc = [1]
    elif loc[0] == 0:
        loc[0] = 1

    for m25 in range(int(loc[0]), topof.shape[1]):  # Do for all columns
        diss[:, m25] = (cellsizef / waterdepth[:, m25]) - (cellsizef / waterdepth[:, loc[0]][:, 0])  # Dissipation corrected for cellsize; calcs difference is dissipation between cells in first row and cells in row m25
        cumdiss[:, m25] = diss[:, m25 - 1] + cumdiss[:, m25 - 1]  # Cumulative dissipation from the shore, excluding the current cell; lowers hydrodynamic energy for landward celss

    # --------------------------------------
    # CALCULATING PROBABILITY OF HYDRODYNAMIC EROSION (phydro = pwave - pinun)

    # Dissipation of wave strength across the topography (wave strength times dissiptation)
    pwave = (m27af * pwavemaxf - m26f * cumdiss)
    pwave_ero = (m27af * pwavemaxf - m26f * cumdiss)  # Wave used for dune attack

    # Normalize and remove negatives
    pwave_ero[pwave_ero < 0] = 0  # Set negative values to 0
    pwave_ero = pwave_ero / np.max(pwave_ero)  # Set max between 1 and 0
    pwave[pwave < 0] = 0  # Set negative values to 0
    pwave = pwave / np.max(pwave)  # Set max between 1 and 0

    pwave[np.logical_and.reduce((pwave < pwaveminf, topof < totalwater, pexposed > 0))] = pwaveminf  # In area with waves always potential for action

    pcurr = toolow * pcurr

    # Local reduction of erosive strength due to vegetation
    pbare = 1 - m28f * vegf  # If vegf = 1, still some chance for being eroded

    # Ssumming up erosion probabilities
    phydro = copy.deepcopy(pwave)
    phydro_ero = pwave_ero + pcurr

    # --------------------------------------
    # UPDATING THE TOPOGRAPHY

    pbeachupdate = pbare * phydro * (topof < totalwater) * pexposed  # Probability for beachupdate, also indication of strength of process (it does not do anything random)
    pbeachupdate[pbeachupdate < 0] = 0  # Keep probabilities to 0 - 1 range
    pbeachupdate[pbeachupdate > 1] = 1

    pbeachupdate_ero = pbare * phydro_ero * (topof > totalwater)
    pbeachupdate_ero[pbeachupdate_ero < 0] = 0  # Keep probabilities to 0 - 1 range
    pbeachupdate_ero[pbeachupdate_ero > 1] = 1

    # Stochastic element
    width, length = pbeachupdate.shape
    dbeachupdate = np.random.rand(width, length) < pbeachupdate  # Do beachupdate only for random cells
    dbeachupdate_ero = np.random.rand(width, length) < pbeachupdate_ero  # Do beachupdate only for random cells

    # # marine_processes3_diss3e
    topof = topof - ((topof - eqtopof) * dbeachupdate)  # Adjusting the topography
    topof[topof < -10] = eqtopof[topof < -10]  # Update those that migrated to the ocean by the open boundary

    # # Changed after revision (JK 21/01/2015)
    # # Limit filling up of topography to the maximum water level, so added (accreted) cells cannot be above the maximum water level
    # eqtopof[(eqtopof > totalwater)] = totalwater
    #
    # topof = topof - (topof - eqtopof) * dbeachupdate  # Adjusting the topography
    # topof[np.logical_and(topof > totalwater, dbeachupdate > 0)] = totalwater
    # # Changed after revision (FGS)
    # # Above proposed change has the counterpart of a lot of filling when beach width is short, since it starts to build-up dunes instead of eroding it

    if np.isnan(np.sum(dbeachupdate_ero)) is False:
        topof[np.logical_and(topof >= totalwater, dbeachupdate_ero > 0)] = topof[np.logical_and(topof >= totalwater, dbeachupdate_ero > 0)] - (topof[np.logical_and(topof >= totalwater, dbeachupdate_ero > 0)] - eqtopof[np.logical_and(topof >= totalwater, dbeachupdate_ero > 0)])

    return topof, inundatedf, pbeachupdate, diss, cumdiss, pwave


def marine_processes(total_tide, msl, slabheight, cellsizef, topof, eqtopof, vegf, m26f, m27af, m28f, pwavemaxf, pwaveminf, depthlimitf, shelterf):
    """Calculates the effects of high tide levels on the beach and foredune (with stochastic element)
    % Beachupdate in cellular automata fashion
    % Sets back a certain length of the profile to the equilibrium;
    % if waterlevel exceeds dunefoot, this lead to dune erosion;
    % otherwise, only the beach is reset.
    %
    % tide: height of storm surge [slabs]
    % slabheight: slabheight in m [slabheightm] [m]
    % cellsizef: interpreted cell size [cellsize] [m]
    % topof: topography map [topo] [slabs]
    % eqtopof: equilibrium beach topography map [eqtopo] [slabs]
    % vegf: map of combined vegetation effectiveness [veg] [0-1]
    % m26f: parameter for dissipation strength ~[0.01 - 0.02]
    % m27af: wave energy factor
    % m28f: resistance of vegetation: 1 = full, 0 = none.
    % pwavemaxf: maximum erosive strenght of waves (if >1: overrules vegetation)
    % pwaveminf: in area with waves always potential for action
    % depthlimitf: no erosive strength in very shallow water
    % shelterf: exposure of sheltered cells: 1 = full shelter, 0 = no shelter.
    % phydro: probability of erosion due to any other hydrodynamic process rather than waves"""

    # --------------------------------------
    # WAVE RUNUP
    # Offshore measured tide (sealevf) has to be converted to effective tide
    # level at the shoreline. The vertical limit of wave runup (R) is a function of
    # tide, slope and wave conditions. Since wave conditions are correlated
    # with tide level (higher waves associated with higher storms), we use a
    # simplified expression where R = f(tide, slope).
    #
    # Original expression of wave runup height controlled by foreshore slope
    # (Stockdon et al, 2006):
    #   Irribarren number = b/sqrt(H/L)
    #   for dissipative beaches (Irribarren number < 0.3):
    #       runup_m = 0.043 * sqrt(H*L);
    #   for intermediate or reflective beaches (Irribarren number >= 0.3):
    #       runup = 1.1*(.35*tan(b)*sqrt(H*L) + sqrt(H*L*(0.563*tan(b^2)+0.004))/2);

    # Derive gradient of eqtopo as an approximation of foreshore slope
    b = np.nanmean(np.gradient(eqtopof[0, :] * slabheight))

    # Tidal elevation above MSL
    tide = total_tide - msl
    tide_m = tide * slabheight
    msl_m = msl * slabheight

    H = max(0, -2.637 + 2.931 * tide_m)
    L = max(0, -30.59 + 46.74 * tide_m)

    # Runup as a function of wave conditions (H, L) and foreshore slope (b)
    if L == 0 and H > 0:
        runup_m = 0.043 * math.sqrt(H * L)
    elif (L > 0 and H > 0) and b / math.sqrt(H / L) < 0.3:
        runup_m = 0.043 * math.sqrt(H * L)
    else:
        runup_m = 1.1 * (0.35 * math.tan(b) * math.sqrt(H * L) + math.sqrt(H * L * (0.563 * math.tan(b ** 2) + 0.004)) / 2)

    # Add runup to tide to arrive at total water level (=tide + runup + msl)
    totalwater_m = tide_m + runup_m + msl_m
    totalwater = totalwater_m / slabheight

    # --------------------------------------
    # IDENTIFY CELLS EXPOSED TO WAVES
    # by dunes and embryodunes, analogous to shadowzones but no angle

    # toolow = topof < totalwater  # [0 1] Give matrix with cells that are potentially under water
    pexposed = np.ones(topof.shape)  # Initialise matrix
    for m20 in range(len(topof[:, 0])):  # Run along all the rows
        twlloc = np.argwhere(topof[m20, :] >= totalwater)
        if len(twlloc) > 0:  # If there are any sheltered cells
            m21 = twlloc[0][0]  # Finds for every row the first instance where the topography exceeds the sea level
            pexposed[m20, m21: -1] = 1 - shelterf  # Subtract shelter from exposedness: sheltered cells are less exposed

    # --------------------------------------
    # FILL TOPOGRAPHY TO EQUILIBRIUM

    inundatedf = copy.deepcopy(pexposed)  # Inundated is the area that really receives sea water

    # --------------------------------------
    # WAVES

    waterdepth = (totalwater - topof) * pexposed * slabheight  # [m] Exposed is used to avoid negative waterdepths
    waterdepth[waterdepth <= depthlimitf] = depthlimitf  # This limits high dissipitation when depths are limited; remainder of energy is transferred landward

    # Initialise dissiptation matrices
    diss = np.zeros(topof.shape)
    cumdiss = np.zeros(topof.shape)

    # Calculate dissipation
    diss[:, 0] = cellsizef / waterdepth[:, 0]  # Dissipation corrected for cellsize
    cumdiss[:, 0] = diss[:, 0]
    for m25 in range(1, topof.shape[1]):  # Do for all columns
        diss[:, m25] = cellsizef / waterdepth[:, m25]  # Dissipation corrected for cellsize
        cumdiss[:, m25] = diss[:, m25 - 1] + cumdiss[:, m25 - 1]  # Cumulative dissipation from the shore, excluding the current cell

    # Initial wave strength m27f
    m27f = m27af * totalwater

    # Dissipation of wave strength across the topography (wave strength times dissiptation)
    pwave = (pwavemaxf - m26f * cumdiss) * m27f
    pwave[np.logical_and.reduce((pwave < pwaveminf, topof < totalwater, pexposed > 0))] = pwaveminf  # In area with waves always potential for action
    # pwave[waterdepth < slabheightmf * depthlimitf] = 0  # No erosive strength in very shallow water

    # Local reduction of erosive strength due to vegetation
    pbare = 1 - m28f * vegf  # If vegf = 1, still some chance for being eroded

    # Updating the topography
    pbeachupdate = pbare * pwave * pexposed  # Probability for beachupdate, also indication of strength of process (it does not do anything random)
    pbeachupdate[pbeachupdate < 0] = 0  # Keep probabilities to 0 - 1 range
    pbeachupdate[pbeachupdate > 1] = 1

    # Changed after revision (JK 21/01/2015)
    # Limit filling up of topography to the maximum water level so added (accreted) cells cannot be above the maximum water level
    # eqtopof[eqtopof > sealevf) = sealevf
    topof = topof - (topof - eqtopof) * pbeachupdate  # Adjusting the topography
    topof[np.logical_and(topof > totalwater, pbeachupdate > 0)] = totalwater

    return topof, inundatedf, pbeachupdate, diss, cumdiss, pwave


def growthfunction1_sens(spec, sed, A1, B1, C1, D1, E1, P1):
    """Input is the vegetation map, and the sedimentation balance in units of cell dimension (!), i.e. already adjusted by the slabheight
    Output is the change in vegetation effectiveness
    Ammophilia-like vegetation"""

    # Physiological range (needs to be specified)
    minimum = 0.0
    maximum = 1.0

    # Vertices (these need to be specified)
    x1 = A1  # was x1 = -1.4
    x2 = B1  # was x2 = 0.1
    x3 = C1  # was x3 = 0.45
    x4 = D1  # was x4 = 0.85
    x5 = E1  # was x5 = 1.4
    y1 = -1.0  # y1 = -1.0; y1 = -1.0
    y2 = 0.0  # y2 = 0.0; y2 = 0.0
    y3 = P1  # y3 = 0.4; y3 = P1;
    y4 = 0.0  # y4 = 0.0; y4 = 0.0
    y5 = -1.0  # y5 = -1.0; y5 = -1.0

    # Slopes between vertices (calculated from vertices)
    s12 = (y2 - y1) / (x2 - x1)
    s23 = (y3 - y2) / (x3 - x2)
    s34 = (y4 - y3) / (x4 - x3)
    s45 = (y5 - y4) / (x5 - x4)

    leftextension = (sed < x1) * -1
    firstleg = (sed >= x1) * (sed < x2) * ((sed - x1) * s12 + y1)
    secondleg = (sed >= x2) * (sed < x3) * ((sed - x2) * s23 + y2)
    thirdleg = (sed >= x3) * (sed < x4) * ((sed - x3) * s34 + y3)
    fourthleg = (sed >= x4) * (sed < x5) * ((sed - x4) * s45 + y4)
    rightextension = (sed >= x5) * -1

    spec = spec + leftextension + firstleg + secondleg + thirdleg + fourthleg + rightextension

    spec[spec < minimum] = minimum
    spec[spec > maximum] = maximum

    return spec


def growthfunction2_sens(spec, sed, A2, B2, D2, E2, P2):
    """Input is the vegetation map, and the sedimentation balance in units of cell dimension (!), i.e. already adjusted by the slabheight
    Output is the change in vegetation effectiveness
    Buckthorn-type vegetation"""

    # Physiological range (needs to be specified)
    minimum = 0.0  # was -0.5
    maximum = 1.0  # was 1.5

    # Vertices (these need to be specified)
    x1 = A2  # was x1 = -1.3
    x2 = B2  # was x2 = -0.65
    x3 = 0  # was x3 = 0
    x4 = D2  # was x4 = 0.2
    x5 = E2  # was x5 = 2.2
    y1 = -1.0  # y1 = -1.0
    y2 = 0.0  # y2 = 0.0
    y3 = P2  # y3 = 0.1
    y4 = 0.0  # y4 = 0.0
    y5 = -1.0  # y5 = -1.0

    # Slopes between vertices (calculated from vertices)
    s12 = (y2 - y1) / (x2 - x1)
    s23 = (y3 - y2) / (x3 - x2)
    s34 = (y4 - y3) / (x4 - x3)
    s45 = (y5 - y4) / (x5 - x4)

    leftextension = (sed < x1) * -1
    firstleg = (sed >= x1) * (sed < x2) * ((sed - x1) * s12 + y1)
    secondleg = (sed >= x2) * (sed < x3) * ((sed - x2) * s23 + y2)
    thirdleg = (sed >= x3) * (sed < x4) * ((sed - x3) * s34 + y3)
    fourthleg = (sed >= x4) * (sed < x5) * ((sed - x4) * s45 + y4)
    rightextension = (sed >= x5) * -1

    spec = spec + leftextension + firstleg + secondleg + thirdleg + fourthleg + rightextension

    spec[spec < minimum] = minimum
    spec[spec > maximum] = maximum

    return spec


def lateral_expansion(veg, dist, prob):
    """LATERAL_EXPANSION implements lateral expansion of existing vegetation patches.
    1) mark cells that lie within <dist> of existing patches: probability for new vegetated cells = 1
    2) cells not adjacent to existing patches get probability depending on elevation: pioneer most likely to establish between 3 and 5 m + sea level.
    Returns logical array of which cells veg has successfully expanded into."""

    # Pad vegetation matrix with zeros for rolling
    veg = veg > 0
    vegpad = np.zeros(np.add(veg.shape, (2, 2)))
    vegpad[1: -1, 1: -1] = veg
    veg3 = copy.deepcopy(vegpad)

    # Add shifted matrices to initial matrix to include boundaries
    for i in [-dist, 0, dist]:
        for j in [-dist, 0, dist]:
            # Only 4 neighbouring cells (Von Neumann neighbourhood)
            # if i*3+5+j % 2 == 0:
            veg2 = np.roll(vegpad, (i, j), axis=(0, 1))
            veg3 = veg3 + veg2

    veg3[veg3 > 1] = 1
    lateral_expansion_possible = veg3[1: -1, 1: -1]

    # Lateral expansion only takes place in a fraction <prob> of the possible cells
    width, length = veg.shape
    lateral_expansion_allowed = np.random.rand(width, length) < (lateral_expansion_possible * prob)
    # Include existing vegetation to incorporate growth or decay of existing patches
    lateral_expansion_allowed = lateral_expansion_allowed + veg
    lateral_expansion_allowed = lateral_expansion_allowed > 0

    return lateral_expansion_allowed


def establish_new_vegetation(topof, mht, prob):
    """Establishment of pioneer veg in previously bare cells.
    Represents the germination and development of veg from seeds or rhizome fragments distributed by the sea or wind.
    Returns logical array of which cells new veg has successfully established in."""

    # Convert topography to height above current sea level
    topof = topof - mht

    # Vertices (these need to be specified)
    x1 = 0.0
    x2 = 2.0
    x3 = 3.0
    x4 = 4.5
    x5 = 20.0
    y1 = prob
    y2 = prob
    y3 = prob
    y4 = prob
    y5 = prob

    # Slopes between vertices (calculated from vertices)
    s12 = (y2 - y1) / (x2 - x1)
    s23 = (y3 - y2) / (x3 - x2)
    s34 = (y4 - y3) / (x4 - x3)
    s45 = (y5 - y4) / (x5 - x4)

    leftextension = (topof < x1) * 0
    firstleg = (topof >= x1) * (topof < x2) * ((topof - x1) * s12 + y1)
    secondleg = (topof >= x2) * (topof < x3) * ((topof - x2) * s23 + y2)
    thirdleg = (topof >= x3) * (topof < x4) * ((topof - x3) * s34 + y3)
    fourthleg = (topof >= x4) * (topof < x5) * ((topof - x4) * s45 + y4)
    rightextension = (topof >= x5) * 0

    pioneer_establish_prob = leftextension + firstleg + secondleg + thirdleg + fourthleg + rightextension
    width, length = topof.shape
    pioneer_established = np.random.rand(width, length) < pioneer_establish_prob

    return pioneer_established
