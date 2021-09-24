import logging
from typing import Dict, List, Tuple
import os

import numpy as np

from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
import astropy.units as u

from collector import Program
from greedy_max.schedule import Observation, Visit
import selector.horizons as hz
from common.structures.band import Band
from common.structures.site import Site
from common.structures.target import TargetTag


class Ranker:
    def __init__(self, sites: frozenset[Site], times: List[Time]):
        self.sites = sites
        self.times = times
        self.params = Ranker._params()

    # TODO: unused variables obs and site_location.
    def _query_coordinates(self,
                           obs: Observation,
                           site: Site, night, tag, des, coords,
                           ephem_dir, site_location, overwrite=False, checkephem=False, ):
        # Query coordinates, including for nonsidereal targets from Horizons
        # iobs = list of observations to process, otherwise, all non-sidereal

        query_coords = []

        for nn in night:
            tstart = self.times[nn][0].strftime('%Y%m%d_%H%M')
            tend = self.times[nn][-1].strftime('%Y%m%d_%H%M')

            if tag is TargetTag.Sidereal:
                coord = coords
            else:
                if tag is None or des is None:
                    coord = None
                else:
                    if tag is TargetTag.Comet:
                        hzname = 'DES=' + des + ';CAP'
                    elif tag == 'asteroid':
                        hzname = 'DES=' + des + ';'
                    else:
                        hzname = hz.get_horizon_id(des)

                    # File name
                    ephname = ephem_dir + '/' + site.name + '_' + des.replace(' ', '').replace('/', '') + '_' + \
                              tstart + '-' + tend + '.eph'

                    ephexist = os.path.exists(ephname)

                    if checkephem and ephexist and not overwrite:
                        # Just checking that the ephem file exists, dont' need coords
                        coord = None
                    else:

                        try:
                            horizons = hz.Horizons(site.value.upper(), airmass=100., daytime=True)
                            time, ra, dec = horizons.Coordinates(hzname,
                                                                 self.times[nn][0],
                                                                 self.times[nn][-1],
                                                                 step='1m',
                                                                 file=ephname, overwrite=overwrite)
                            coord = None
                        except:
                            logging.error(f'Horizons query failed for {des}.')
                            coord = None
                        else:
                            coord = SkyCoord(ra, dec, frame='icrs', unit=(u.rad, u.rad))

            query_coords.append(coord)

        return query_coords

    def score(self,
              visits: List[Visit],
              programs: Dict[int, Program],
              location: Dict[Site, EarthLocation],
              inight: int,
              ephem_path: str,
              pow: int = 2,
              metpow: float = 1.0,
              vispow: float = 1.0,
              whapow: float = 1.0,
              remaining: Time = None):

        def combine_score(x: np.ndarray) -> np.ndarray:
            return np.array([np.max(x)]) if 0 not in x else np.array([0])

        for visit in visits:
            site = visit.site
            site_location = location[site]
            visit_score = np.empty((0, len(self.times[inight])), dtype=float)
            for obs in visit.observations:

                score = np.zeros(len(self.times[inight]))
                program_id = obs.get_program_id()
                program = programs[program_id]

                if remaining is None:
                    remaining = (obs.length - obs.observed) * u.hr

                cplt = (program.used_time + remaining) / program.time

                # Metric and slope
                metrc, metrc_s = self._metric_slope(np.array([cplt.value]),
                                                    np.ones(1, dtype=int) * program.band,
                                                    np.ones(1) * 0.8,
                                                    pow=pow,
                                                    thesis=program.thesis,
                                                    thesis_factor=1.1)

                # Get coordinates
                coord = self._query_coordinates(obs,
                                                site,
                                                [inight],
                                                obs.target.tag,
                                                obs.target.designation,
                                                obs.target.coordinates,
                                                ephem_path,
                                                site_location)[0]

                # HA/airmass
                ha = obs.visibility.hour_angle[inight]

                if coord is not None:
                    if site_location.lat < 0. * u.deg:
                        decdiff = site_location.lat - np.max(coord.dec)
                    else:
                        decdiff = np.min(coord.dec) - site_location.lat
                else:
                    decdiff = 90. * u.deg

                if abs(decdiff) < 40. * u.deg:
                    c = np.array([3., 0.1, -0.06])  # weighted to slightly positive HA
                else:
                    c = np.array([3., 0., -0.08])  # weighted to 0 HA if Xmin > 1.3

                wha = c[0] + c[1] * ha / u.hourangle + (c[2] / u.hourangle ** 2) * ha ** 2
                kk = np.where(wha <= 0.0)[0][:]
                wha[kk] = 0.

                # p = metrc[0] * wha  # Match Sebastian's Sept 30 Gurobi test?
                # p = metrc[0] * metrc_s[0] * self.visfrac[site_name][ii] * wha  # My favorite
                # p = metrc_s[0] * self.visfrac[site_name][ii] * wha # also very good
                p = (metrc[0] ** metpow) * (obs.visibility.fraction ** vispow) * (wha ** whapow)

                score[obs.visibility.visibility[inight]] = p[obs.visibility.visibility[inight]]

                obs.score = score

                visit_score = np.append(visit_score, np.array([score]), axis=0)

            visit.score = np.apply_along_axis(combine_score, 0, visit_score)[0]

    def _metric_slope(self,
                      completion: np.ndarray,
                      band: np.ndarray,
                      b3min: np.ndarray,
                      pow: int = 1,
                      thesis: bool = False,
                      thesis_factor: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:

        """
        Compute the metric and the slope as a function of completness fraction and band

        Parameters
            completion: array/list of program completion fractions
            band: integer array of bands for each program
            b3min: array of Band 3 minimum time fractions (Band 3 minimum time / Allocated program time)
            params: dictionary of parameters for the metric
            pow: exponent on completion, pow=1 is linear, pow=2 is parabolic
        """

        eps = 1.e-7
        nn = len(completion)
        metric = np.zeros(nn)
        metric_slope = np.zeros(nn)
        for ii in range(nn):
            sband = Band(band[ii])
            param = self.params[sband]

            # If Band 3, then the Band 3 min fraction is used for xb
            if band[ii] == Band.Band3:
                xb = b3min[ii]
                # b2 = xb * (param['m1'] - param['m2']) + param['xb0']
            else:
                xb = param['xb']
                # b2 = param['b2']

            # Determine the intercept for the second piece (b2) so that the functions are continuous
            b2 = 0
            if pow == 1:
                b2 = xb * (param['m1'] - param['m2']) + param['xb0'] + param['b1']
            elif pow == 2:
                b2 = param['b2'] + param['xb0'] + param['b1']

            # Finally, calculate piecewise the metric and slope
            if completion[ii] <= eps:
                metric[ii] = 0.0
                metric_slope[ii] = 0.0
            elif completion[ii] < xb:
                metric[ii] = param['m1'] * completion[ii] ** pow + param['b1']
                metric_slope[ii] = pow * param['m1'] * completion[ii] ** (pow - 1.)
            elif completion[ii] < 1.0:
                metric[ii] = param['m2'] * completion[ii] + b2
                metric_slope[ii] = param['m2']
            else:
                metric[ii] = param['m2'] * 1.0 + b2 + param['xc0']
                metric_slope[ii] = param['m2']

        if thesis:
            metric += thesis_factor
            # metric *= thesis_factor
            # metric_slope *= thesis_factor

        return metric, metric_slope

    # Only call this method as part of the Ranker constructor to avoid having to dynamically recalculate these
    # constants every time scoring is done.
    @staticmethod
    def _params() -> Dict[Band, Dict[str, float]]:
        params9 = {Band.Band1: {'m1': 1.406, 'b1': 2.0, 'm2': 0.50, 'b2': 0.5, 'xb': 0.8, 'xb0': 0.0, 'xc0': 0.0},
                   Band.Band2: {'m1': 1.406, 'b1': 1.0, 'm2': 0.50, 'b2': 0.5, 'xb': 0.8, 'xb0': 0.0, 'xc0': 0.0},
                   Band.Band3: {'m1': 1.406, 'b1': 0.0, 'm2': 0.50, 'b2': 0.5, 'xb': 0.8, 'xb0': 0.0, 'xc0': 0.0},
                   Band.Band4: {'m1': 0.000, 'b1': 0.1, 'm2': 0.00, 'b2': 0.0, 'xb': 0.8, 'xb0': 0.0, 'xc0': 0.0}}
        # m2 = {Band.Bard3: 0.5, Band.Band2: 3.0, Band.Band1:10.0} # use with b1*r where r=3
        m2 = {Band.Band4: 0.0, Band.Band3: 1.0, Band.Band2: 6.0, Band.Band1: 20.0}  # use with b1 + 5.
        xb = 0.8
        r = 3.0
        # b1 = np.array([6.0, 1.0, 0.2])
        b1 = 1.2
        for band in [Band.Band3, Band.Band2, Band.Band1]:
            #     b2 = b1*r - m2[band]
            # intercept for linear segment
            b2 = b1 + 5. - m2[band]
            # parabola coefficient so that the curves meet at xb: y = m1*xb**2 + b1 = m2*xb + b2
            m1 = (m2[band] * xb + b2) / xb ** 2
            params9[band]['m1'] = m1
            params9[band]['m2'] = m2[band]
            params9[band]['b1'] = b1
            params9[band]['b2'] = b2
            params9[band]['xb'] = xb
            # zeropoint for band separation
            b1 += m2[band] * 1.0 + b2
        return params9

