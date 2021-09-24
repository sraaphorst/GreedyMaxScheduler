import re
from typing import Dict, List

import numpy as np

from astropy.units.quantity import Quantity
import astropy.units as u

from greedy_max.category import Category
from common.structures.band import Band
from common.structures.conditions import SkyConditions
from common.structures.instrument import Instrument
from common.structures.elevation import ElevationConstraints
from common.structures.site import Site
from common.structures.target import Target
from common.structures.too_type import ToOType


class Observation: 
    """
    The data that comprises an observation.
    """
    def __init__(self,
                 idx,
                 name: str,
                 band: Band,
                 category: Category,
                 observed: int, # observation time on time slots
                 length: int, # acq+time (this need to be change)
                 instrument: Instrument,
                 sky_conditions: SkyConditions,
                 elevation: ElevationConstraints,
                 target: Target,
                 status: str,
                 too_status: ToOType,
                 #acquisition: int
                 ) -> None:
        self.idx = idx
        self.name = name
        self.band = band
        self.category = category
        self.observed = observed
        self.length = length
        self.instrument = instrument
        self.sky_conditions = sky_conditions
        self.elevation = elevation
        self.target = target
        self.status = status #Observation status
        self.too_status = too_status
        self.visibility = None
        self.score = None
        #self.acquisition = acquisition
  
    def acquisition(self):
        """
        Calculate acquisition time of the observation
        """
        mode = self.instrument.observation_mode()
        name = self.instrument.name 
        
        gmos = {'imaging': 6.*u.min, 'longslit': 16.*u.min, 'ifu': 18.*u.min, 'mos': 18.*u.min}
        f2 = {'imaging': 6.*u.min, 'longslit': 20.*u.min, 'mos': 30.*u.min}

        acquisition_lookup = {
                                'GMOS': gmos[mode] if 'GMOS' in self.name else 0.*u.min ,
                                'Flamingos2': f2[mode] if 'Flamingos2' in self.name else 0.*u.min,
                                'NIFS': 11.*u.min,
                                'GNIRS':  15.*u.min,
                                'NIRI': 6.*u.min,
                                'GPI': 10.*u.min,
                                'GSAOI':  30.*u.min,
                                'Alopeke': 6.0*u.min,
                                'Zorro': 6.0*u.min,
                                'MAROON-X': 10*u.min,
                                'IGRINS': 10*u.min,
                                'Visitor Instrument': 10*u.min
        }
        
        return  acquisition_lookup['GMOS'] if 'GMOS' in name else acquisition_lookup[name]

    def get_program_id(self):
        """
        Get the ID for the observation's Program
        """
        return self.name[0:self.name.rfind('-')]

    def __str__(self) -> str:
        return f'{self.idx}-{self.name}'

class Visit:
    def __init__(self,
                 idx: int,
                 site: Site, 
                 observations: List[Observation], 
                 calibrations: List[Observation],
                 can_be_split: bool,
                 standard_time: int
                 ) -> None:
        self.idx = idx
        self.site = site
        self.observations = observations # group or a single science observation
        self.calibrations = calibrations # group or a single cal observation
        self.can_be_split = can_be_split # split flag
        self.standard_time = standard_time # standard time in time slots 
        self.score = None
        self.sky_conditions = self.sky_constraints()

    def length(self) -> int:
        """
        Calculate the length of the unit based on both observation and calibrations times
        """
        obs_slots = sum([obs.length for obs in self.observations])

        if self.standard_time > 0: # not a single science observation
            standards_needed = max(1, int(obs_slots // self.standard_time))

            if standards_needed == 1:
                cal_slots = self.calibrations[0].length #take just one
            else:
                cal_slots = sum([cal.length for cal in self.calibrations])

            return obs_slots + cal_slots
        else:
            return obs_slots

    def observed(self) -> int:
        """
        Calculate the observed time for both observation and calibrations
        """
        obs_slots = sum([obs.observed for obs in self.observations])
        cal_slots = sum([cal.observed for cal in self.calibrations])
        return obs_slots + cal_slots
    
    def acquisition(self) -> None:
        """
        Add acquisition overhead to the total length of each observation in the unit
        """
        for observation in self.observations:
            if observation.observed < observation.length: # not complete observation
                observation.length += observation.acquisition

    def get_observations(self) -> Dict[int,Observation]:
        total_obs = {}
        for obs in self.observations:
            total_obs[obs.idx] = obs
        for cal in self.calibrations:
            total_obs[cal.idx] = cal
        return total_obs
    
    def airmass(self, obs_idx: int) -> float:
        """
        Get airmass values for the observation
        """
        if obs_idx in self.observations:
            return self.observations[obs_idx].visibility.airmass
        if obs_idx in self.calibrations:
            return self.calibrations[obs_idx].visibility.airmass
        else:
            return None
    
    def sky_constraints(self) -> None:
        '''
        Create a new SkyConditions object based on the observation level objects 
        Use the most restrictive value for each condition. 
        '''
        restrictive_iq = min([obs.sky_conditions.iq for obs in self.observations])
        restrictive_bg = min([obs.sky_conditions.sb for obs in self.observations])
        restrictive_cc = min([obs.sky_conditions.cc for obs in self.observations])
        restrictive_wv = min([obs.sky_conditions.wv for obs in self.observations])

        return SkyConditions(restrictive_iq,restrictive_bg,restrictive_cc,restrictive_wv)
            

    def __contains__(self, obs_idx:int) -> bool:
        
        if obs_idx in [sci.idx for sci in self.observations]:
            return True
        elif obs_idx in [cal.idx for cal in self.calibrations]:
            return True
        else:
            return False
    
    def __str__(self) -> str:
        return f'Visit {self.idx} \n '+\
                f'-- observations: \n {[str(obs) for obs in self.observations]} \n' +\
                f'-- calibrations: {[str(cal) for cal in self.calibrations]} \n'
             

ValuesByObservation = Dict[int,List[float]]

class TimeSlots:
    decoder = {'A':'0','B':'1','Q':'0',
                'C':'1','LP':'2','FT':'3',
                'SV':'8','DD':'9'}
    pattern = '|'.join(decoder.keys())
    def __init__(self, 
                 time_slot_length: Quantity, 
                 weights: Dict[Site,ValuesByObservation],
                 airmass:  Dict[Site,ValuesByObservation], 
                 total_amount: int, 
                 fpu: Dict[Site, List[str]], 
                 fpur: Dict[Site, List[str]], 
                 grat: Dict[Site, List[str]], 
                 instruments: Dict[Site, List[str]], 
                 lgs: Dict[Site,bool] , 
                 mode: Dict[Site,str], 
                 fpu2b: Dict[str,str], 
                 ifus: Dict[str,str]):
        
        self.slot_length = time_slot_length
        self.weights = weights
        self.airmass = airmass
        self.total = total_amount
        self.fpu = fpu
        self.fpur = fpur
        self.grating = grat 
        self.instruments = instruments
        self.laser_guide = lgs
        self.mode = mode
        self.fpu_to_barcode = fpu2b
        self.ifu = ifus

    def non_zero_intervals(self, site: Site, obs_idx: int, interval: np.ndarray) -> np.ndarray:

        weights_on_interval = self.weights[site][obs_idx][interval]
        # Create an array that is 1 where the weights is greater than 0, and pad each end with an extra 0.
        isntzero = np.concatenate(([0], np.greater(weights_on_interval, 0), [0]))
        absdiff = np.abs(np.diff(isntzero))
        # Get the ranges for each non zero interval
        ranges = np.where(absdiff == 1)[0].reshape(-1, 2) 
        return ranges
    def _decode_mask(self, mask_name: str ) -> str:
        return '1'+ re.sub(f'({TimeSlots.pattern})', 
                           lambda m: TimeSlots.decoder[m.group()], mask_name).replace('-','')[6:]
    
    def is_instrument_available(self, site: Site, instrument: str) -> bool:
        return instrument in self.instruments[site]
    
    def max_weight(self, site: Site, idx: int, interval: np.ndarray) -> float:
        return np.max(self.weights[site][idx][interval])

    def is_mask_available(self, site: Site, fpu_mask: str, mask_type: str) -> bool:
        
        barcode = None
        if fpu_mask in self.fpu_to_barcode:
            barcode = self.fpu_to_barcode[fpu_mask]
        else:
            barcode = self._decode_mask(fpu_mask)
        if mask_type == 'FPU':
            return barcode in self.fpu[site] 
        elif mask_type == 'FPUr':
            return barcode in self.fpur[site] 
        elif mask_type == 'GRAT':
            return barcode in self.grating[site]
        else:
            return False
        

