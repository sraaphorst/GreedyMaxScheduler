import numpy as np
from enum import Enum
from typing import Optional


class Sign(Enum):
    POS = '+'
    NEG = '-'

    @staticmethod
    def apply_sign(num, sign):
        return num if sign == Sign.POS else -1. * num


class HMS:
    def __init__(self, h: int, m: int, s: float):
        self.h = h
        self.m = m
        self.s = s

    def to_deg(self) -> float:
        return rad2deg(self.to_rad())

    def to_rad(self) -> float:
        return hrs2rad(self.h + self.m / 60. + self.s / 3600.)

    @staticmethod
    def from_hrs(hrs: Optional[float]):
        if hrs is None:
            return None

        h = int(hrs)
        tmp = (hrs - h) * 60.
        m = int(tmp)
        s = (tmp - m) * 60.
        return HMS(h, m, s)

    @staticmethod
    def from_deg(deg: Optional[float]):
        if deg is None:
            return None

        rad = deg2rad(deg / 180. * np.pi)
        return HMS.from_rad(rad)

    @staticmethod
    def from_rad(rad: Optional[float]):
        if rad is None:
            return None

        hours = rad * 12. / np.pi
        ra = abs(hours)
        h = int(ra)
        tmp = (ra - h) * 60.
        m = int(tmp)
        s = (tmp - m) * 60.
        return HMS(h, m, s)

    @staticmethod
    def from_str(coordstring: Optional[str]):
        if coordstring is None:
            return None
        c = coordstring.split(':') if ':' in coordstring else coordstring.split()

        if c[0][0] == '-':
            s = '-'
        else:
            s = '+'

        c1 = abs(int(c[0]))
        c2 = int(c[1])
        c3 = float(c[2])

        # Return the sign last so that it may be ignored for HMS triples:
        return HMS(c1, c2, c3)


class DMS:
    def __init__(self, sign: Sign, d: int, m: int, s: float):
        self.sign = sign
        self.d = d
        self.m = m
        self.s = s

    def to_deg(self):
        return Sign.apply_sign(self.d + self.m / 60. + self.s / 3600., self.sign)

    def to_rad(self):
        return deg2rad(self.to_deg())

    @staticmethod
    def from_rad(rad: Optional[float]):
        if rad is None:
            return None

        deg = rad2deg(rad)
        dec = abs(rad2deg(rad))
        d = int(dec)
        tmp = (dec - d) * 60.
        m = int(tmp)
        s = (tmp - m) * 60.
        sign = Sign.POS if deg >= 0 else Sign.NEG

        return DMS(sign, d, m, s)

    @staticmethod
    def from_deg(deg: float):
        if deg is None:
            return None

        dec = abs(deg)
        d = int(dec)
        tmp = (dec - d) * 60.
        m = int(tmp)
        s = (tmp - m) * 60.
        sign = Sign.POS if deg >= 0 else Sign.NEG
        return DMS(sign, d, m, s)

    @staticmethod
    def from_str(coordstring: Optional[str]):
        if coordstring is None:
            return None
        c = coordstring.split(':') if ':' in coordstring else coordstring.split()

        sign = Sign.POS if c[0][0] != '-' else Sign.NEG
        c1 = abs(int(c[0]))
        c2 = int(c[1])
        c3 = float(c[2])

        # Return the sign last so that it may be ignored for HMS triples:
        return DMS(sign, c1, c2, c3)


def rad2deg(rad: float) -> float:
    return rad / np.pi * 180.


def deg2rad(deg: float) -> float:
    return deg / 180. * np.pi


def arcsec2rad(arcsec: float) -> float:
    return arcsec / 3600. / 180. * np.pi


def rad2arcsec(rad: float) -> float:
    return rad / np.pi * 180. * 3600.


def hrs2rad(hrs):
    return hrs / 12. * np.pi


# Calculate the angular separation between two points whose coordinates are given in RA and Dec
# From angsep.py Written by Enno Middelberg 2001
# http://www.stsci.edu/~ferguson/software/pygoodsdist/pygoods/angsep.py
def angsep(ra1rad: float,
           dec1rad: float,
           ra2rad: float,
           dec2rad: float) -> float:
    """ Determine separation in degrees between two celestial objects.
        Arguments are RA and Dec in radians.  Output in radians.
    """
    if abs(ra1rad - ra2rad) < 1e-8 and abs(dec1rad - dec2rad) < 1e-8:  # to avoid arccos errors
        sep = np.sqrt((np.cos((dec1rad + dec2rad) / 2.) * (ra1rad - ra2rad))**2 + (dec1rad - dec2rad)**2)

    else:
        # calculate scalar product for determination of angular separation
        x = np.cos(ra1rad) * np.cos(dec1rad) * np.cos(ra2rad) * np.cos(dec2rad)
        y = np.sin(ra1rad) * np.cos(dec1rad) * np.sin(ra2rad) * np.cos(dec2rad)
        z = np.sin(dec1rad) * np.sin(dec2rad)
        rad = np.arccos(x + y + z) # Sometimes gives warnings when coords match

        if rad < 4.848e-6:  # Use Pythargoras approximation if rad < 1 arcsec (= 4.8e-6 radians)
            sep = np.sqrt((np.cos((dec1rad + dec2rad) / 2.) * (ra1rad - ra2rad))**2 + (dec1rad - dec2rad)**2)
        else:
            sep = rad
    return sep


# Calculate the angular separation between two points whose coordinates are given in RA and Dec
# From Astronomical Algorithms by Jean Meeus, Chapter 16
# cos(separation) = sin(d1)sin(d2) + cos(d1)cos(d2)cos(ra1-ra2)
def angsep2(ra1rad: float,
            dec1rad: float,
            ra2rad: float,
            dec2rad: float) -> float:
    """ Determine separation in degrees between two celestial objects.
        Arguments are RA and Dec in radians.  Output in radians.
    """

    t1 = np.sin(dec1rad) * np.sin(dec2rad)
    t2 = np.cos(dec1rad) * np.cos(dec2rad) * np.cos(ra1rad - ra2rad)
    sep = np.arccos(t1 + t2)

    if sep < 2.9e-3:  # (10 arcmin) then use the Pythargoras approximation
        sep = np.sqrt(((ra1rad - ra2rad) * np.cos((dec1rad + dec2rad) / 2.))**2 + (dec1rad - dec2rad)**2)
    return sep


# Calculate the angular separation between two points whose coordinates are given in RA and Dec
# From Astronomical Algorithms by Jean Meeus, Chapter 16
# cos(separation) = sin(d1)sin(d2) + cos(d1)cos(d2)cos(ra1-ra2)
# This is the same as angsep2, but converted to handle input np arrays
def angsep3(ra1rad: np.ndarray,
            dec1rad: np.ndarray,
            ra2rad: np.ndarray,
            dec2rad: np.ndarray) -> float:
    """ Determine separation in degrees between two celestial objects.
        Arguments are RA and Dec in radians.  Output in radians.
        REQUIRES np arrays.
    """

    t1 = np.sin(dec1rad) * np.sin(dec2rad)
    t2 = np.cos(dec1rad) * np.cos(dec2rad) * np.cos(ra1rad - ra2rad)
    sep = np.arccos(t1 + t2)

    # If < 10 arcmin use the Pythargoras approximation:
    sep[sep < 0.0029] = \
        np.sqrt(((ra1rad[sep < 0.0029] - ra2rad[sep < 0.0029]) *
                 np.cos((dec1rad[sep < 0.0029] + dec2rad[sep < 0.0029]) / 2.))**2 +
                (dec1rad[sep < 0.0029] - dec2rad[sep < 0.0029])**2)
    return sep


# Calculate the angular separation between two points whose coordinates are given in RA and Dec (radians)
# For SMALL separations only (<10 arcmin)
def smallangsep(ra1rad: float,
                dec1rad: float,
                ra2rad: float,
                dec2rad: float) -> Optional[float]:
    if ra1rad is None or dec1rad is None or ra2rad is None or dec2rad is None:
        return None
    return np.sqrt(((ra1rad - ra2rad) * np.cos((dec1rad + dec2rad) / 2.))**2 + (dec1rad - dec2rad)**2)


# Calculate the angular separation between two nearby (<10 arcmin) coordinates in degrees
def smallangsepdeg(ra1deg: float,
                   dec1deg: float,
                   ra2deg: float,
                   dec2deg: float) -> Optional[float]:
    if ra1deg is None or dec1deg is None or ra2deg is None or dec2deg is None:
        return None
    return rad2deg(smallangsep(deg2rad(ra1deg), deg2rad(dec1deg), deg2rad(ra2deg), deg2rad(dec2deg)))
