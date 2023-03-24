# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from typing import List, FrozenSet, Union, NewType

import strawberry  # noqa
from lucupy.minimodel import ObservationID, Site, ALL_SITES

from scheduler.config import ConfigurationError
from scheduler.core.plans import Plan, Plans, Visit


def parse_sites(sites: Union[str, List[str]]) -> FrozenSet[Site]:
    """Parse Sites from Sites scalar

        Args:
            sites (Union[str, List[str]]): Option can be a list of sites or a single one

        Returns:
            FrozenSet[Site]: a frozen site that contains lucupy Site enums
                corresponding to each site.
        """

    def parse_specific_site(site: str):
        try:
            return Site[site]
        except KeyError:
            raise ConfigurationError('Missing site', site)

    if sites == 'ALL_SITES':
        # In case of ALL_SITES option, return lucupy alias for the set of all Site enums
        return ALL_SITES

    if isinstance(sites, list):
        return frozenset(map(parse_specific_site, sites))
    else:
        # Single site case
        return frozenset([parse_specific_site(sites)])


Sites = strawberry.scalar(NewType("Sites", FrozenSet[Site]),
                          description="Depiction of the sites that can be load to the collector",
                          serialize=lambda x: x,
                          parse_value=lambda x: parse_sites(x)) # noqa