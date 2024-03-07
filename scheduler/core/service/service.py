# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from datetime import datetime
from typing import FrozenSet, Optional, Dict

import numpy as np
from astropy.time import Time
from lucupy.minimodel import Site, Semester, NightIndex, TimeslotIndex, CloudCover, ImageQuality

from scheduler.core.builder import Blueprints
from scheduler.core.builder.modes import dispatch_with, SchedulerModes
from scheduler.core.components.changemonitor import ChangeMonitor, TimeCoordinateRecord
from scheduler.core.components.collector import Collector
from scheduler.core.components.optimizer import Optimizer
from scheduler.core.components.ranker import RankerParameters, DefaultRanker
from scheduler.core.components.selector import Selector
from scheduler.core.eventsqueue import EventQueue, EveningTwilightEvent, MorningTwilightEvent, Event
from scheduler.core.eventsqueue.nightchanges import NightlyTimeline
from scheduler.core.plans import Plans
from scheduler.core.sources.sources import Sources
from scheduler.core.statscalculator import StatCalculator
from scheduler.services import logger_factory


__all__ = [
    'Service',
]


_logger = logger_factory.create_logger(__name__)


class Service:

    def __init__(self):
        pass

    @staticmethod
    def _setup(night_indices, sites, mode):

        queue = EventQueue(night_indices, sites)
        sources = Sources()
        builder = dispatch_with(mode, sources, queue)
        return builder

    @staticmethod
    def _schedule_nights(night_indices: FrozenSet[NightIndex],
                         sites: FrozenSet[Site],
                         collector: Collector,
                         selector: Selector,
                         optimizer: Optimizer,
                         change_monitor: ChangeMonitor,
                         next_update: Dict[Site, Optional[TimeCoordinateRecord]],
                         queue: EventQueue,
                         ranker_parameters: RankerParameters,
                         cc_per_site: Optional[Dict[Site, CloudCover]] = None,
                         iq_per_site: Optional[Dict[Site, ImageQuality]] = None):

        time_slot_length = collector.time_slot_length.to_datetime()
        nightly_timeline = NightlyTimeline()

        # Add the twilight events for every night at each site.
        # The morning twilight will force time accounting to be done on the last generated plan for the night.
        for site in sites:
            night_events = collector.get_night_events(site)
            for night_idx in night_indices:
                eve_twi_time = night_events.twilight_evening_12[night_idx].to_datetime(site.timezone)
                eve_twi = EveningTwilightEvent(time=eve_twi_time, description='Evening 12° Twilight')
                queue.add_event(night_idx, site, eve_twi)

                morn_twi_time = night_events.twilight_morning_12[night_idx].to_datetime(
                    site.timezone) - time_slot_length
                morn_twi = MorningTwilightEvent(time=morn_twi_time, description='Morning 12° Twilight')
                queue.add_event(night_idx, site, morn_twi)

        for night_idx in sorted(night_indices):
            night_indices = np.array([night_idx])
            ranker = DefaultRanker(collector, night_indices, sites, params=ranker_parameters)

            for site in sorted(sites, key=lambda site: site.name):
                # Site name so we can change this if we see fit.
                site_name = site.name

                # Reset the Selector to the default weather for the night and reset the time record.
                # The evening twilight should trigger the initial plan generation.
                cc_value = cc_per_site and cc_per_site.get(site)
                iq_value = iq_per_site and iq_per_site.get(site)
                selector.update_cc_and_iq(site, cc_value, iq_value)

                # Plan and event queue management.
                plans: Optional[Plans] = None
                events_by_night = queue.get_night_events(night_idx, site)
                if events_by_night.is_empty():
                    raise RuntimeError(f'No events for site {site_name} for night {night_idx}.')

                # We need the start of the night for checking if an event has been reached.
                # Next update indicates when we will recalculate the plan.
                night_events = collector.get_night_events(site)
                night_start = night_events.twilight_evening_12[night_idx].to_datetime(site.timezone)
                next_update[site] = None

                current_timeslot: TimeslotIndex = TimeslotIndex(0)
                next_event: Optional[Event] = None
                next_event_timeslot: Optional[TimeslotIndex] = None
                night_done = False

                while not night_done:
                    # If our next update isn't done, and we are out of events, we're missing the morning twilight.
                    if next_event is None and events_by_night.is_empty():
                        raise RuntimeError(f'No morning twilight found for site {site_name} for night {night_idx}.')

                    if next_event_timeslot is None or current_timeslot >= next_event_timeslot:
                        # Stop if there are no more events.
                        while events_by_night.has_more_events():
                            top_event = events_by_night.top_event()
                            top_event_timeslot = top_event.to_timeslot_idx(night_start, time_slot_length)

                            # TODO: Check this over to make sure if there is an event now, it is processed.
                            # If we don't know the next event timeslot, set it.
                            if next_event_timeslot is None:
                                next_event_timeslot = top_event_timeslot
                                next_event = top_event

                            if current_timeslot > next_event_timeslot:
                                _logger.warning(f'Received event for {site_name} for night idx {night_idx} at timeslot'
                                                f'{next_event_timeslot} < current time slot {current_timeslot}.')

                            # The next event happens in the future, so record that time.
                            if top_event_timeslot > current_timeslot:
                                next_event_timeslot = top_event_timeslot
                                break

                            # We have an event that occurs at this time slot and is in top_event, so pop it from the
                            # queue and process it.
                            events_by_night.pop_next_event()
                            _logger.info(
                                f'Received event for site {site_name} for night idx {night_idx} to be processed '
                                f'at timeslot {next_event_timeslot}: {next_event.__class__.__name__}')

                            # Process the event: find out when it should occur.
                            # If there is no next update planned, then take it to be the next update.
                            # If there is a next update planned, then take it if it happens before the next update.
                            # Process the event to find out if we should recalculate the plan based on it and when.
                            time_record = change_monitor.process_event(site, top_event, plans)
                            if time_record is not None:
                                # In the case that:
                                # * there is no next update scheduled; or
                                # * this update happens before the next update
                                # then set to this update.
                                if next_update[site] is None or time_record.timeslot_idx < next_update[site].timeslot_idx:
                                    next_update[site] = time_record
                                    _logger.debug(f'Next update for site {site_name} scheduled at '
                                                  f'timeslot {next_update[site].timeslot_idx}')

                    # If there is a next update, and we have reached its time, then perform it.
                    # This is where we perform time accounting (if necessary), get a selection, and create a plan.
                    if next_update[site] is not None and current_timeslot >= next_update[site].timeslot_idx:
                        # Remove the update and perform it.
                        update = next_update[site]
                        next_update[site] = None

                        if current_timeslot > update.timeslot_idx:
                            _logger.error(
                                f'Plan update was supposed to happen at site {site.name} for night {night_idx} '
                                f'at timeslot {update.timeslot_idx}, but now timeslot is {current_timeslot}.')

                        # We will update the plan up until the time that the update happens.
                        # If this update corresponds to the night being done, then use None.
                        if update.done:
                            end_timeslot_bounds = {}
                        else:
                            end_timeslot_bounds = {site: update.timeslot_idx}

                        # If there was an old plan and time accounting is to be done, then process it.
                        if plans is not None and update.perform_time_accounting:
                            if update.done:
                                ta_description = 'for rest of night.'
                            else:
                                ta_description = f'up to timeslot {update.timeslot_idx}.'
                            _logger.info(f'Time accounting: site {site_name} for night {night_idx} {ta_description}')
                            collector.time_accounting(plans,
                                                      sites=frozenset({site}),
                                                      end_timeslot_bounds=end_timeslot_bounds)
                            if update.done:
                                # In the case of the morning twilight, which is the only thing that will
                                # be represented here by update.done, we add no plans (None) since the plans
                                # generated up until the terminal time slot will have been added by the event
                                # that caused them.
                                nightly_timeline.add(NightIndex(night_idx),
                                                     site,
                                                     current_timeslot,
                                                     update.event,
                                                     None)

                        # Get a new selection and request a new plan if the night is not done.
                        if not update.done:
                            _logger.info(f'Retrieving selection for {site_name} for night {night_idx} '
                                         f'starting at time slot {current_timeslot}.')
                            selection = selector.select(night_indices=night_indices,
                                                        sites=frozenset([site]),
                                                        starting_time_slots={site: {night_idx: current_timeslot
                                                                                    for night_idx in night_indices}},
                                                        ranker=ranker)

                            # Right now the optimizer generates List[Plans], a list of plans indexed by
                            # every night in the selection. We only want the first one, which corresponds
                            # to the current night index we are looping over.
                            _logger.info(f'Running optimizer for {site_name} for night {night_idx} '
                                         f'starting at time slot {current_timeslot}.')
                            plans = optimizer.schedule(selection)[0]
                            nightly_timeline.add(NightIndex(night_idx),
                                                 site,
                                                 current_timeslot,
                                                 update.event,
                                                 plans[site])

                        # Update night_done based on time update record.
                        night_done = update.done

                    # We have processed all events for this timeslot and performed an update if necessary.
                    # Advance the current time.
                    current_timeslot += 1

        return nightly_timeline

    def run(self,
            mode: SchedulerModes,
            start_vis: Time,
            end_vis: Time,
            sites: FrozenSet[Site],
            ranker_parameters: RankerParameters = RankerParameters(),
            semester_visibility: bool = True,
            num_nights_to_schedule: Optional[int] = None,
            cc_per_site: Optional[Dict[Site, CloudCover]] = None,
            iq_per_site: Optional[Dict[Site, ImageQuality]] = None,
            program_file: Optional[bytes] = None):

        semesters = frozenset([Semester.find_semester_from_date(start_vis.datetime),
                               Semester.find_semester_from_date(end_vis.datetime)])

        if semester_visibility:
            end_date = max(s.end_date() for s in semesters)
            end = Time(datetime(end_date.year, end_date.month, end_date.day).strftime("%Y-%m-%d %H:%M:%S"))
            diff = end_vis - start_vis + 1
            diff = int(diff.jd)
            night_indices = frozenset(NightIndex(idx) for idx in range(diff))
            num_nights_to_schedule = diff
        else:
            night_indices = frozenset(NightIndex(idx) for idx in range(num_nights_to_schedule))
            end = end_vis
            if not num_nights_to_schedule:
                raise ValueError("num_nights_to_schedule can't be None when visibility is given by end date")

        builder = self._setup(night_indices, sites, mode)

        # Build
        collector = builder.build_collector(start_vis,
                                            end,
                                            sites,
                                            semesters,
                                            Blueprints.collector,
                                            program_file)

        selector = builder.build_selector(collector,
                                          num_nights_to_schedule,
                                          Blueprints.selector,
                                          cc_per_site,
                                          iq_per_site)

        # Create the ChangeMonitor and keep track of when we should recalculate the plan for each site.
        change_monitor = ChangeMonitor(collector=collector, selector=selector)

        # Don't use this now, but we will use it when scheduling sites at the same time.
        next_update: Dict[Site, Optional[TimeCoordinateRecord]] = {site: None for site in sites}

        optimizer = builder.build_optimizer(Blueprints.optimizer)

        timelines = self._schedule_nights(night_indices,
                                          sites,
                                          collector,
                                          selector,
                                          optimizer,
                                          change_monitor,
                                          next_update,
                                          builder.events,
                                          ranker_parameters,
                                          cc_per_site,
                                          iq_per_site)

        # Calculate plans stats
        plan_summary = StatCalculator.calculate_timeline_stats(timelines, night_indices, sites, collector)

        return timelines, plan_summary
