from multiprocessing import Process
from datetime import datetime
from .runner import StandardRunner
from .task import SchedulerTask, TaskType


from typing import NoReturn


class ProcessManager:
    """
    Main handler for each runner, which is responsible for scheduling the task. 
    """
    def __init__(self, timeout: int, size: int):
        self.realtime_runner = StandardRunner(1, timeout)
        self.standard_runner = StandardRunner(size, timeout)

    def schedule_with_runner(self, task: SchedulerTask, mode: TaskType):
        """
        Schedule a task with the corresponding runner for the given mode.
        """
        # TODO: Probably good for enums but right now the original input is a string
        # so it seems unnecessary to use enums?
        if mode == TaskType.REALTIME:
            return self.realtime_runner.schedule(Process(target=task.target), task.timeout)
        elif mode == TaskType.STANDARD:
            return self.standard_runner.schedule(Process(target=task.target), task.timeout)
        else:
            raise ValueError(f'Invalid mode {mode}')

    def add_task(self, start: datetime, target: callable, mode: TaskType, timeout: int = None) -> NoReturn:
        task = SchedulerTask(start,
                             target,
                             timeout=timeout)
        self.schedule_with_runner(task, mode)
    
    def shutdown(self):
        """
        Callback for shutting down the process manager.
        """
        self.realtime_runner.terminate_all()
        self.standard_runner.terminate_all()
