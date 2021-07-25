import multiprocessing
import time
from functools import partial
from multiprocessing import Process
from random import choice, shuffle
from typing import Any, Dict, List, Optional, Set, Tuple


class Parallelizer(object):
    """
    Parallelizes the execution of task groups

    A task is a tuple of worker and string representing a task group.
    A task group is a group of workers that execute the same task.
    If any worker finds a solution, all other workers of the same
    task group will be killed/not started. 

    Attributes:
        tasks (List[Tuple[Any, str]]): List of functions to execute
        task_groups (List[str]): List of task groups.
        process_to_task_group (Dict[Process, str]): Maps processes to task groups.
        process_to_task_id (Dict[Process, int]): Maps processes to tasks.
        task_group_results (Dict[str, Any]): Stores results for individual task groups.
        max_processes (int): maximum number of processes to launch.
    """

    def __init__(self, tasks: 'List[Tuple[Any, str]]', max_processes: int = 0):
        """
        Initializes the Parallelizer.

        Args:
            tasks (List[Tuple[partial[Any], str]]): List of tasks.
            max_processes (int, optional): Number of workers to execute in parallel. Defaults to 0.
        """
        self.functions: 'List[Any]' = [t for (t, _) in tasks]
        self.task_groups: List[str] = [tg for (_, tg) in tasks]

        self.process_to_task_group: Dict[Process, str] = dict()
        self.process_to_task_id: Dict[Process, int] = dict()
        self.task_group_results: Dict[str, Any] = dict()

        if max_processes == 0:
            max_processes = multiprocessing.cpu_count()
        self.max_processes: int = max_processes

    def execute(self) -> List[Any]:
        """
        Executes a list of tasks parallel.

        Once a task for the same task group was successful, other workers
        will be killed/not launched.

        Returns:
            List[Any]: List of results filled by workers.
        """
        # initialise parallel data structures
        manager = multiprocessing.Manager()
        results: List[Any] = manager.list()
        processes: List[Process] = [None] * len(self.functions)  # type: ignore

        # set task_groups to not-finished
        task_group_states = {}
        for task in self.task_groups:
            task_group_states[task] = 0

        # initialise process mappings
        process_to_task_group: Dict[Process, str] = dict()
        process_to_index: Dict[Process, int] = dict()

        # initialise processes
        for i in range(len(processes)):
            # extend results
            results.append(None)

            # create process
            processes[i] = multiprocessing.Process(
                target=self.functions[i], args=(results, i))

            # map process to process index
            process_to_index[processes[i]] = i

            # choose task group randomly
            process_to_task_group[processes[i]] = self.task_groups[i]

        # initialise data structures
        active_processes: Set[Process] = set()
        done: Set[Process] = set()
        process_counter = -1

        # random permutation of process indexes
        random_process_indices = list(range(len(processes)))
        shuffle(random_process_indices)

        start_time = time.time()
        # iterate until all processes have been processed
        while len(done) < len(processes):
            # add more processes, if # processes < # cpu cores and there are processes remaining
            while len(active_processes) < self.max_processes and process_counter < len(processes) - 1:
                # increase index
                process_counter += 1
                # random process index
                random_process_index = random_process_indices.pop()
                # get next process
                process = processes[random_process_index]

                # get process' task group
                task_group = process_to_task_group[process]

                # print random_process_index, task_group

                # process' taskgroup has been solved
                if task_group_states[task_group]:
                    done.add(process)

                # get next process
                if process in done:
                    continue

                # start process
                process.start()

                # add to active processes
                active_processes.add(process)

            # if there are active processes
            if active_processes:
                # choose random process
                process = choice(list(active_processes.copy()))

                # process has been terminated
                if not process.is_alive():
                    # get process index
                    process_index = process_to_index[process]
                    # get result
                    result = results[process_index]

                    # if process terminated with a result:
                    if result:
                        # get process' task group
                        task_group = process_to_task_group[process]
                        # set task group to finished
                        task_group_states[task_group] = 1

                        # store task group's result
                        self.task_group_results[task_group] = result

                        # terminate active processes in current task group
                        for process in active_processes.copy():
                            # process is in the same task group?
                            if task_group == process_to_task_group[process]:
                                # kill process
                                process.terminate()
                                # add to done
                                done.add(process)
                                # remove from active processes
                                active_processes.remove(process)

                    # delete process
                    else:
                        # add to done
                        done.add(process)
                        # remove from active processes
                        active_processes.remove(process)

        return results
