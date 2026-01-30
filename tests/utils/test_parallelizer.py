from __future__ import annotations

from msynth.utils.parallelizer import Parallelizer


class _FakeManager:
    def list(self):
        return []


class _FakeProcess:
    def __init__(self, target=None, args=()):
        self._target = target
        self._args = args
        self._alive = False

    def start(self):
        self._alive = True
        if self._target:
            self._target(*self._args)
        self._alive = False

    def is_alive(self):
        return self._alive

    def terminate(self):
        self._alive = False


class _FakeMultiprocessing:
    def __init__(self):
        self.Process = _FakeProcess

    def Manager(self):
        return _FakeManager()

    def cpu_count(self):
        return 1


def worker_success(results, index) -> None:
    results[index] = ("ok", index)


def worker_none(_results, _index) -> None:
    return None


def test_parallelizer_records_first_success(monkeypatch) -> None:
    tasks = [
        (worker_success, "group1"),
        (worker_none, "group1"),
    ]

    monkeypatch.setattr("msynth.utils.parallelizer.multiprocessing", _FakeMultiprocessing())
    parallelizer = Parallelizer(tasks, max_processes=1)
    results = parallelizer.execute()

    assert "group1" in parallelizer.task_group_results
    assert parallelizer.task_group_results["group1"][0] == "ok"
    assert any(r is not None for r in results)


def test_parallelizer_no_success_results_empty(monkeypatch) -> None:
    tasks = [
        (worker_none, "group1"),
        (worker_none, "group1"),
    ]

    monkeypatch.setattr("msynth.utils.parallelizer.multiprocessing", _FakeMultiprocessing())
    parallelizer = Parallelizer(tasks, max_processes=1)
    results = parallelizer.execute()

    assert parallelizer.task_group_results == {}
    assert all(r is None for r in results)
