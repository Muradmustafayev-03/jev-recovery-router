"""Bound an explicitly launched cloud phase, including environment setup time."""
from __future__ import annotations
import argparse
from collections import namedtuple
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import time


Process = namedtuple("Process", "pid ppid pgid start_ticks state")


def process_snapshot(proc_root=Path("/proc")):
    processes = {}
    for directory in proc_root.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            # comm can contain spaces and parentheses; fields after its final ')'
            # begin at field 3, making starttime (field 22) index 19.
            fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            pid = int(directory.name)
            processes[pid] = Process(pid, int(fields[1]), int(fields[2]), int(fields[19]), fields[0])
        except (OSError, ValueError, IndexError):
            continue  # A process may disappear during the snapshot.
    return processes


def enable_subreaper():
    """Keep orphaned descendants attached to this standalone Linux supervisor."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        error = ctypes.get_errno()
        raise OSError(error, "Cannot enable descendant reaping; refusing an unbounded cloud phase")


class Descendants:
    def __init__(self, child_pid):
        self.child_pid = child_pid
        self.supervisor_pid = os.getpid()
        self.supervisor_group = os.getpgrp()
        self.identities = {}

    def refresh(self):
        processes = process_snapshot()
        if self.supervisor_pid not in processes:
            raise RuntimeError("Cannot verify process ownership from /proc")
        owned = {pid for pid, start in self.identities.items()
                 if pid in processes and processes[pid].start_ticks == start}
        # Subreaper adoption makes a formerly unobserved orphan our direct child.
        while True:
            discovered = {pid for pid, proc in processes.items()
                          if pid != self.supervisor_pid
                          and (proc.ppid == self.supervisor_pid or proc.ppid in owned)}
            if discovered <= owned:
                break
            owned.update(discovered)
        for pid in owned:
            self.identities[pid] = processes[pid].start_ticks
        live = {pid: processes[pid] for pid in owned if processes[pid].state not in ("Z", "X")}
        for pid in owned:
            proc = processes[pid]
            if pid != self.child_pid and proc.ppid == self.supervisor_pid and proc.state == "Z":
                try:
                    os.waitpid(pid, os.WNOHANG)
                except (ChildProcessError, ProcessLookupError):
                    pass
        return processes, live

    def signal_owned(self, signum):
        processes, live = self.refresh()
        groups = {proc.pgid for proc in live.values()}
        for group in groups:
            members = {pid for pid, proc in processes.items() if proc.pgid == group and proc.state not in ("Z", "X")}
            # Never signal a remembered, empty or potentially unrelated group.
            if group > 1 and group != self.supervisor_group and members and members <= live.keys():
                try:
                    os.killpg(group, signum)
                except ProcessLookupError:
                    pass
            else:
                # An owned process in a shared group may be stopped individually.
                for pid in members & live.keys():
                    try:
                        os.kill(pid, signum)
                    except ProcessLookupError:
                        pass
        return live

    def stop(self, child, grace_seconds=65, kill_seconds=5):
        for signum, seconds in ((signal.SIGTERM, grace_seconds), (signal.SIGKILL, kill_seconds)):
            deadline = time.monotonic() + seconds
            while True:
                live = self.signal_owned(signum)
                child.poll()  # Popen alone owns the original child's wait status.
                if not live:
                    return {"status": "complete", "surviving_processes": []}
                if time.monotonic() >= deadline:
                    break
                time.sleep(min(.25, max(0, deadline - time.monotonic())))
        _, live = self.refresh()
        return {"status": "incomplete" if live else "complete",
                "surviving_processes": [proc._asdict() for proc in live.values()]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["bootstrap", "experiment"])
    parser.add_argument("--overall-hours", type=float, default=9)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    started = json.loads((root / ".bootstrap-timing.json").read_text())["start_unix"]
    deadline = started + args.overall_hours * 3600
    if args.phase == "bootstrap":
        deadline = min(deadline, time.time() + 3600)
        command = ["bash", "scripts/bootstrap_a100.sh"]
    else:
        command = ["bash", "scripts/run_a100.sh", "--stress", "--max-hours", str(args.overall_hours)]
    directory = root / "runs/cloud"
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / f"{args.phase}-status.json"
    state = dict(phase=args.phase, supervisor_pid=os.getpid(), start_unix=time.time(),
                 billing_start_unix=started, deadline_unix=deadline, status="starting", command=command)
    def save():
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        os.replace(temporary, state_path)
    save()
    if time.time() >= deadline:
        raise RuntimeError("The cloud deadline has already passed")
    env = os.environ.copy()
    env.update(TORCH_CUDA_FLAVOR="cu118", PYTHONUNBUFFERED="1", PIP_DEFAULT_TIMEOUT="60")
    child = None
    descendants = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    try:
        enable_subreaper()
        with (directory / f"{args.phase}.log").open("a", buffering=1) as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     env=env, start_new_session=True)
            descendants = Descendants(child.pid)
            state.update(status="running", child_pid=child.pid)
            save()
            while True:
                returncode = child.poll()
                _, live = descendants.refresh()
                if returncode is not None:
                    if live:
                        raise RuntimeError("Cloud launcher exited while descendant processes remain active")
                    break
                if time.time() >= deadline:
                    raise TimeoutError("Absolute cloud runtime deadline reached")
                time.sleep(2)
            state.update(status="complete" if child.returncode == 0 else "failed", exit_code=child.returncode)
    except BaseException as error:
        state.update(status="stopped", error_type=type(error).__name__, error=str(error))
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal.SIG_IGN)
        if descendants is not None:
            try:
                state["cleanup"] = descendants.stop(child)
            except Exception as cleanup_error:
                state["cleanup"] = dict(status="incomplete", error_type=type(cleanup_error).__name__,
                                        error=str(cleanup_error))
        raise
    finally:
        state.update(end_unix=time.time(), total_cloud_elapsed_hours=(time.time()-started)/3600,
                     billing_caveat="Stopping this process does not stop notebook billing.")
        save()


if __name__ == "__main__":
    main()
