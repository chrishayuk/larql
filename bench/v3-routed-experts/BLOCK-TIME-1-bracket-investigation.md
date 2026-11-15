# BLOCK-TIME-1: bracket instability in the admissible run

Status: **investigation of existing records. Nothing here
was measured for this note, and nothing here changes revision 2 or its
result.** The run is
[`20261115-01-admissible`](link-results/block-time/20261115-01-admissible/);
its result is recorded in the
[runs README](link-results/block-time/README.md).

Sources used: the run's `trials.json` (per-trial wall time, decomposition and
one-minute load average), trial file modification times, the macOS unified
log and power log for 09:27 to 10:09 UTC, file-system modification times,
and git worktree metadata. No process sampling was running during the run,
so what other processes used the CPU is inferred, not observed.

## What drifted

The bracket gate compares the `sequential` 0 ms trials either side of a
delay. Splitting the brackets by component shows where the time went
(ms per evaluated position):

| Trial | Ends (UTC) | Wall | Worker execution | Coordinator | Transport | Load (1 min) |
|---|---|---|---|---|---|---|
| bracket 012 | 09:36:17 | 127.3 | 84.5 | 35.2 | 6.4 | 3.6 |
| bracket 018 | 09:40:30 | 133.9 | 88.7 | 36.8 | 7.1 | 4.5 |
| bracket 024 | 09:48:30 | 135.1 | 89.2 | 36.5 | 7.9 | 3.8 |
| bracket 026 | 09:49:30 | 129.1 | 85.5 | 35.8 | 6.6 | 3.8 |
| bracket 032 | 09:57:28 | 129.0 | 85.7 | 35.6 | 6.5 | 4.8 |
| bracket 038 | 10:01:48 | **158.9** | **104.3** | **44.9** | 8.1 | 4.7 |
| bracket 049 | 10:08:06 | 126.5 | 84.0 | 35.1 | 6.3 | 5.0 |

Two episodes, both visible in every trial that overlaps them, not only the
brackets:

1. **A mild episode, about 09:39 to 09:49.** Worker execution in profiled
   trials rose from about 84 to 85 ms to 87 to 92 ms (trials 017 to 025),
   then returned (026: 85.5). The coordinator rose by 1 to 2 ms. This
   voided repeat 1's 5 ms arms (bracket drift 5.1%). Repeat 1's 20 ms arms
   fell inside the same episode but between brackets that agreed (0.9%),
   so they are valid.
2. **A strong episode, about 10:00:30 to 10:05:40.** Worker execution rose
   to 101 to 130 ms and the coordinator to 45 to 63 ms (trials 037 to 043);
   transport barely moved. The load average reached 14.0 at trial-043
   (10:05:06). By trial-045 (10:06:12) the coordinator was back to normal.
   This voided repeat 2's 5 ms and 0 ms arms (20.8% and 22.8%).

**The worker and the coordinator are separate processes, and both slowed
at the same time, while transport through the relay did not.** That is the
signature of a host-wide loss of CPU, by contention or by clock speed, not
of the network, the relay or one process. It is not a property of any arm:
the same arm reads normally outside the episodes.

One consequence of the schedule is visible: **repeat 2's 5 ms arms
(033 to 037, 09:57 to 10:01) were themselves mostly before the strong
episode** (worker execution 86 to 92 ms), and were voided because their
*closing* bracket, run after them, fell inside it. With five or ten trials
between brackets, a disturbance anywhere in a window of about four to six
minutes voids the whole group.

## What the host records show

- **No power, sleep or thermal event is logged.** The power log has no
  entry between 09:27 and 10:13. The unified log has no thermal-pressure
  message in the window; `thermalmonitord` logged nothing. Recovery to the
  baseline at 026 and 049 also argues against cumulative thermal
  throttling. Thermal state was not sampled, so it is not excluded.
- **Memory pressure: nothing found.** The only `memorystatus` messages are
  routine `runningboardd` errors unrelated to pressure.
- **Spotlight indexing bursts coincide with both episodes.** `launchd`
  started `mdworker_shared` about once a minute through the run, but
  **52 times at 09:42 and 29 times at 09:48** (mild episode) and **47 times
  at 10:04** (strong episode), with `mds` logging about 7,700 lines in that
  minute.
- **A git worktree of this repository was created during the quiet
  window.** `~/chris-source/larql-introspect` (branch
  `lql/tensor-introspection`, at main `78d69be4`) was created at 10:04:08
  and checked out by 10:04:10, writing the whole tree; that is the 10:04
  Spotlight burst. Its `target` directory was created at 10:13:42, after the
  run, so no compilation took place in the window. The worktree's
  metadata does not say which session created it. Both peer sessions had
  confirmed quiet, and their confirmations did not mention git checkouts or
  worktree creation, which the request did not explicitly exclude
  ("git or gh commands are fine").
- **The strong episode began before the worktree was created.** The
  coordinator was already slow in trial-037 (49.2 ms, 10:00:15 to 10:01:14)
  and worker execution in bracket 038 (10:01:14 to 10:01:48). What used
  the CPU from about 10:00:30 to 10:04 is not identified by any record
  found.
- **The mild episode has no identified cause.** Besides the Spotlight
  bursts, the unified log shows a burst from `trustd`,
  `BackgroundShortcutRunner` and `siriactionsd` at 09:42 (a Shortcuts
  automation). Neither is shown to have taken a measurable share of CPU.

## What can and cannot be said

- Can: both void groups coincide with host-wide slowdowns that affected the
  worker and the coordinator together; the network path was not the cause.
- Can: a worktree checkout during the quiet window, and the Spotlight
  indexing it triggered, overlap the strong episode.
- Cannot: that the worktree checkout or Spotlight caused either episode.
  The strong episode began about three and a half minutes before the
  checkout, and nothing sampled CPU use by process.
- Cannot: that thermal or frequency changes played no part. They were not
  recorded.

## What follows

The cause of the episodes is not established, but the schedule's exposure
to them is: brackets several minutes apart let one disturbance void a whole
delay. That is fixed independently of the cause by
[BLOCK-TIME-2](BLOCK-TIME-2.md) (draft), which brackets each 5 and 20 ms
pair separately, records host telemetry, and names worktree creation,
checkouts and large file operations in the quiet request. The worktree
creation and the Spotlight activity are recorded here as documented
interference risks, not as causes.
