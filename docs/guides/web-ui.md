# Web UI

Autonima ships a local web application alongside the CLI. Both drive the same
pipeline and produce identical outputs; they differ in what they keep track of
for you.

```bash
pip install -e .[ui]
autonima ui --workspace .
```

The app binds to `127.0.0.1:8765` by default and opens a browser. It is a local
tool: there is no authentication, and it should not be bound to a public
interface.

| Option | Default | |
|---|---|---|
| `--workspace` | current directory | where `.autonima-ui/` state is kept |
| `--host` | `127.0.0.1` | bind address |
| `--port` | `8765` | bind port |
| `--open` / `--no-open` | `--open` | open a browser on launch |

## When to use which

The CLI runs **one config, once**. You point it at a YAML file, it executes, it
writes a folder. Everything it knows lives in that folder, which makes it the
right tool for scripted and reproducible work: a shell loop over nine projects,
a Makefile, a job on a cluster.

The UI manages **many projects over time**. It adds a workspace — a
`.autonima-ui/` directory holding `workspace.json` and `projects.json` — that
remembers which projects exist, which runs belong to them, and what happened.
That is the difference: the CLI has no memory between invocations, the UI does.

Use the CLI when the work is repeatable and you want it in version control. Use
the UI when you are developing criteria, comparing configurations, or watching a
long run.

## What the UI can do that the CLI cannot

**Watch a run in progress.** Stage-by-stage progress and streaming logs while
the pipeline executes, rather than a terminal that blocks until it finishes.

**Cancel a run.** The CLI offers no way to stop cleanly short of interrupting
the process.

**Preview before running.** A run preview reports what the cache plan implies —
which stages would re-execute and which would be reused — before anything is
spent on API calls.

**Clone a project.** Copy an existing project and its configuration as the
starting point for a variant. This is how you build a `v1` → `v2` progression
without hand-copying YAML.

**Import an existing project.** Adopt a directory that already contains runs,
so work started at the command line can be continued in the UI.

**Delete with a preview.** See exactly which runs and files a delete would
remove before confirming it.

**Browse meta-analysis artifacts.** List and download the maps and tables a
`meta` run produced, without navigating the output tree by hand.

**Export missing-full-text lists.** Download the PMIDs a run could not retrieve,
as `.txt` or `.csv`, for manual acquisition.

**Manage API credentials.** Store and check the keys the LLM and retrieval
stages need, instead of exporting environment variables per shell.

## What the CLI can do that the UI cannot

**Stop after an early stage.** `autonima run-search` and `autonima run-abstract`
execute part of the pipeline and stop. The UI runs the full pipeline as
configured.

**Compose with other tools.** Shell loops, schedulers, CI, and anything else
that wants a process with an exit code.

**Run without a browser.** On a remote host, `autonima run` needs nothing but a
terminal.

## They share everything that matters

Both read the same YAML configuration, execute the same stages, write the same
`outputs/` layout, and honour the same cache. A project created in the UI can be
run from the CLI afterwards, and a directory produced by the CLI can be imported
into the UI. Neither converts or locks anything.

The one piece of state the UI adds, `.autonima-ui/`, records only which projects
and runs the workspace knows about. Deleting it loses the workspace listing, not
any pipeline output.

## Typical loop

1. Create or import a project, and edit its configuration in the browser.
2. Validate it; fix whatever the validator reports.
3. Preview the run to see what the cache plan will re-execute.
4. Start the run and watch the stages; cancel if the criteria are clearly wrong.
5. Inspect the outputs, adjust the criteria, clone the project, and try again.
6. When the configuration is settled, commit the YAML and run it from the CLI so
   the result is reproducible.

That last step matters. The UI is for the iteration; the CLI is for the record.

## See also

- [CLI Usage](cli.md)
- [Configuration](configuration.md)
- [Outputs](outputs.md)
