# Demo material

Two ways of flying the simulated Tello, each flown for real and recorded.

| control path | what flew it | flight video | dashboard | command log |
|---|---|---|---|---|
| **GUI controller** | a tkinter controller, unmodified, buttons pressed in order | `gui_controller_flight.mp4` | `gui_controller_dashboard.png` | `gui_controller_commands.jsonl` |
| **Console / CLI** | `run_cli.py`, the djitellopy API typed at the prompt | `console_flight.mp4` | `console_dashboard.png` | `console_commands.jsonl`, `console_transcript.txt` |

Both flew in the furnished lab with a 1.0 m/s draught and turbulence, against a
simulated drone speaking real SDK over real UDP on 127.0.0.1.

No GUI controller ships with this project, so the first row cannot be
reproduced from a fresh clone exactly as flown. Its artefacts are kept because
they are the simulator's own output, not the controller's: a command log and a
rendered flight. Point `--controller` at any tkinter Tello controller to fly
the same path yourself.

The `.mp4`s here are the **simulator's own render of the flight** — the room, the
path, altitude, battery and wind. They show what the drone did, not what the
operator saw.

## Screen recordings

`scripts/record_demos.py` produces the other kind: a screen recording of the real
programs flying, with the live 3D view open beside them and captions burned in.

```bash
source .venv/bin/activate
cd tello_sim
python scripts/record_demos.py            # both, about 4 minutes
python scripts/record_demos.py cli        # just the console one
python scripts/record_demos.py controller # just the GUI one
```

It records the **whole screen**, so close or move anything you would not want on
film, and leave the machine alone while it runs. It needs a lit display: with the
lid shut every frame comes back black, and it will say so rather than record
nothing. Output lands in this folder as `tello_controller_demo.mp4` and
`tello_cli_demo.mp4`.

What it does, for each demo:

1. starts the simulator and opens the 3D view in a bare Chrome window on the left,
2. puts the controller (or a Terminal running the console) on the right,
3. records, while `scripts/demo_drive_controller.py` presses the buttons — or
   `scripts/demo_drive_cli.py` types the calls one character at a time into a pty,
   so the terminal echoes them exactly as it would echo a person,
4. burns in the captions from the timeline the driver wrote as it went, and adds a
   title card.

Nothing is staged: the presses, the typing, the timings and the refusals are the
real programs' own. The drivers are worth reading if you want to change the
flight — the sequence is a list at the top of each.

## Re-running the flights without recording anything

```bash
python run_sim3d.py --no-browser --wind 1.0 --seed 3    # in one terminal
python scripts/demo_drive_controller.py                 # in another

python scripts/demo_drive_cli.py                        # owns its own simulator
```

Each writes a run to `logs/`, and `render.dashboard(...)` / `render.animate(...)`
turn that into the dashboard and the flight video.
