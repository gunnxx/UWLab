# 1. Getting into the container

Everything in this guide runs inside the Forge image, `uwlab-omnireset:local`. The image holds Isaac Sim,
Isaac Lab 3.0 and the project's dependencies, and your working copy is mounted into it at `/workspace/uwlab`,
so edits on the host are seen inside at once. This page gets you from a fresh machine to a prompt inside that
container, with a window when you want one.

## Step 1. Build the image once

```bash
docker build -f docker/forge/Dockerfile -t uwlab-omnireset:local docker/forge
```

This takes a while the first time and does not need repeating until the Dockerfile changes. The helper script
in the next step expects the tag `uwlab-omnireset:local`. Set `IMAGE=<tag>` in the environment if you built it
under another name.

## Step 2. Open a shell, or run one command

The script `docker/forge/run_local.sh` starts the container the same way a Forge job sees it. Without
arguments it drops you into a shell at the repository root:

```bash
docker/forge/run_local.sh
```

With a command it runs that command and exits:

```bash
docker/forge/run_local.sh python scripts/environments/list_envs.py --keyword OmniReset
```

The script keeps two folders under `~/.cache/uwlab-local` between runs. One stands in for the team's blob
disk, which is where the pipeline of page 3 caches assets and datasets, and the other holds Isaac Sim's shader
and Kit caches, which is why the second start of any scene is much faster than the first. Set `LOCAL_ROOT` to
move both.

If you prefer an editor, VS Code can open the repository inside the same image. Press F1, choose
"Dev Containers: Reopen in Container", and use the integrated terminal. The commands in this guide are the
same there.

## Step 3. Get a window when you need one

Isaac Lab 3.0 runs headless by default, which is what every recorder and training run wants. When you want to
see the scene, add `--viz kit` to the command. The script forwards your display into the container, and the
dev container does the same, but the container's root user must be allowed to use it. Run this once per login
on the host:

```bash
xhost +si:localuser:root
```

Expect about a minute of start-up before a window shows a scene, and longer on the very first run while the
shader cache fills. A 16 GB GPU holds one Kit window at a time, so close a windowed run before starting the
next, and never start one while a recorder or a training run is using the GPU.

## Step 4. Learn the two conventions

Every environment takes its object pair as a Hydra override appended to the command. Most examples in this
guide use the peg and the peg hole:

```bash
env.scene.insertive_object=peg env.scene.receptive_object=peghole
```

Any other configuration field can be overridden the same way, by its dotted path from `env`. A field that the
configuration does not declare needs a leading plus sign, `+key=value`, or Hydra refuses it.

Commands in the rest of this guide are written as typed inside the container. To run one from the host,
prefix it with `docker/forge/run_local.sh`.

## Check that it works

From a shell inside the container, this prints the registered OmniReset environments and exits:

```bash
python scripts/environments/list_envs.py --keyword OmniReset
```

If it lists tasks, you are ready for [page 2](02_environments.md), which explains what they are.
