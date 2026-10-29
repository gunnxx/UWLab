# OmniReset command guide

This guide walks through everything you run by hand for the OmniReset environments: getting into the
container, finding the right environment, recording the datasets a new robot needs, training and playing a
policy, and looking at a scene when something does not behave. Each page is written as a sequence of steps
that you can follow from top to bottom, with the command, what it produces and how to check that it worked.

Every command in this guide runs inside the Forge image, where `python` is Isaac Sim's own interpreter. If
you have not opened the container yet, start with the first page.

## The pages

1. [Getting into the container](01_getting_in.md). Build the image once, open a shell or run a single
   command, get a window when you want one, and learn the two conventions every later command uses.
2. [Which environment is which](02_environments.md). The three robots, the three families of tasks, the
   task ids, and the dataset files each environment reads at reset.
3. [Recording the datasets](03_datasets.md). The seven recording steps for a new robot in the order they
   depend on each other, by hand or all at once through the pipeline script.
4. [Training and playing a policy](04_training.md). A smoke-test run, a real run, where the checkpoints
   land, and how to watch a trained policy.
5. [Looking at a scene and finding out why it fails](05_inspection.md). The keyboard inspector, replaying
   a recorded dataset, probing a reset-state family that yields nothing, and tuning a grasp by hand.
6. [Tests and formatting](06_tests_and_formatting.md). The two commands to run before a commit.

## Which page do I need?

- You have just cloned the repository or rebuilt the image. Page 1, then page 2.
- You changed a robot asset, an object or a table layout and the reset-state datasets are now stale.
  Page 3, and read its last section first, which says what has to be re-recorded.
- You want to train, or you have a checkpoint and want to see it move. Page 4.
- A scene looks wrong, a recorder reports no successes, or you want to see what a reset draws. Page 5.
- You want to understand how the grasp sampler chooses where the hand goes before you touch it. That is
  not a command page but a note, [How the grasp sampler works](../omnireset_grasp_sampling.html).

## Two conventions used everywhere

The object pair is a Hydra override appended to any command. Most examples use the peg and the peg hole:

```bash
env.scene.insertive_object=peg env.scene.receptive_object=peghole
```

Commands are shown as they are typed inside the container, from the repository root at `/workspace/uwlab`.
To run one from the host instead, prefix it with `docker/forge/run_local.sh`, as page 1 explains.
