# memory-bot
Robot with memory. Project for Learning for Interact Robots

## Set up
```
module load anaconda

conda activate maniskill_env  ## Use your maniskill conda environment name

pip install openai
pip install opencv-python
```

## experiments

* exp-1 -- original example `https://github.com/haosulab/ManiSkill/blob/main/mani_skill/examples/motionplanning/panda/solutions/pick_cube.py`
* exp-2 -- exp-1 with cup.glb added to the environment. The robot can pick up the cup
* exp-3 -- exp-2 with pandas robot swapped to pandas-wrist cam. Code saves wrist cam footage to image
* exp-4 -- exp-3 with multiple objects on the table
* exp-5 -- exp-4 with robot able to execute the entire re-arrange task

