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
* exp-4 -- exp-3 with multiple objects on the table. 
    TODOs: 
        Fix the roaming objects issue
* exp-5 -- exp-4 with the robot able to pick up any object on-command
* exp-6 -- exp-5 with the robot able to drop the selected object in designated place
* exp-7 -- exp-6 with the robot able to (pick up + drop off) multiple objects on-command
* exp-8 -- exp-7 with the robot able to take in a recipe, identify objects to re-arrange, and execute
