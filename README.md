# vis_slam_game

This is the solution for the course project for NYU ROB-GY 6203 Robot Perception. 

# Installation steps for the environment
1. Environment Setup  
Execute the following commands to set up the environment:  
```commandline
conda update conda
git clone https://github.com/ai4ce/vis_nav_player.git
cd vis_nav_player
conda env create -f environment.yaml
conda activate game
```

2. Installing LightGlue  
LightGlue can be installed by following the instructions in the official documentation, available [here](https://github.com/cvg/LightGlue), or by executing these commands:  
```commandline
git clone https://github.com/cvg/LightGlue.git && cd LightGlue
python -m pip install -e .
```

3. Git clone repository  
Clone the vis_slam_game repository using:
```commandline
https://github.com/Harshit0803/vis_slam_game.git
```
Replace the player.py with the modified player.py provided in this repository.

4. Play using the default keyboard player
```commandline
python player.py
```
5. Exploring the Environment  
Navigate the environment using keyboard movement keys. The areas explored by the user will be visible on the map.

6. Completing the Exploration Phase  
Press the ESC key to complete the exploration phase.

7. Visual Place Recognition  
The query images will be processed using the Visual Place Recognition algorithm, specifically VLAD.

8. Identifying the Likely Location  
The location with the minimum covariance will be identified as the most probable location.

9. Auto-Navigation to Final Destination  
After pressing the ESC key, the AUTO_NAV feature in the code will guide the player to the final destination.

Note: The map displays probable locations with red dots based on the query image match. If the camera continually collides with walls, the auto-navigation feature can be toggled in player.py:
```commandline
self.AUTO_NAV = True  #change this to enable/disable NAV
```

Project Video:  

https://github.com/Harshit0803/vis_slam_game/assets/29999543/c63ec94c-ae37-44ef-bf47-429fdae33eaf

This project was created by Prof. Chen Feng (cfeng at nyu dot edu) of [AI4CE Lab](https://github.com/ai4ce/vis_nav_player/tree/master)


