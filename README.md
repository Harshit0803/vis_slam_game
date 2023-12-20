# vis_slam_game

This is the solution for the course project for NYU ROB-GY 6203 Robot Perception. 

# Installation steps for the environment
1. Setup the environment
```commandline
conda update conda
git clone https://github.com/ai4ce/vis_nav_player.git
cd vis_nav_player
conda env create -f environment.yaml
conda activate game
```

2. Installing LightGlue  
To install LightGlue you can either follow the official documentation which can be found [here](https://github.com/cvg/LightGlue) or  
```commandline
git clone https://github.com/cvg/LightGlue.git && cd LightGlue
python -m pip install -e .
```

3. Git clone repository
```commandline
https://github.com/Harshit0803/vis_slam_game.git
```
Replace the player.py with our player.py

4. Play using the default keyboard player
```commandline
python player.py
```
5. Explore the environment using the move keys on the keyboard. The part of the map explored by the user is visible on the map.
6. Once the exploration phase is complete, press the ESC key on the keyboard.
7. The query images will be fed into the Visual Place Recognition algorithm which in our case is VLAD.
8. The place with the minimum covariance is the place where the likely location is.
9. Once the ESC key is pressed, the AUTO_NAV functionality in the code will lead you to the final destination.

The map using the red dots points out the likely location of the places matching the query image.  
If the camera continuously collides with the walls, you can turn off the auto navigation feature in player.py
```commandline
self.AUTO_NAV = True  #change this to enable/disable NAV
```

This project was created by Prof. Cheng Feng (cfeng at nyu dot edu) of [AI4CE Lab](https://github.com/ai4ce/vis_nav_player/tree/master)


