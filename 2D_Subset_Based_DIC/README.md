# run-dic
Open-source DIC software in Python

The main.ipynb notebook is currently setup to run DIC between two images.
Please load the two images in the folder named 'images' in your current working directory.There are two sample DIC images loaded to this folder at the moment.
Use the naming convention 'img0', 'img1' to ensure they are read in the correct order, or change the source code in the LoadImages() function.

The requirements.txt file contain the packages installed in the virtual environment.
Additional packages must be installed from:

interpolation:
https://github.com/dbstein/fast_interp

regularized differentiation:
https://github.com/rickchartrand/regularized_differentiation
