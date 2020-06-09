# 2D and 3D Heat Flow Simulation
This project will be simulating the diffusion of heat in two or three dimensions using CUDA. 
This project will be using simplified heat diffusion equations. For 2 dimensions the assumption is that there is a rectangular room that is
divided into a grid. Inside the grid will be ”heaters” with various fixed temperatures.


Given the rectangular room and configuration of heaters, the code simulates what happens to the temperature in every grid cell as time progresses. Cells with heaters in them always remain constant for simplicity. At each time step, assuming heat flows between a cell and its neighbors. If a cell’s neighbor is warmer than it is, it will tend to heat up. Conversely, if a cell’s neighbor is colder than it is, it will tend to cool down. The simplified equation to model this behavior:

<a href="https://www.codecogs.com/eqnedit.php?latex=T_{new}=T_{old}&plus;\sum\limits_{neighbors}&space;k(T_{neighbor}-T_{old})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T_{new}=T_{old}&plus;\sum\limits_{neighbors}&space;k(T_{neighbor}-T_{old})" title="T_{new}=T_{old}+\sum\limits_{neighbors} k(T_{neighbor}-T_{old})" /></a>

The demo of config file is provided in the .conf file.
