# delayed_highmass_WDs_Gaia
I compare the ages of white dwarfs (WDs) obtained from H--R diagram and from transverse velocities to infer delays in their evolution. I use this information to estimate the fraction of WD-WD merger products among high-mass WDs and the properties of an abnormal cooling delay which produces the Q branch. We use Gaia DR2 data. This work has been published in the paper https://arxiv.org/abs/1905.12710.

This repository contains python files for functions and the MCMC code, and jupyter notebooks for making figures and inspecting data. To run the codes other than MCMC, one need to have numpy, matplotlib, scipy, astropy packages in python 3. To run the MCMC code, one need additional emcee and multiprocessing packages.
This repository also contains data tables with information of the white dwarfs we analysed in the paper. 

This repository is in construction and will be finished soon. If you want more information or have suggestions, please do not hesitate to contact me: s.cheng@jhu.edu

Below is a demonstration of our two-population scenario of the Q branch. The first population of white dwarfs cools normally, and the second population has an extra cooling delay on the Q branch. More animations illustrating this abnormal white-dwarf cooling are in "./gif_animation/" and on my webpage https://pages.jh.edu/~scheng40/.
![the two-population scenario of the Q branch](/gif_animation/gif_green.gif) ![the two-population scenario of the Q branch](/gif_animation/gif_orange.gif)
