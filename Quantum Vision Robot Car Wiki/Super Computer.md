For this project we used the ASU Sol super computer to help with the more computationally intensive task, such as the training the [[Neural Network]] or batch processing many images through the [[QED]] to make the training set. In this page we will go over how to set up with the super computer and run jobs.

Note: If you don't have access to the ASU Sol super computer then this page is useless.

Start by following [this tutorial](https://asurc.atlassian.net/wiki/spaces/RC/pages/1642692609/New+User+Guide+for+Sol+Compute+Resources) to get connected to the super computer.

# Performance

100 active cores 100 Gb RAM -> ~1photo/min
	for 12hr run got ~800 photos
16 active cores 16? Gb RAM -> ~0.5photo/min
	for 16hr run got 

Can get more performance if you do 3 threads, one for each label. and let each thread use 20 cores. So job would need 3 * 20 = 60 cores
