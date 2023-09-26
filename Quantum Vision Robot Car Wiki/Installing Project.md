To get this project installed and running follow the steps below. Note that these instructions will be for Linux OS, but still gives the general steps for any OS you may just need to search up how to do it in your OS.

## Step 1: Install Mamba
Let's start by installing mamba which will hold our virtual python environment. A virtual environment is helpful to prevent package collisions with anything else that may be installed on your system.

- Go to [Mamba Forge](https://github.com/conda-forge/miniforge#mambaforge) GitHub page to get the installer for your OS.

- In your downloads folder open a terminal and run "bash MAMBA-INSTALLER", and replace MAMBA-INSTALLER with the name of the file you downloaded.

- #TODO put whatever the installer ask to do.

- If it installed correctly you can restart your terminal and see "(base)" at the start of every line

Now you should have the ability to make virtual python environments.
## Step 2: Create Mamba Environment And Add Packages
With mamba environments we can make isolated versions of python that can have different packages. We will now make a new environment for our project.

- In your terminal enter "mamba create -n NAME IPython matplotlib pylatexenc qiskit qiskit-aer qiskit_algorithms qiskit-ibm-runtime qiskit-machine-learning" where NAME is the name you wish to refer you mamba environment as.

- #TODO add any notes about the package installer

- Once the packages finish installing run "mamba activate NAME" to get into your virtual python environment. You should see (Base) change to (NAME). NOTE if you are on the ASU Sol supercomputer use "source activate NAME" FALURE TO DO SO COULD CASUE YOUR JOBS TO NOT RUN PROPERLY.

Once the packages finish installing you can proceed to getting the project files from GitHub.
## Step 3: Download GitHub Project
Now that the virtual environment s set up let's download the project files.

- Navigate in the terminal to where you want to install the project folder.

- Run "git clone https://github.com/Ruktismo/Capstone-Project-Quantum-Edge-Detection.git"

You should now see a new folder named Capstone-Project-Quantum-Edge-Detection, open it up and see that all the project files are there.
## Step 4: Run Example Files
Now everything should be installed so we can run some example files to make sure that everything is able to run properly.

- Start by cd-ing into the Examples folder of the project.

- Run "python 8x8local.py" to see if it runs to completion with the proper output.

- Then run "python 256x256local.py 16 4" to do the 256 local sim with a chunk size of 16 and 4 threads. If you get a pickling error then one of the threads crashed, go back and run 8x8 and 16x16 to see if they run properly. 

If 256 local runs properly then everything should be good to go!