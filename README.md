# LES-emulator-02postpros

This repository holds scripts to plot figures in Ahola et al 2021.

## Instructions for use

### Prerequisities

Create parameterisations according to [LES-emulator-02postpros](https://github.com/JaakkoAhola/LES-emulator-02postpros).

### Configuration files

- Download configuration files from [LES-emulator-04configFiles](https://github.com/JaakkoAhola/LES-emulator-04configFiles).
	- locationsMounted.yaml
	- phase02.yaml

- Modify locationsMounted.yaml according to the instructions at [LES-emulator-04configFiles](https://github.com/JaakkoAhola/LES-emulator-04configFiles)

- Use condaEnvironment.yaml as a base for python conda environment.

### environment variables

Set environment variable LESMAINSCRIPTS to point to the location of the library [LES-03plotting](https://github.com/JaakkoAhola/LES-03plotting).


### running [plotEMUL.py](plotEMUL.py)

run [plotEMUL.py](plotEMUL.py) with a command  
`python plotEMUL.py locationsMounted.yaml`

This script will plot figures 2,4,5,6.

### [profileExample.py](profileExample.py)

This script plots the temperature and moisture profiles of SALSA nighttime simulation number 40, i.e. figure nro 3.


### Author

Jaakko Ahola
