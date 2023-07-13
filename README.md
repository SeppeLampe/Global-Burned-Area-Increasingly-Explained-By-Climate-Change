# Global Burned Area Increasingly Explained By Climate Change

<img align="right" src="https://github.com/SeppeLampe/Global-Burned-Area-Increasingly-Explained-By-Climate-Change/assets/56223069/38ac7a74-f439-4c70-bb12-a4dfa1798a18" width="200" />

Python code accompanying ISIMIP3a burned area attribution. (updated July 2023)

Chantelle Burton & Seppe Lampe

Full author list: Chantelle Burton<sup>1</sup> and Seppe Lampe<sup>1</sup> <sup>2</sup>, Douglas I. Kelley, Wim Thiery, Stijn Hantson, Nikos Christidis, Lukas Gudmundsson, Matthew Forrest, Eleanor Burke, Jinfeng Chang, Huilin Huang, Akihiko Ito, Sian Kou-Giesbrecht, Gitta Lasslop, Wei Li, Lars Nieradzik, Fang Li, Yang Chen, Jim Randerson, Christopher P.O. Reyer & Matthias Mengel

<a name="contribution"><sup>1</sup></a> Equal contribution

<a name="contact"><sup>2</sup></a> Corresponding authour: seppe.lampe@vub.be

Article under review.

## Code availability 

This repository contains [5 notebooks](https://github.com/SeppeLampe/Global-Burned-Area-Increasingly-Explained-By-Climate-Change/tree/5feeb0b8eb55d232c65a01a4d8803d4cb00d705d/Scripts) that contain the entire workflow of our analysis.

__Important message!__<br>
Currently, not all the simulations used in this analysis are openly available yet. All of them will be openly available in the coming months.
We make use of two observational burned area datasets, [FireCCI5.1](https://dx.doi.org/10.5285/58f00d8814064b79a0c49662ad3af537) and [GFED5](https://doi.org/10.5194/essd-2023-182).
We also use simulations by seven models of the ISIMIP framework. 
However, from these seven models, two are currently available from the [ISIMIP data repository](https://data.isimip.org/). The other five will become available during the coming weeks/months.

### How to run this code

1. Navigate to a desired directory and clone this repository (in terminal)
   ```
   git clone https://github.com/SeppeLampe/Global-Burned-Area-Increasingly-Explained-By-Climate-Change.git
   ```
2. Download the data in the corresponding folders

   Download the [FireCCI5.1](https://data.ceda.ac.uk/neodc/esacci/fire/data/burned_area/MODIS/grid/v5.1) filestructure to [Data\Observations\FireCCI5.1](Data\Observations\FireCCI5.1).<br>
   Download the [GFED5]([https://doi.org/10.5194/essd-2023-182](https://doi.org/10.5281/zenodo.7668424)) filestructure to [Data\Observations\GFED5](Data\Observations\GFED5).<br>
   Download the ISIMIP Fire Sector OutputData to [Data\ISIMIP\OutputData\fire](Data\ISIMIP\OutputData\fire) (__Not fully available yet__).<br>

4. Create a new conda environment
   ```
   conda env create --file=environment.yml
   ```
5. Launch Jupyter Lab
   ```
   jupyter lab
   ```
