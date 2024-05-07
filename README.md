<div align=center>
<img src="/fig/abstract.png" width="300px" align="float:center" />
</div>

# DeepMIR
There are some codes for the paper DeepMIR: a versatile and accurate component identification method for high-throughput mid-infrared spectra. We developed a method named Deep learning-based library search method for Mid-infrared spectroscopy (DeepMIR). Other methods for comparison in the paper are also presented. Organic solvents datasets are public for example applications.
# Requirements
Before running codes on your own computer, make sure you have established an environment required by this project.
## Create a Conda environment
    conda env create -f requirements.yml -n myenv
## Activate the Conda environment
    activate myenv
# Example
You can see the [example.ipynb](https://github.com/LinTan-CSU/DeepMIR/blob/main/src/example.ipynb) from the "src" folder for the utilization of DeepMIR. If you want to try the codes on your own computer, please download the whole project and the Organic Solvents dataset from the Releases. Try other sub-datasets based on the example.ipynb. 
# Download datasets
If you want to download the datasets automatically from the Releases, [wget](https://eternallybored.org/misc/wget/) is recommended.
## Use wget
    wget https://github.com/LinTan-CSU/DeepMIR/releases/tag/OrganicSolventsDataset/Binary.npy -P ../data/
