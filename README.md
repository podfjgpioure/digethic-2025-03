# python-template

Precondition:
Windows users can follow the official microsoft tutorial to install python, git and vscode here:

- ​​https://docs.microsoft.com/en-us/windows/python/beginners
- german: https://docs.microsoft.com/de-de/windows/python/beginners

## Visual Studio Code

This repository is optimized for [Visual Studio Code](https://code.visualstudio.com/) which is a great code editor for many languages like Python and Javascript. The [introduction videos](https://code.visualstudio.com/docs/getstarted/introvideos) explain how to work with VS Code. The [Python tutorial](https://code.visualstudio.com/docs/python/python-tutorial) provides an introduction about common topics like code editing, linting, debugging and testing in Python. There is also a section about [Python virtual environments](https://code.visualstudio.com/docs/python/environments) which you will need in development. There is also a [Data Science](https://code.visualstudio.com/docs/datascience/overview) section showing how to work with Jupyter Notebooks and common Machine Learning libraries.

The `.vscode` directory contains configurations for useful extensions like [GitLens](https://marketplace.visualstudio.com/items?itemName=eamodio.gitlens0) and [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python). When opening the repository, VS Code will open a prompt to install the recommended extensions.

## Development Setup

Open the [integrated terminal](https://code.visualstudio.com/docs/editor/integrated-terminal) and run the setup script for your OS (see below). This will install a [Python virtual environment](https://docs.python.org/3/library/venv.html) with all packages specified in `requirements.txt`.

### Linux and Mac Users

1. run the setup script: `./setup.sh` or `sh setup.sh`
2. activate the python environment: `source .venv/bin/activate`

### Windows Users

1. run the setup script `.\setup.ps1`
2. activate the python environment: `.\.venv\Scripts\Activate.ps1`

Troubleshooting:

- If your system does not allow to run powershell scripts, try to set the execution policy: `Set-ExecutionPolicy RemoteSigned`, see https://www.stanleyulili.com/powershell/solution-to-running-scripts-is-disabled-on-this-system-error-on-powershell/
- If you still cannot run the setup.ps1 script, open it and copy all the commands step by step in your terminal and execute each step

### Executing the Projekt

Note: This was executed on a Windows 10 Computer. Code has minimal focus on beeing platform independent!

1. Download the Dataset and extract into the "data" folder. You should have: "data\household_power_consumption.txt"
2. Execute: `python src\power-01-extract-final.py`
3. You got: "data\household_power_consumption-valid.txt"
4. Execute: `python src\power-02-gan-final.py`
5. You got: "data\generated_data_gan.txt"
6. Execute: `python src\power-02-vae-final.py`
7. You got: "data\generated_data_vae.txt"
8. Execute: `python src\power-03-plots-final.py`
9. You got: 8 Plots of Original Days, 6 Plots for GAN-Generated Data and 6 Plots for VAE-Generated Data