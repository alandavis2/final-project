# Project Setup

**Note** : This project works without any breaking errors in `Python 3.11`. Other versions are not tested.

## Prerequisites

### 1. Install Visual Studio 2022 for Windows

Visual Studio can be downloaded from [here](https://visualstudio.microsoft.com/downloads/). Follow the onscreen instructions and install

1. `Desktop development with C++`
2. `.Net desktop build tools`

### 2. Optional: Only if Nvidia graphics is available on your computer

Install [Cuda-toolkit](https://developer.nvidia.com/cuda-downloads) for Windows.

**Note**: For this project, cuda toolkit version `12.4` is used. For other versions of cuda, please make sure you are also installing the appropriate versions of `torch` rather than installing it using `requirements.txt`. The required steps are also mentioned below.

## Starting Project

### 1. Create Virtual Environment

Create a new virtual environment in your project directory:

```bash
python -m venv venv
```

### 2. Activate Virtual Environment

```bash
venv\Scripts\activate.bat
```

### 3. Install Required Packages

Once the virtual environment is activated:

- If a `requirements.txt` file is present and **ONLY IF** you are using `same version of cuda toolkit`, run:

```bash
pip install -r requirements.txt
```

- Else, manually install the necessary packages and generate the file:

**Note**: Appropriate pip commands for pytorch packages for CUDA can be found [here](https://pytorch.org/get-started/locally/).
**Note**: If any errors are thrown during installation, try pip installing each package seperately.

With CUDA cores (for systems with Nvidia cards):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers[torch] datasets accelerate scipy scikit-learn pandas numpy matplotlib seaborn flask ipykernel jupyter
pip freeze > requirements.txt
```

Without CUDA cores (for systems without Nvidia)

```bash
pip install torch transformers[torch] datasets accelerate scipy scikit-learn pandas numpy matplotlib seaborn flask ipykernel jupyter
pip freeze > requirements.txt
```

## VS Code Configuration

### Setting Up Jupyter Notebook

1. Register the new kernel in Jupyter:

```bash
python -m ipykernel install --user --name=venv
```

2. Now restart Jupyter Notebook in vscode, select the new kernel (venv) from the kernel menu

3. Configure the kernel:
   - Look for the kernel selector in the top right corner
   - Click "Select Another Kernel"
   - Choose "Python Environments"
   - Select your virtual environment (labeled as "venv")

## Notes

- Always ensure your virtual environment is activated before installing new packages
- The virtual environment name ("venv") will appear in your terminal prompt when activated
- To test CUDA Availability, run this Python script in the active virtual environment:

```bash
python test_gpu.py
```

- Use command `deactivate` to exit the virtual environment when needed

**Additional info: Not Related to Project** : To delete all globally installed pip packages

```bash
pip list
pip freeze > requirements.txt
pip uninstall -y -r requirements.txt
del requirements.txt
pip cache purge
```
