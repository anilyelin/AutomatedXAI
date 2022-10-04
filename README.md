# AutomatedXAI
This repository contains the Python based implementation of the proposed automated explainability checker framework in my master thesis. 
To run this project on your local computer you need to perform the following steps:

1. Download [Anaconda](https://www.anaconda.com/) which makes it easy to create environments and installing relevant libraries

2. Clone this project to your PC
```bash
git clone https://github.com/anilyelin/AutomatedXAI.git
```

3. Create an environment based on the ```requirements.txt```file. 
```bash
conda create --name <env_name> --file requirements.txt
```

4. Activate the prior created environment

```bash
conda activate <env-name>
```

5. Navigate to the folder main
```bash 
cd AutomatedXAI/src/main/
```

6. Type in the following command to start the Streamlit App

```bash
streamlit run streamlit-test.py
```

## Architecture

<img src="https://github.com/anilyelin/AutomatedXAI/blob/main/src/main/method.png" alt="drawing" width="300"/>

### Screenshots

<img src="https://github.com/anilyelin/AutomatedXAI/blob/main/src/prototype.gif" alt="drawing" width="300"/>