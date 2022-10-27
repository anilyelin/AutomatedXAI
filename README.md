# AutomatedXAI
This repository contains the Python based implementation of the proposed automated explainability checker framework in my master thesis. 
You can either run this project on your local PC or you can easily access it via Streamlit Cloud. The respective links are below.
## Streamlit Cloud

If you don't want to install the project on your PC, you can access it via your browser. 

1. The main version (Parkinson Dataset)
```
https://automated-xai.streamlitapp.com/
```

2. The custom version (use your own datasets)
```
https://automated-xai-custom.streamlitapp.com
```

## Local Installation

1. Download [Anaconda](https://www.anaconda.com/) which makes it easy to create environments and installing relevant libraries

2. Clone this project to your PC
```bash
git clone https://github.com/anilyelin/AutomatedXAI.git
```

3. If you use Mac. Create an environment based on the ```environment_mac.yml```file. 
```bash
conda create --name <env_name> --file environment_mac.yml
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
streamlit run streamlit-main.py
```

7. If you want to try out a custom dataset you have to run the following Streamlit App
```bash
streamlit run streamlit-custom.py
```


## Architecture

<img src="https://github.com/anilyelin/AutomatedXAI/blob/main/src/main/method.png" alt="drawing" width="300"/>

### Screenshots

<img src="https://github.com/anilyelin/AutomatedXAI/blob/main/src/prototype.gif" alt="drawing" width="600"/>