# AutomatedXAI
This repository contains the Python based implementation of the proposed automated explainability checker framework in my master thesis. 
To run this project on your local computer you need to perform the following steps:

1. Download Anaconda which makes it easy to create environments and installing relevant libraries

2. Clone this project to your PC
    ```git clone https://github.com/anilyelin/AutomatedXAI.git```

3. Create an environment based on the ```requirements.txt```file. 
```conda create --name <env_name> --file requirements.txt```

4. Activate the prior created environment

```conda activate <env-name>```

5. Navigate to the folder main
```bash 
cd AutomatedXAI/src/main/
```

6. Type in the following command to start the Streamlit App

```streamlit run streamlit-test.py```