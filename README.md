# AptosApp
An end to end flow from training to locally exposing API POST call for a Deep Learning Model , which predicts Diabetic Retinopathy on Retina images.



To run - Activate the conda environment to be used for the project.


1. Download the dataset from https://www.kaggle.com/c/aptos2019-blindness-detection/data and extract into the working directory.

2. pip install backend/requirements.txt

3. python create_split.py

4. python model.py

5. python train.py ( Training takes about 13 minutes per epoch on a Colab instance with a P100 GPU)

Model weights after training will be saved in 'efficientnet_baseline.pth' file in the working directory.
To run the web ap using streamlit - 

1. Run python backend/main.py in one terminal.

2. Run streamlit run frontend/main.py in a second terminal.

3. Access the streamlit webpage using the link in the terminal.

