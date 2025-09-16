## Medical Charges Prediction using Machine Learning

The project aims at predicting the medical charges based on factors: 
1. Age
2. BMI
3. Region
4. Smoking Habit
5. Number of children
6. Gender

The first step is the EDA performed on the dataset. 
A multivariate analysis is provided below: 

<img width="1093" height="980" alt="image" src="https://github.com/user-attachments/assets/08fb6e2b-fbdd-4831-b0ad-455d4e2c53c4" />

9 models are used to train the data and the best model is chosen based on the highest R2 score:
1. Linear Regression
2. Lasso
3. Ridge
4. K-Neighbors Regressor
5. Decision Tree
6. Random Forest Regressor
7. XGBRegressor
8. CatBoosting Regressor
9. AdaBoost Regressor

Now, lets talk about the deployment.  

There are two deployment files for 2 methods: 
1. Flask Deployment (app.py)
The app is deployed using Flask on the local server. - app.py  
Install python and run the code  
python app.py  
Open browser with http://127.0.0.1:5000/  
Voila, your app is ready  

2. Streamlit Deployment (app_streamlit.py)  
The app is deployed in streamlit by linking the github account.  
Next, add the app_streamlit.py file as the application file.  
Voila, your app is ready to use  
