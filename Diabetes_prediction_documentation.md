Diabetes Detection Analysis using Deep Learning

•	Imported all the required libraries.

![image](https://github.com/user-attachments/assets/305acb9f-2e07-46dc-8f37-6c884079c9dd)

 
•	Connecting with the google drive to access the csv file.

 ![image](https://github.com/user-attachments/assets/b6ebaccb-d0dc-4648-bedc-9a6299b8d219)

•	Reading the csv file using “pd.read_csv” command.

 ![image](https://github.com/user-attachments/assets/1e65daef-7cbd-44fa-a20f-79935802b203)

•	Dimensions of the dataset

 ![image](https://github.com/user-attachments/assets/965e3efb-bc1e-4076-868d-bdf5b3cffb41)

•	Printing first five rows of the dataset using df.head() so as to know the columns and row information contained in the dataset.

 ![image](https://github.com/user-attachments/assets/8809bbbb-e1a0-4dd0-b5fb-f000c6cbfb71)

•	Printing the information about the dataset in order to know whether any column contains null values or not and verify the data type of each column.

 ![image](https://github.com/user-attachments/assets/4176fd68-2f50-4050-bf18-417df8249c64)

•	After confirming the data type and non null status of each column, starting the Exploratory Data Analysis and Preprocessing, whereby preprocessing will deal with cleaning, manipulating the data in order to make it ready for the analysis, in EDA, the analysis of data is being conducted, consisting of univariate analysis, bivariate analysis and correlation analysis in order to conduct better feature engineering or feature selection and control the multicollinearity if it exists.
•	In preprocessing, firstly checked for the NaN values if they existed.

 ![image](https://github.com/user-attachments/assets/0178760f-087b-4c89-914b-51807878a4bb)

•	Determined the percentage distribution and absolute distribution of the classes i.e. output classes in the dataset .

 ![image](https://github.com/user-attachments/assets/68b78e79-5532-4298-b1dd-9ff5695e3f0a)

•	Mapping the output classes with the integer values, Negative:0 and positive:1.

 ![image](https://github.com/user-attachments/assets/f3f1116b-a063-4198-82f7-e0e86daed61b)

•	Determined the distribution of Genders: Male and Female in the dataset and mapped their values with integers values, Male:0 and Female:1

 ![image](https://github.com/user-attachments/assets/dd9abe95-62ab-4fcc-953a-4a63ce81cd7d)

•	Determined the value counts of different categories in different columns in order to make out if the distribution of the categories is skewed or not.

 ![image](https://github.com/user-attachments/assets/0bf37674-4b3d-423d-a225-95f49b39f747)

•	Dataset consisted of almost 14 columns which were categorized in Yes and No. Therefore, after finding out the columns that contained those categories, with the help of loop, mapped the Yes with 1 and No with 0. And then used df.head() to see the changes in dataset. 

 ![image](https://github.com/user-attachments/assets/38e6e3ed-01da-4499-8197-ad1146540f62)

•	Called df.info() in order to cross check if the data type of columns are as required.
 
![image](https://github.com/user-attachments/assets/2bc931fd-8453-45db-a39d-0dbcee0cda91)

•	Retrieved Statistical summary of the dataset after columns conversion in order to get better analysis at their mean and median values.

 ![image](https://github.com/user-attachments/assets/884eea01-95d8-4abb-ac54-0b59ec182876)

•	After completing with all the required steps in preprocessing, we have now reached with a dataset which is clean, doesn’t contain any null values and is transformed into integer values for all the columns. Now, we will start with EDA process in order to do better feature engineering and notice the relationship between the different variables.
•	Started with univariate analysis. Here, we will incorporate one feature at a time and analyse them in isolation.

 ![image](https://github.com/user-attachments/assets/be97ed73-ca70-4a7b-8eb9-e70d551ab5f6)

•	Created a distplot for age and we can see the age distribution is highly scattered but the spread is mostly around mean and median value. 

 ![image](https://github.com/user-attachments/assets/1db681e6-4482-4288-a627-f309785b2e2c)

•	Distplot for gender and it showed that Males(0) are more than females(1) in the dataset.

 ![image](https://github.com/user-attachments/assets/954fa3b8-8825-482f-9b0e-53acec912af0)

•	Distribution of classes countplot and it is found that class 1: Positive had more instances in the dataset which means more people have diabetes than the ones who doesn’t.

 ![image](https://github.com/user-attachments/assets/60c34c33-8685-4a7e-90f2-bd8fd08d3baf)

•	Defined a function count_xplot to create several count plots for different columns.

 ![image](https://github.com/user-attachments/assets/b726e347-8793-4d98-b248-6cf2216ce736)

•	INSIGHTS DRAWN:
	It is found that equal number of people from class 0 and class 1 suffer from polyuria. It denotes that polyuria isn’t something that can be seen in diabetic patients only.
	The second observation was that people who belonged to class 0 i.e. people who doesn’t have diabetes, their number was far more when suffering from polydipsia as compared to class 1.

 ![image](https://github.com/user-attachments/assets/68bdd204-f94b-443e-9d72-d9c9247608a6)

	  Sudden weight loss is found more in the people not having diabetes than those having diabetes.
	Talking about weakness, it is observed that number of diabetic people having weakness are more than number of non diabetic people showing symptoms of weakness.

 ![image](https://github.com/user-attachments/assets/40e885b1-5091-4e2f-b65e-164d899b8587)

	The number of people who have polyphagia and genital thrush are more for the instances that doesn’t have diabetes while the number of diabetic patients having polyphagia and genital thrush were comparatively less.

 ![image](https://github.com/user-attachments/assets/58c145df-cea7-4073-a9c9-ad642f46cbe2)

•	Conducted Bivariate Analysis.
•	INSIGHTS:
	Drawing up  a strip plot for age vs class showed that the scatter for class 1 is significantly more crowded than the class 0. Moreover, we observe that the people start to have diabetes below 30 years and it is still shown in the people aged above almost 75 years. 

 ![image](https://github.com/user-attachments/assets/48ae08f5-0b09-487c-92f2-0dafab919439)

	Drawing up a line plot for age vs class showed that there is a linear relationship between the classes and the age factor. As we go up the age scale, there is a high chance that a particular person will belong to class 1.

 ![image](https://github.com/user-attachments/assets/1c0a600d-9cf2-4e6c-92f5-1fdbac29a962)

	Detected outlier through box plot and it is found that class 1 had few outliers but since they were meagre,  their impact on the analysis will be less, hence they are neglected in the dataset.

 ![image](https://github.com/user-attachments/assets/455e4066-aec8-4abc-b1e2-86e833e1a61d)

•	Correlation analysis using heatmap.

 ![image](https://github.com/user-attachments/assets/814bdda7-372d-4a88-a9c4-ea7f6940ac7a)

•	Multi collinearity Analysis: Conducted this analysis in order to examine if any strong correlation exists between the columns. For this, imported variance inflation factor and calculated VIF for all the columns. For all the columns, we saw that the values stood less than 15, hence it means no significant correlation exists between the columns and whatever meagre correlation do exist within the VIF range of 5-15, it can be resorted through standardization later on.

 ![image](https://github.com/user-attachments/assets/6bb2b757-bf65-4b4e-8928-859674f1535b)


 ![image](https://github.com/user-attachments/assets/b2376506-996b-472f-aa0c-90610a972655)


•	Separated the dataset into input and output and printed the dimensions of both.

 ![image](https://github.com/user-attachments/assets/06e73f49-4537-462b-a5dc-bdbf14d64fd9)

•	Split the dataset for training and testing purposes.

 ![image](https://github.com/user-attachments/assets/7de8d903-7c55-47f9-abd9-67b804093d33)

•	Scaled the training and testing dataset in order to ensure that all the columns comes down to within the same range from -1 to 1 using standard scaler.

 ![image](https://github.com/user-attachments/assets/8a310c12-401f-41c6-862f-5cb447974004)

•	Created DEEP LEARNING MODEL: Incorporated three hidden layers with each layer having 10 nodes with activation function as relu. Installed a drop out layer at the last hidden layer so as to bring randomness in the model and discouraging the model from learning by overly attaching itself from certain nodes or giving higher attention/weightage to some nodes while neglecting the other. And lastly, gave one last output layer and used activation as sigmoid because this is a binary class classification problem.

 ![image](https://github.com/user-attachments/assets/02b43248-56bf-4e4c-af08-a340f987fdec)

•	Printing the model summary and we find that there are 401 trainable parameters in our model.

 ![image](https://github.com/user-attachments/assets/eee074c0-f34a-43e9-bbab-e2119c7576ef)

•	Compiled the model, using loss function as binary cross entropy since it is a binary class problem, optimizer as Adam and metrics as accuracy.

 ![image](https://github.com/user-attachments/assets/1b9addc8-83e2-4900-a689-3bed69d38c55)

•	Called Early Stopping from Keras library in order to ensure the model terminates the loop the moment it detects that model has started to overfit.

 ![image](https://github.com/user-attachments/assets/3582f2f4-8240-4168-b2e9-de4c5d3a5f35)

•	Created an object called history and within that object, stored the training of the model that will run for 100 epochs and terminate the moment it detects overfitting.

 ![image](https://github.com/user-attachments/assets/c4535d63-72e6-4593-ba11-77bc2d486099)


 ![image](https://github.com/user-attachments/assets/1cad576f-3fa2-4edd-867e-aa385185ce32)


 ![image](https://github.com/user-attachments/assets/fc61ee81-80b7-414b-8fbc-d0d6c6a78d44)


 ![image](https://github.com/user-attachments/assets/3b6d2f04-5b7f-48d2-bb8c-73a6eb6b36d7)


 ![image](https://github.com/user-attachments/assets/4c2537e5-fc42-404e-88c8-abf3d4540d2c)


 ![image](https://github.com/user-attachments/assets/fb0cab2d-33f6-4a1e-ae32-650b22156424)

At the 99th epoch, it showed early stopping and terminated the loop indicating that if the model is ran for epochs more than 99, it will start to overfit.
•	Plot Actual loss v/s Validation loss in the model:
	Insight Drawn is that both actual loss and val loss showed decreased as the number of epochs increased. However, actual loss graph showed more instability as it moved down in comparison to the val loss which was rather smooth which showed smooth decrease as the number of epochs increased. Also, towards the end, we can see that validation loss was quite lower than actual loss of the model.


 ![image](https://github.com/user-attachments/assets/9a76ab94-40c8-454d-ac24-63e9d908fddd)

•	Plot actual accuracy v/s validation accuracy.
	Insight drawn is that both actual and validation accuracy increased as the number of epochs increased. However, val accuracy increased sharply and rapidly than actual accuracy and achieved quite more accuracy percentage than actual accuracy towards the last epoch.


 ![image](https://github.com/user-attachments/assets/74447792-9bae-47c5-bb44-20b6d958b1ba)

•	Make the model predict the value of y

 ![image](https://github.com/user-attachments/assets/27d18ab7-2b4c-4479-bca7-6103bce4d2b2)

•	Created y_pred object to store all the predictions of the model and converted them in the form of 0 or 1.
•	Calculated the accuracy of the model using test_y and y_pred and found that model showed 99.03% accuracy.

 ![image](https://github.com/user-attachments/assets/e8fba128-8a58-4bc3-9767-7659e99a30ba)

•	Finally made the model predict the value of y from the dataset values to check if the predicted value by model is equal to actual value of the dataset.
 

![image](https://github.com/user-attachments/assets/866572f9-f512-48e4-b5ed-c04140f6bec3)


