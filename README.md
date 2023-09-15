# TASK6-VIGNESWARSIDDU
#### COMPANY: KAIBURR

#### QUESTION: Task 6. Data Science example.
Perform a Text Classification on consumer complaint dataset
(https://catalog.data.gov/dataset/consumer-complaint-database) into following categories.
0 Credit reporting, repair, or
other
1 Debt collection
2 Consumer Loan
3 Mortgage
Steps to be followed -
1. Explanatory Data Analysis and Feature Engineering
2. Text Pre-Processing
3. Selection of Multi Classification model
4. Comparison of model performance
5. Model Evaluation
6. Prediction

#### SOLUTION:
##### My Understanding of the Dataset:
This is a supervised text classification issue because each complaint is associated with a distinct product. We applied various machine learning algorithms to create more precise predictions in order to categorize future complaints.

##### My work outlook:
Complaints are the predefined categories,
Algorithms used: Linear Support Vector Machine, Multinomial Naive Bayes.

##### STEP 1: Download Dataset from the source: https://catalog.data.gov/dataset/consumer-complaint-database [and I'm using Google Collaboratory Notebook, and I will access the dataset from the MyDrive I'm uploaded]
##### -> Importing the libraries:
<img width="959" alt="image" src="https://github.com/Vigneswarsiddu/TASK6-VIGNESWARSIDDU/assets/93468524/248b8015-afe8-430c-a9ff-c23df6cc13c3">

### 1. Explanatory Data Analysis and Feature Engineering
##### STEP 2: Loading the Dataset:
##### OBSERVATION: Here there are more than 4 million instances (rows) and 18 features (columns).
<img width="730" alt="image" src="https://github.com/Vigneswarsiddu/TASK6-VIGNESWARSIDDU/assets/93468524/23077b7b-cf2c-4183-88bb-f49ca78a1a68">

##### NOTE: The above dataset contains features that are not necessary for the multi-classification. So I'm going to make another data frame that contains ‘Product’ and ‘Consumer complaint narrative’ in the below code. (which is known as Consumer_complaint)
<img width="960" alt="image" src="https://github.com/Vigneswarsiddu/TASK6-VIGNESWARSIDDU/assets/93468524/1b1693f9-ce6d-4d3d-9f16-a3286456f416">

<img width="767" alt="image" src="https://github.com/Vigneswarsiddu/TASK6-VIGNESWARSIDDU/assets/93468524/a978ca70-b106-4c8c-aa59-3f57df700333">

<img width="960" alt="image" src="https://github.com/Vigneswarsiddu/TASK6-VIGNESWARSIDDU/assets/93468524/3fcae365-3416-4544-acd0-87d194e4a2db">

##### Graph for the First 15 Categorical Distributions
<img width="432" alt="image" src="https://github.com/Vigneswarsiddu/TASK6-VIGNESWARSIDDU/assets/93468524/77540d9d-cd8c-4f2b-9354-8a9573bb6a3c">

<img width="771" alt="image" src="https://github.com/Vigneswarsiddu/TASK6-VIGNESWARSIDDU/assets/93468524/206941af-bda1-402a-b250-8d725418041c">

##### OBSERVATION: From the above dataset more than 4 million complaints, from the above info I get to know that there are about 360,200 cases with text (36.2% of the original dataset is not null). Further, I will now look at the categories that I want to classify each complaint.

<img width="960" alt="image" src="https://github.com/Vigneswarsiddu/TASK6-VIGNESWARSIDDU/assets/93468524/d6be936c-a8df-4eba-ac74-07533ab1509c">

##### OBSERVATION: There are 21 different classes or categories (target). Now I need to make the algorithm that can classify the consumer complaint.
##### STEP 3: Because of the data the computation process is more time-consuming, so now the solution is to sample the data.
<img width="772" alt="image" src="https://github.com/Vigneswarsiddu/TASK6-VIGNESWARSIDDU/assets/93468524/35f40fdd-6d2f-4bbc-8aff-ba53a11e897c">

<img width="768" alt="image" src="https://github.com/Vigneswarsiddu/TASK6-VIGNESWARSIDDU/assets/93468524/7e85b520-0c9c-4305-b201-24b85eacb7f5">

##### OBSERVATION: If you Observe the above Output, I can say that the number of classes was reduced from 21 to 13, Now I need to Represent each category as a Numerical Value.
##### DATA VISUALIZATION:
<img width="960" alt="image" src="https://github.com/Vigneswarsiddu/TASK6-VIGNESWARSIDDU/assets/93468524/c5d5becd-de8a-4280-a799-8db05a2c23df">

##### DISTRIBUTIONS GRAPHS:

#### 1. Values
<img width="278" alt="image" src="https://github.com/Vigneswarsiddu/TASK6-VIGNESWARSIDDU/assets/93468524/f97b39ec-6e18-42e7-a247-3c1bf7ee9a36">

#### 2. Distributions
<img width="295" alt="image" src="https://github.com/Vigneswarsiddu/TASK6-VIGNESWARSIDDU/assets/93468524/6a866785-d369-4bb3-95b8-c5a717d73567">

#### 3. Categorical distributions
<img width="376" alt="image" src="https://github.com/Vigneswarsiddu/TASK6-VIGNESWARSIDDU/assets/93468524/9f5d8440-67de-4647-ad6d-360fef34ae9c">

#### 4. 2D Categorical distributions
<img width="590" alt="image" src="https://github.com/Vigneswarsiddu/TASK6-VIGNESWARSIDDU/assets/93468524/e314f416-a123-4e9b-b8dc-cb141607ddfa">

#### 5. Time Series
<img width="952" alt="image" src="https://github.com/Vigneswarsiddu/TASK6-VIGNESWARSIDDU/assets/93468524/3572d661-5e8e-469c-b148-849119fac27c">

##### OBSERVATION: if you observe from the above graphs, I can say that most customer complaints are due to:
credit reporting, credit repair
debt collection
mortgage



















