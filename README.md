# Improving Southwest Airlines Customer Service Operations with Text Analytics and Classification
## Background

As a Business Analytics student, I have always been interested in using data to solve real-world problems, and my background in crisis management drew me to this customer service project for Southwest Airlines. Through this project, I have gained valuable experience in data collection, preprocessing, and machine learning techniques. I am excited to share my work with others.

## Problem Statement:
Southwest Airline received a large volume of customer inquiries on Twitter when their system was broken down in December 2022. This has led to a poor customer experience and loss of business. The focus of this project is to build a classification model that can automatically classify customer inquiries on Twitter, which will help to improve the efficiency of Southwest Airline's customer service operations. 
For example, if a customer is asking for refund on twitter to Southwest Airline's, Southwest Airlines’ Social Media team would manually read the tweet and replied to navigate the customer to send DM. Our goal is to automate this process and send the case to a dedicated customer service team that is specialised in refund policy, which improves the efficieny of the escalation process.

## Dataset & Text Preprocessing
As part of this project, I scraped a dataset of 2500 customer service tweets directed at Southwest Airlines using the Twitter API. I then labeled each tweet based on the inquiry topic, such as refunds, flight changes, and no response needed, and created a pandas dataframe based on the labelled dataset. I then performed text preprocessing on the "text" column, which involved tokenizing, stemming, and removing stop words.

## Model Building
Using this preprocessed dataset, I built several machine learning models to classify customer service tickets based on their inquiry topic. I experimented with Decision Trees, Random Forests, Support Vector Machines (SVMs), Naive Bayes Classifier, K-Nearest Neighbors (KNN), and Neural Networks, and evaluated the performance of each model using accuracy score. 

Of all the models I trained, the Support Vector Machine (SVM) classifier performed the best, achieving an accuracy score of 0.858. This means that the SVM correctly predicted the inquiry topic of 85.8% of the customer service tickets in the test set. I used the "linear" kernel with regularization strength (C) equal to 1 and "gamma" set to "scale" for SVM.

Through this project, I learned how to use popular machine learning libraries in Python, such as scikit-learn and pandas, and gained valuable experience in text preprocessing techniques and model selection. 

Overall, this project has been a great learning experience for me, and I am excited to continue exploring the field of text analytics and machine learning.  I welcome any feedback or suggestions for improvement.






