#-------------------------------------------------------------------------
# AUTHOR: Henry Do
# FILENAME: knn.py
# SPECIFICATION: Read the file email_classification.csv and compute the LOO-CV error rate for a 1NN classifier on the spam/ham classification task
# FOR: CS 4210- Assignment #2
# TIME SPENT: 30 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#Reading the data in a csv file
with open('email_classification.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

# Variable to store misclassifications
misclassified_count = 0
total_samples = len(db)

#Loop your data to allow each instance to be your test set
for i in db:

    #Add the training features to the 2D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 3], [2, 1,], ...]].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    # Prepare the training set (excluding the i-th sample)
    X = []  # Feature matrix
    Y = []  # Class labels
    for j in range(total_samples):
        if db[j] != i:  # Exclude the i-th sample
            # Convert feature values to float
            X.append([float(value) for value in db[j][:-1]])
    #Transform the original training classes to numbers and add them to the vector Y.
    #Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
            Y.append(1 if db[j][-1] == "spam" else 0)


    #Store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    testSample = [float(value) for value in i[:-1]]
    true_label = 1 if i[-1] == "spam" else 0

    #Fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2]])[0]
    #--> add your Python code here
    class_predicted = clf.predict([testSample])[0]
    
    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    if class_predicted != true_label:
        misclassified_count += 1
#Print the error rate
#--> add your Python code here
error_rate = misclassified_count / total_samples
print(f"Error rate: {error_rate:.2f}")






