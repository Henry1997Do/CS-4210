#-------------------------------------------------------------------------
# AUTHOR: Henry Do
# FILENAME: naive_bayes.py
# SPECIFICATION: Read file and output the classification of each of the 10 instances from the file weather_test (test set) if the classification confidence is >= 0.75
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1.5 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

#Reading the training data in a csv file
#--> add your Python code here
X = []  # Feature matrix
Y = []  # Class labels

with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)

    # Define categorical mappings
    outlook_map = {"Sunny": 1, "Overcast": 2, "Rain": 3}
    temperature_map = {"Hot": 1, "Mild": 2, "Cool": 3}
    humidity_map = {"High": 1, "Normal": 2}
    wind_map = {"Weak": 1, "Strong": 2}
    class_map = {"Yes": 1, "No": 2}

    for row in reader:
        outlook = outlook_map[row[1]]
        temperature = temperature_map[row[2]]
        humidity = humidity_map[row[3]]
        wind = wind_map[row[4]]
        class_label = class_map[row[5]]
        
        #Transform the original training features to numbers and add them to the 4D array X.
        #For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
        #--> add your Python code here
        X.append([outlook, temperature, humidity, wind])

        #Transform the original training classes to numbers and add them to the vector Y.
        #For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
        #--> add your Python code here
        Y.append(class_label)

#Fitting the naive bayes to the data
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

#Reading the test data in a csv file
#--> add your Python code here
test_data = []
test_days = []  # Store Day column for output
original_test_data = []  # Store the full test row for output

with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header row
   

    for row in reader:
        if len(row) < 5:  # Skip empty or incomplete rows
            continue
        test_days.append(row[0])  # Store the day ID
        original_test_data.append(row)
        test_data.append([
            outlook_map[row[1]],
            temperature_map[row[2]],
            humidity_map[row[3]],
            wind_map[row[4]]
        ])

#Printing the header os the solution
#--> add your Python code here
print("Day      Outlook   Temperature   Humidity   Wind   PlayTennis   Confidence")

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
predictions = clf.predict(test_data)
probabilities = clf.predict_proba(test_data)

# Printing results only if confidence ≥ 0.75
class_reverse_map = {1: "Yes", 2: "No"}  # Reverse mapping for output labels
for i in range(len(test_data)):
    confidence = max(probabilities[i])  # Get the highest confidence probability
    row = original_test_data[i]  # Get the full row for proper formatting
    if confidence >= 0.75:
        print(f"{row[0]:<9} {row[1]:<10} {row[2]:<12} {row[3]:<10} {row[4]:<10} {class_reverse_map[predictions[i]]:<12} {confidence:.2f}")