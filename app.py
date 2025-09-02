#=================flask code starts here
from flask import Flask, render_template, request, redirect, url_for, session,send_from_directory
import os
from werkzeug.utils import secure_filename
from fileinput import filename
import smtplib 
from email.message import EmailMessage
from datetime import datetime
from werkzeug.utils import secure_filename
import sqlite3
import pickle
import sqlite3
import random

#importing all required python libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt #use to visualize dataset values
import seaborn as sns
from math import sqrt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.cluster import DBSCAN#loading DBSCAN clustering
from sklearn.neighbors import KernelDensity #loading kernel density algorithms
from sklearn.decomposition import PCA #pca for dimension reduction
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error

UPLOAD_FOLDER = os.path.join('static', 'uploads') 
# Define allowed files
ALLOWED_EXTENSIONS = {'csv'}


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'welcome'

#class to normalize dataset values
scaler = MinMaxScaler(feature_range = (0, 1))
scaler1 = MinMaxScaler(feature_range = (0, 1))

#loading and displaying cellular lte dataset
dataset = pd.read_csv("Dataset/SRFG-v1.csv", nrows=10000)

#dataset peprocessing converting datetime to numeric values and non-numeric values to numeric values
dataset['time'] = pd.to_datetime(dataset['time'])
dataset['year'] = dataset['time'].dt.year
dataset['month'] = dataset['time'].dt.month
dataset['day'] = dataset['time'].dt.day
dataset['hour'] = dataset['time'].dt.hour
dataset['minute'] = dataset['time'].dt.minute
dataset['second'] = dataset['time'].dt.second
label_encoder = []
columns = dataset.columns
types = dataset.dtypes.values
for i in range(len(types)):
    name = types[i]
    if name == 'object': #finding column with object type
        le = LabelEncoder()
        dataset[columns[i]] = pd.Series(le.fit_transform(dataset[columns[i]].astype(str)))#encode all str columns to numeric
        label_encoder.append([columns[i], le])
#handling and removing missing values        
dataset.fillna(0, inplace = True)
print("Cleaned Dataset Values")

Y = dataset['netmode'].ravel()
Y1 = dataset['datarate'].ravel()
dataset.drop(['time','netmode', 'datarate'], axis = 1,inplace=True)#drop ir-relevant columns
X = dataset.values
X = scaler.fit_transform(X)#normalizing dataset values using minmax scaling
selector = SelectKBest(chi2, k=20)#selecting top 20 features using Select k BEST
X = selector.fit_transform(X, Y)
#applying PCA for dimension reduction
pca = PCA(n_components=15)
X = pca.fit_transform(X)
print("PCA Selected features = "+str(X))

#applying DBSCAN based clustering with kernel density to select cluster with most similarity
dbscan = DBSCAN(eps=0.9, min_samples=8)#generating DBSCAN clustering
labels = dbscan.fit_predict(X)
# Kernel Density Estimation for each cluster
unique_labels = np.unique(labels)
choosen_cluster = []
choosen_labels = []
for label in unique_labels:
    if label == -1:  # Noise points
        continue
    cluster_data = X[labels == label]
    YY = Y1[labels == label]
    kde = KernelDensity(bandwidth=0.5).fit(cluster_data)#applying kernel density
    density = kde.score_samples(cluster_data)
    choosen_cluster.append(cluster_data)
    choosen_labels.append(YY)
similarity = 100000
selected = -1
for i in range(0, len(choosen_cluster)):
    cluster1 = choosen_cluster[i]
    for j in range(0, len(choosen_cluster)):
        if i != j:
            cluster2 = choosen_cluster[j]
            sim = cosine_similarity(cluster1, cluster2)#measuring similarity between clusters
            if np.mean(sim) < similarity:
                similarity = np.mean(sim)
                selected = i
X = choosen_cluster[selected]    
Y = choosen_labels[selected]
print("Number of selected Clusters = "+str(len(choosen_cluster)))
print("Cluster values with most similarity = "+str(X))

Y = Y.reshape(-1, 1)
Y = scaler1.fit_transform(Y)
#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
print("Train & Test Dataset Split")

#training extension XGBOOST algorithm
from xgboost import XGBRegressor
xgboost = XGBRegressor(n_estimators=200)
xgboost.fit(X_train, y_train.ravel())
#perform prediction on test datat
predict = xgboost.predict(X_test)


@app.route('/home', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/notebook')
def notebook():
    return render_template('CellularTrafficPrediction.html')


@app.route('/PredictAction', methods=['GET', 'POST'])
def PredictAction():
    if request.method == 'POST':
        f = request.files.get('file')
        data_filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],data_filename))
        session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'],data_filename)
        data_file_path = session.get('uploaded_data_file_path', None)
        test_data = pd.read_csv(data_file_path,encoding='unicode_escape')
        temp = test_data.values
        test_data['time'] = pd.to_datetime(test_data['time'])#convert datetime to numeric format
        test_data['year'] = test_data['time'].dt.year
        test_data['month'] = test_data['time'].dt.month
        test_data['day'] = test_data['time'].dt.day
        test_data['hour'] = test_data['time'].dt.hour
        test_data['minute'] = test_data['time'].dt.minute
        test_data['second'] = test_data['time'].dt.second
        for i in range(len(label_encoder)):
            le = label_encoder[i]
            test_data[le[0]] = pd.Series(le[1].fit_transform(test_data[le[0]].astype(str)))#encode all str columns to numeric
        #handling and removing missing values        
        test_data.fillna(0, inplace = True)
        test_data.drop(['time','netmode', 'datarate'], axis = 1,inplace=True)#drop ir-relevant columns
        test_data = test_data.values
        test_data = selector.transform(test_data)#select features using select k-best
        test_data = pca.transform(test_data)#dimension reduction using PCA
        predict = xgboost.predict(test_data)#perform prediction using XGBOOST on test data
        predict = predict.reshape(-1, 1)
        predict = scaler1.inverse_transform(predict)#denormalize pedicted demand
        output = ""
        for i in range(len(predict)):
            output += "Test Data = "+str(temp[i])+" Forecasted Cellular Traffic Demand = "+str(predict[i,0])+"<br/><br/>"
        return render_template('result.html', msg=output)

@app.route('/logon')
def logon():
	return render_template('register.html')

@app.route('/login')
def login():
	return render_template('login.html')

@app.route("/signup")
def signup():
    global otp, username, name, email, number, password
    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    otp = random.randint(1000,5000)
    print(otp)
    msg = EmailMessage()
    msg.set_content("Your OTP is : "+str(otp))
    msg['Subject'] = 'OTP'
    msg['From'] = "vandhanatruprojects@gmail.com"
    msg['To'] = email
    
    
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login("vandhanatruprojects@gmail.com", "pahksvxachlnoopc")
    s.send_message(msg)
    s.quit()
    return render_template("val.html")

@app.route('/predict_lo', methods=['POST'])
def predict_lo():
    global otp, username, name, email, number, password
    if request.method == 'POST':
        message = request.form['message']
        print(message)
        if int(message) == otp:
            print("TRUE")
            con = sqlite3.connect('signup.db')
            cur = con.cursor()
            cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
            con.commit()
            con.close()
            return render_template("login.html")
    return render_template("register.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("home.html")
    else:
        return render_template("login.html")


    
if __name__ == '__main__':
    app.run()