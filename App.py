import pandas as pd
import numpy as np
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
# =============================================================================
# import matplotlib.pyplot as plt
# import seaborn as sns
# import catboostimport sweetviz
# import xgboost
# =============================================================================
df = pd.read_csv("INDUSTRY_COPPER.csv")  # Reading the data from csv file
#df.columns
#df.head()
#df.describe()
def Preprocessing(df):  
        # id column is unique and not useful to our analysis
        df.drop(columns = ["id"], axis = 1, inplace = True)
        #cleaning the columns material_ref and quantity tons
        a = df["material_ref"].str.startswith("0000000000")
        b = (a==True)
        df["material_ref"][b] = np.NAN
        df["quantity tons"].values[173086] = 0
        df["quantity tons"] = pd.to_numeric(df["quantity tons"])
        
        cols = ['quantity tons', 'customer', 'country', 'application', 'thickness', 'width', 'selling_price'  ,'status', 'item type', 'material_ref', 'product_ref']
        cont_cols = ['quantity tons', 'customer', 'country', 'application', 'thickness', 'width', 'selling_price']
        cat_cols = [ 'status', 'item type', 'material_ref', 'product_ref']
    
        # Treating null values
        for i in cols:
             if i == 'thickness':
                  si = SimpleImputer(strategy = 'median')       
                  df[i] = si.fit_transform(np.array(df[i]).reshape(-1,1))
             elif i in cat_cols:
                  si = SimpleImputer(strategy = 'most_frequent')       
                  df[i] = si.fit_transform(np.array(df[i]).reshape(-1,1))
             else:
                  si = SimpleImputer(strategy = 'mean')
                  df[i] = si.fit_transform(np.array(df[i]).reshape(-1,1))
        df = df.dropna()
        for i in cols:
            if i in cont_cols:
                    print(i, df[i].apply(lambda x : isinstance(x, float) or isinstance(x, int)).all())
        
        y = df["selling_price"]
        y[y <= 0] = 1e-8
        y = np.log(np.array(y))
        y[y == np.inf] = np.nan
        y[y == -np.inf] = np.nan
        si = SimpleImputer(strategy = 'mean')
        y = si.fit_transform(np.array(y).reshape(-1,1))

        #df["selling_price"].skew()
        # winsorizing to reduce skewness
        df["quantity tons"] = winsorize(df["quantity tons"], limits = [0.1, 0.1])
        df["thickness"] = winsorize(df["thickness"], limits = [0.1, 0.1])
        for col in ['quantity tons', 'width', 'selling_price']:
            df[col] = winsorize(df[col], limits=[0.1, 0.1])
        #df[['quantity tons','width', 'selling_price']].plot.box(figsize = (10,5))
        #st.write(len(df)) # 181674
        # creating a new feature delivery time and deleting the features item date and delivery date
        df['item_date'] = pd.to_datetime(df['item_date'].astype(str).str.rstrip('.0'), format='%Y%m%d', errors='coerce')
        df['delivery date'] = pd.to_datetime(df['delivery date'].astype(str).str.rstrip('.0'), format='%Y%m%d', errors='coerce')
        df.dropna(subset=['item_date','delivery date'], inplace=True)
        df['Delivery_Time'] = (df['delivery date'] - df['item_date']).dt.total_seconds()
        df = df.drop(columns = ["item_date", "delivery date"], axis = 1)
        #st.write(len(df)) # 181667

        # EDA
# =============================================================================
#         sns.heatmap(df.corr()).plot()
#         my_report = sweetviz.analyze([df, "Train"], target_feat = "selling_price")
#         my_report.show_html()
#         df.plot(kind = "box", subplots = True, figsize = (12,5), fontsize = 8)
#         df[['quantity tons','width', 'selling_price']].plot.box(figsize = (10,5))
#         df["selling_price"].plot(kind = 'hist')
# =============================================================================
        # Taking a copy of the dataframe
        copy = df
        return(df)
def Regression(df1, new):
        #encoding the categorical features of the dataset df1
        target_en = ce.TargetEncoder(cols = [ 'status', 'item type', 'material_ref', 'product_ref'])
        b = df1["selling_price"]
        df1 = target_en.fit_transform(df1.drop(columns = ["selling_price"]), df1["selling_price"])
        df1["selling_price"] = b
        
        #encoding the categorical features of the dataset new
        b = new["selling_price"]
        new = target_en.fit_transform(new.drop(columns = ["selling_price"]), new["selling_price"])
        new["selling_price"] = b
        
        # splitting the data
        x = df1.iloc[:, :11]
        y =  df1['selling_price']  
        new = new.drop(columns = ["selling_price"], axis = 1)
        x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = 0.2,  random_state = 5)
        
        # scaling the data
        scaler = StandardScaler()
        # Fit the scaler on the training data
        if new.shape[0] > 0:
                x = scaler.fit(x)
                new = scaler.transform(new)
        else:
                st.write("You have given invalid values. Please check again!!")
        # building machine learning models using algorithms
        from sklearn.ensemble import RandomForestRegressor 
        rf = RandomForestRegressor()
        rf.fit(x_train, y_train)
        #print(r2_score(rf.predict(x_test), y_test))
        c = rf.predict(new[:])
        return(c)
#from collections import Counter
#Counter(copy["status"])
     
def Classification(copy, new1):  
        new1 = new1.drop(columns = ["status"], axis = 1)
        # setting the status to either won or lost
        for i in range(len(copy["status"])):
            if copy["status"].values[i] not in ["Won", "Lost"]:
                copy["status"].values[i] = np.nan
        copy = copy.dropna()

        cat_cols = ['item type', 'material_ref', 'product_ref']
        y = copy["selling_price"]
        z = new1["selling_price"]
        # encoding the categorical features using target encoder
        target_en = ce.TargetEncoder(cols = cat_cols)
        copy = target_en.fit_transform(copy.drop(columns = ["selling_price"]), copy["selling_price"])
        new1 = target_en.fit_transform(new1.drop(columns = ["selling_price"]), new1["selling_price"])
        copy["selling_price"] = y
        new1["selling_price"] = z
        # encoding the status feature using label encoder
        le = LabelEncoder()
        copy["status"] = le.fit_transform(np.array(copy["status"]).reshape(-1,1))        
        # splitting the data
        y = copy["status"]
        x = copy.drop(columns = ["status"], axis = 1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 5)
        # scaling the data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        # Fit the scaler on the training data
        if new1.shape[0] > 0:
                x = scaler.fit(x)
                new1 = scaler.transform(new1)
        else:
                st.write("You have given invalid values. Please check again!!")
        # building ml model using random forest
        rf = RandomForestClassifier()
        rf.fit(x_train, y_train)
        tr_preds = rf.predict(x_train)
        te_preds = rf.predict(x_test)
        #print("acc of tr : ", accuracy_score(tr_preds, y_train))
        #print("acc of te : ", accuracy_score(te_preds, y_test))
        c = rf.predict(np.array(new1[:]))
        return(c)
# building streamlit application
st.title("Industrial Copper Modeling")  # Setting the title of the page
st.header("Regression")
columns = ['item_date', 'quantity tons', 'customer', 'country', 'status', 'item type', 'application', 'thickness', 'width', 'material_ref', 'product_ref', 'delivery date']
for column in columns:
    df.loc[len(df), column] = st.text_input(column)

submitted1 = st.button("Submit1")
if submitted1:
      df = Preprocessing(df)
      df1 = df.iloc[:(len(df)-1), :]
      new = df.iloc[(len(df)-1):(len(df)),:]
      c = Regression(df1, new )  
      st.write("Predicted selling price is ", c)
st.header("Classification")
columns1 = ['item_date', 'quantity tons', 'customer', 'country', 'item type', 'application', 'thickness', 'width', 'material_ref', 'product_ref', 'delivery date', 'selling_price']
# Assuming you have a DataFrame called df
df.drop(index=181667, inplace=True)
for i in columns1:
    df.loc[len(df), i] = st.text_input(i + " " + "input")
submitted2 = st.button("Submit2")
if submitted2:
      df = Preprocessing(df)
      copy= df.iloc[:(len(df)-1), :]
      new1 = df.iloc[(len(df)-1):(len(df)),:]
      c = Classification(copy, new1 )  
      if c == 0:
          st.write('The status is : "Lost"')
      elif c == 1:
          st.write('The status is : "Won"')
