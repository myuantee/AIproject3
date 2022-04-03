# imports libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize
from sklearn.cluster import KMeans 
from sklearn import metrics

st.title("Question 3: Loan Application Modelling")

st.sidebar.header("User Input")
train_ratio = st.sidebar.slider("Training Set Ratio", min_value = 0.1, max_value = 1.0, value = 0.7, step = 0.1)
clus_k = st.sidebar.slider("Number of Clusters (k)", min_value = 1, max_value = 10, value = 5, step = 1)
data = {"train_ratio" : train_ratio,
        "clus_k" : clus_k}

test_ratio = round((1 - train_ratio), 1)
clus_k = int(clus_k)

# loads training data
df = pd.read_csv('Bank_CreditScoring.csv')
st.header("About the Dataset")
st.write(df)

st.markdown("The initial data set contains {0} classes and {1} samples.".format(df.shape[1], df.shape[0]))
st.markdown("{0} classes are numerical data while {1} classes are categorical data.".format(len(df.select_dtypes("number").columns), len(df.select_dtypes("object").columns)))

st.markdown("""---""")
st.subheader("Data Analysis before Preprocessing")

st.markdown("**Correlation Matrix**")
st.caption("computes pairwise correlation for numerical columns")
df_corr = df.corr()
st.write(df_corr)

st.markdown("**Correlation Heatmap**")
fig, ax = plt.subplots(figsize = (15, 10))
sns.heatmap(df_corr.round(2), annot = True)
st.pyplot(fig)

st.markdown("There are only {0} variables because *corr()* can only get correlation for numerical data.".format(df_corr.shape[1]))

st.markdown("""---""")
st.subheader("Data Preprocessing")
st.markdown("Since there are no missing values and outliers detected, therefore, we only need to encode the categorical data.")

# encodes categorical data
df["Decision"] = df.Decision.map(dict(Accept = 1, Reject = 0))
le = LabelEncoder()
test_data = df.copy()
test_data = test_data.apply(le.fit_transform)

st.success("Encoding done.")
st.markdown("**After encoding:**")
st.write(test_data)

st.markdown("""---""")
st.subheader("Data Analysis after Preprocessing")
st.markdown("Let's investigate the correlation again after encoding the categorical data.")
st.markdown("**Correlation Matrix**")
test_data_corr = test_data.corr()
st.write(test_data_corr)

st.markdown("**Correlation Heatmap**")
fig, ax = plt.subplots(figsize = (15, 10))
sns.heatmap(test_data_corr.round(2), annot = True)
st.pyplot(fig)

st.markdown("Now, there are {0} variables because encoding has been done for categorical data.".format(test_data_corr.shape[1]))

st.markdown("""---""")
st.subheader("Data Splitting")
st.markdown("First of all, separate the data into independent variables and dependent variable.")

# splits dataset in features and target variable
X = test_data.drop("Decision", axis = 1) # big X represents independent variable
y = test_data["Decision"] # small y represents dependent variable

st.markdown("**Independent Variables:**")
st.write(X)
st.markdown("**Dependent Variable:**")
st.write(y)

st.markdown("Then, split data into train set and test set with the ratio of 7：3 (default) before training classification models.")
st.markdown("Adjust the **Training Set Ratio** in the sidebar and investigate the classification accuracy.")

# splits dataset into training set and test set
# 70-30 split and set random_state = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_ratio, random_state = 1)
st.success("Splitting done.")

# scales the features
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)

st.markdown("""---""")
st.header("Naive Bayes Classifier")
st.markdown("Naive Bayes Classifier is based on Bayes’ Theorem.")

# creates Naive Bayes classifer object
nb = GaussianNB()
nb.fit(X_train, y_train)
st.success("Training done.")

# predicts the response for test dataset
nb_pred = nb.predict(X_test)

# accuracy
st.markdown("**Accuracy:**")
st.markdown(nb.score(X_test, y_test))

# confusion matrix
st.markdown("**Confusion Matrix:**")
nb_matrix = metrics.confusion_matrix(y_test, nb_pred)

fig, nb_cm = plt.subplots(figsize = (2, 1))
nb_cm = sns.heatmap(nb_matrix, annot = True, xticklabels = [0, 1], yticklabels = [0, 1], cbar = False, fmt = "g")
nb_cm.set_xlabel("Predicted label", fontsize = 5)
nb_cm.set_ylabel("True label", fontsize = 5)
st.pyplot(fig)

st.markdown("""---""")
st.header("Decision Tree Classifier")
st.markdown("Decision Tree Classifier is a tree-structured classifier, where internal nodes represent the features of a dataset, branches represent the decision rules and each leaf node represents the outcome. The data is continuously split according to a certain parameter.")

# creates Decision Tree classifer object
dt = DecisionTreeClassifier() # pruning the tree by setting the depth 
dt = dt.fit(X_train, y_train)
st.success("Training done.")

# predicts the response for test dataset
dt_pred = dt.predict(X_test)

# accuracy
st.markdown("**Accuracy before modifying parameter:**")
st.markdown(dt.score(X_test, y_test))

# modifies parameter
st.markdown("Let's try to modify the Decision Tree parameter by changing *Gini* to *Entropy* and prune the tree by setting the depth at 2.")
dt = DecisionTreeClassifier(criterion = "entropy", max_depth = 2)    #pruning the tree by setting the depth 
dt = dt.fit(X_train, y_train)

# predicts the response for test dataset
dt_pred = dt.predict(X_test)

# accuracy
st.markdown("**Accuracy after modifying parameter:**")
st.markdown(dt.score(X_test, y_test))

st.markdown("**The accuracy increased!**")

# confusion matrix
st.markdown("**Confusion Matrix:**")
dt_matrix = metrics.confusion_matrix(y_test, dt_pred)

fig, dt_cm = plt.subplots(figsize = (2, 1))
dt_cm = sns.heatmap(dt_matrix, annot = True, xticklabels = [0, 1], yticklabels = [0, 1], cbar = False, fmt = "g")
dt_cm.set_xlabel("Predicted label", fontsize = 5)
dt_cm.set_ylabel("True label", fontsize = 5)
st.pyplot(fig)

# normalizes data
scaled_data = normalize(test_data)
scaled_data = pd.DataFrame(scaled_data, columns = test_data.columns)

st.markdown("""---""")
st.header("K-Means")
st.markdown("We have selected the variables *\"Years_to_Financial_Freedom\"* and *\"Loan_Amount\"* to perform K-Means clustering.")
# selects data to be visualised
km_data = scaled_data[["Years_to_Financial_Freedom", "Loan_Amount"]]

st.markdown("Before clustering, we have to normalize the data to make sure all the dimensions are treated equally.")
st.success("Normalizing done.")

st.markdown("**Before normalizing:**")
st.write(test_data.head())

st.markdown("**After normalizing:**")
st.write(scaled_data.head())

st.markdown("Now, let's generate a scatter plot for the selected variables.")
ax = sns.relplot(x = "Years_to_Financial_Freedom", y = "Loan_Amount", hue = y, data = scaled_data)
st.pyplot(ax)
st.markdown("In the figure above, it's hard to determine the number of clusters. Therefore, we apply Elbow Method to identify the optimal value of K.")

# uses a loop from 1 to 10 to determine the best k value
distortions = []
for i in range (2, 11):
    km = KMeans(
        n_clusters = i, init = "random",
        n_init = 10, max_iter = 300,
        tol = 1e-04, random_state = 1
    )
    km.fit(km_data)
    distortions.append(km.inertia_)
    
# plot
fig, ax = plt.subplots()
plt.plot(range(2, 11), distortions, marker = "o")
plt.xlabel("Number of Clusters")
plt.ylabel("Distortion")
st.pyplot(fig)

st.markdown("In the graph above, we conclude that the \"elbow\" is 5. Now, we can train the model on the data set with the number of clusters 5.")
st.markdown("Adjust the **Number of Clusters (k)** in the sidebar and investigate the scatter plot again.")

# selects 5 as the number of clusters, k
km = KMeans(n_clusters = clus_k, random_state = 1)
km.fit(km_data)

# creates a copy of scaled_data and merge the newly predicted labels back
clus_data = scaled_data.copy()
clus_data = clus_data[km_data.columns]
clus_data['Clusters'] = km.labels_
clus_data.head()

ax = sns.relplot(x = "Years_to_Financial_Freedom", y = "Loan_Amount", hue = "Clusters", data = clus_data)
st.pyplot(ax)