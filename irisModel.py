import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

from sklearn.preprocessing import StandardScaler

import pickle

from sklearn.datasets import load_iris
import pandas as pd

# Load the dataset
iris = load_iris()

# Convert to a Pandas DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add target labels (species)
df["species"] = iris.target

#Splitting the data into x and y. X being factors and Y being the value the that should be predicted
x=df.drop(columns='species',axis=1)
y=df['species']

#Training
knn=KNeighborsClassifier(n_neighbors=11,metric='euclidean')
knn.fit(x,y)

#Generating a Model
with open('iris.pkl','wb') as f:
    pickle.dump(knn,f)

print('Done!')