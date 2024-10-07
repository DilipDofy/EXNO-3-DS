## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![Screenshot 2024-10-03 103513](https://github.com/user-attachments/assets/59689daa-e4dc-462e-84a5-b97e0be86450)


## ORDINAL ENCODER
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![Screenshot 2024-10-03 103550](https://github.com/user-attachments/assets/576a4ed7-686f-4c32-b582-ce82d75d4411)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![Screenshot 2024-10-03 103621](https://github.com/user-attachments/assets/e7e7a504-d9c6-4c80-96ca-2983f77999d7)


## LABEL ENCODER
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(df[["ord_2"]])
dfc
```
![Screenshot 2024-10-03 103654](https://github.com/user-attachments/assets/914c6fad-448a-4563-bfc3-743a98a010a7)


```
dfc=df.copy()
dfc['con_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![Screenshot 2024-10-03 103732](https://github.com/user-attachments/assets/b6f137ee-65bd-4dac-b83b-c1556dc23ea0)


## ONEHOT ENCODER
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df=df.copy()

enc=pd.DataFrame(ohe.fit_transform(df[['nom_0']]))
enc
```
![Screenshot 2024-10-03 103807](https://github.com/user-attachments/assets/03224a1e-8843-4660-a0e8-f24f51f337a0)


```
df2=pd.concat([df,enc],axis=1)
df2

```

![Screenshot 2024-10-03 103843](https://github.com/user-attachments/assets/dbb0a9e0-8410-4a07-8c89-537782a5c89e)


```
pip install --upgrade category_encoders
```
![Screenshot 2024-10-03 103926](https://github.com/user-attachments/assets/e083239d-935e-41e0-8e44-92c16e7a327e)


## BinaryEncoder
```
from category_encoders import BinaryEncoder
import pandas as pd
df=pd.read_csv("/content/data (1).csv")
df
```
![Screenshot 2024-10-03 104030](https://github.com/user-attachments/assets/1dc4372d-46a5-4da8-9896-646a599e7364)


```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![Screenshot 2024-10-03 104108](https://github.com/user-attachments/assets/e351cdf9-3e76-4315-b041-776ff713fb1f)


## TARGET ENCODER
```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![Screenshot 2024-10-03 104251](https://github.com/user-attachments/assets/0bb9b1b7-cdd0-45b1-a0b4-a18c495d4639)



## FEATURE ENGINEERING
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![Screenshot 2024-10-03 104342](https://github.com/user-attachments/assets/8b5744e8-3fbd-4e01-8164-bbf4577121f2)

```
df.skew()

```
![image](https://github.com/user-attachments/assets/9def3c3a-d4ed-46c5-83f1-399301b35006)

```

df["Highly Positive Skew"]=np.log(df["Highly Positive Skew"])
df

```

![image](https://github.com/user-attachments/assets/f61043ac-3fb1-44b6-b682-1302c7ab4ce2)

```

df["Moderate Positive Skew"]=np.reciprocal(df["Moderate Positive Skew"])
df

```

![image](https://github.com/user-attachments/assets/c9c09563-c5c2-4eca-9681-04d08aaaf817)

```

df["Highly Positive Skew"]=np.sqrt(df["Highly Positive Skew"])
df

```

![image](https://github.com/user-attachments/assets/08f6b9da-99b0-4bbe-a49d-cb6b4d12817f)

```

df["Highly Positive Skew"]=np.square(df["Highly Positive Skew"])
df

```

![image](https://github.com/user-attachments/assets/799e50eb-f4a5-4d10-a4b6-46572e30b85e)

## POWER TRANSFORMATION

```

df["Highly Positive Skew"],parameter=stats.boxcox(df["Highly Positive Skew"])
df

```

![image](https://github.com/user-attachments/assets/b6b0f4f2-e2f5-47d8-a9c6-876ab4366c01)

```

df["Moderate Negative Skew_yeojohnson"],parameter=stats.yeojohnson(df["Moderate Negative Skew"])
df

```


![image](https://github.com/user-attachments/assets/f3c51986-dc9b-4575-ba98-9456f82cacce)

```

import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

```

![image](https://github.com/user-attachments/assets/cc55a050-ffc7-4ed1-9761-d739adb3a207)

```

sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()

```

![image](https://github.com/user-attachments/assets/d44a5f2e-efce-4a3f-922b-8ab97e8791aa)

```

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

```

![image](https://github.com/user-attachments/assets/d8b8ccba-ded9-4b22-9786-383cc9efed47)

# RESULT:
    
Thus,the given data are read and Feature Encoding and Transformation process are performed and the data is saved to the file.


       
