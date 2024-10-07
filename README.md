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
![image](https://github.com/user-attachments/assets/25385a62-dc4d-4b28-8706-f33b65377dcb)



## ORDINAL ENCODER
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/9a472bb4-2472-4f85-9d93-7d49162c2882)


```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/d1a4471a-2a98-4df8-80c9-b0374cea0c88)



## LABEL ENCODER
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(df[["ord_2"]])
dfc
```
![image](https://github.com/user-attachments/assets/ac9daa50-54b5-41a0-9eaa-f7f62445d8a8)


```
dfc=df.copy()
dfc['con_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/1183c083-7455-44d9-ab5a-6364db6bb66e)



## ONEHOT ENCODER
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df=df.copy()

enc=pd.DataFrame(ohe.fit_transform(df[['nom_0']]))
enc
```
![image](https://github.com/user-attachments/assets/24c10026-6c4f-4ab4-8bd2-acc438cfcdb1)



```
df2=pd.concat([df,enc],axis=1)
df2

```
![image](https://github.com/user-attachments/assets/333353b0-c4d3-413d-80a2-6583b0a407d7)



```
pip install --upgrade category_encoders
```
![image](https://github.com/user-attachments/assets/7dda6d75-6fcd-456d-9f89-f365b62cf6ba)


## BinaryEncoder
```
from category_encoders import BinaryEncoder
import pandas as pd
df=pd.read_csv("/content/data (1).csv")
df
```
![image](https://github.com/user-attachments/assets/798048a2-a388-4c9d-82cc-0b0b58c227b1)



```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![image](https://github.com/user-attachments/assets/53658338-c752-428d-92fe-21d434551715)



## TARGET ENCODER
```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![image](https://github.com/user-attachments/assets/d81844f8-201e-41af-b8f5-efb77cd00365)




## FEATURE ENGINEERING
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/937deb29-8cb4-4277-8436-e44cfe711af3)


```
df.skew()

```
![image](https://github.com/user-attachments/assets/f160a7c1-621e-4017-9eb3-cb6d65ea9414)


```

df["Highly Positive Skew"]=np.log(df["Highly Positive Skew"])
df

```

![image](https://github.com/user-attachments/assets/fb4e3477-40f1-4023-bb97-94cdfda1e9b2)


```

df["Moderate Positive Skew"]=np.reciprocal(df["Moderate Positive Skew"])
df

```

![image](https://github.com/user-attachments/assets/edc32358-aed9-46d8-9153-789b5a6cd1b1)


```

df["Highly Positive Skew"]=np.sqrt(df["Highly Positive Skew"])
df

```

![image](https://github.com/user-attachments/assets/a15cb26d-ab41-445d-aaa1-153ce9eabc06)


```

df["Highly Positive Skew"]=np.square(df["Highly Positive Skew"])
df

```

![image](https://github.com/user-attachments/assets/f8dd148c-77d4-4c16-bbce-d1c95118debd)


## POWER TRANSFORMATION

```

df["Highly Positive Skew"],parameter=stats.boxcox(df["Highly Positive Skew"])
df

```
![image](https://github.com/user-attachments/assets/7cab4b9e-a95b-4d78-9e9d-62c9cc638568)


```

df["Moderate Negative Skew_yeojohnson"],parameter=stats.yeojohnson(df["Moderate Negative Skew"])
df

```

![image](https://github.com/user-attachments/assets/83c26ae6-dbee-453d-8312-e998b91e2c25)


```

import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

```

![image](https://github.com/user-attachments/assets/9f76cb8c-9f19-4056-b11e-64ff77841d39)


```

sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()

```

![image](https://github.com/user-attachments/assets/c07f7245-6df9-436c-93f8-801b323c405a)


```

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

```

![image](https://github.com/user-attachments/assets/fc6e95d7-50d0-4bdd-b3c5-7bcb4fd52421)


# RESULT:
    
Thus,the given data are read and Feature Encoding and Transformation process are performed and the data is saved to the file.


       
