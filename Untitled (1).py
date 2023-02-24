#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # Import csv file

# In[10]:


df_candy = pd.read_csv("/Users\Lenovo\Documents\candy-data.csv")


# In[11]:


df_candy.head(3)


# # Get to know my data

# In[12]:


# Largest values for DV "winpercent"
df_candy.sort_values(by="winpercent", ascending=False).head(5)


# In[13]:


# Smallest values for DV "winpercent"
df_candy.sort_values(by="winpercent").head(5)


# In[14]:


# Average of DV "winpercent"

print("Sum of all winpercent: " + str(sum(df_candy["winpercent"])))
print("Number of Datapoints: " + str(len(df_candy["winpercent"])))
print("The average is " + str(sum(df_candy["winpercent"]) / len(df_candy["winpercent"])))


# In[15]:


# Median of DV "winpercent" = average of 42nd and 43rd value = 41st and 42nd Index divided by two

series_winpercent = df_candy["winpercent"]
sorted_series_winpercent = series_winpercent.sort_values()
print("42nd value: " + str(sorted_series_winpercent.iloc[41]))
print("43rd value: " + str(sorted_series_winpercent.iloc[42]))

print("The median is " + str((sorted_series_winpercent.iloc[41] + sorted_series_winpercent.iloc[42]) / 2))


# In[16]:


# Check boxplot DV "winpercent" for outliers
plt.boxplot(df_candy["winpercent"], vert = 0)
plt.title("Are there any outliers in winpercent?")
plt.xlabel("Winpercent")
plt.show()


# In[17]:


# Inspect histogram of DV "winpercent" to assess distribution 
plt.hist(df_candy["winpercent"])
plt.title("Histogram of winpercent")
plt.xlabel("winpercent")
plt.ylabel("Frequency")
plt.show()


# # Do the same for IVs "sugarpercent" and "pricepercent"

# # Inspect other IVs "chocolate", "fruity", etc.

# In[18]:


# Find out distribution of IV "chocolate"

print(sum(df_candy["chocolate"]))
print(len(df_candy["chocolate"]))
print("The proportion of candy that contains chocolate is " + str(sum(df_candy["chocolate"]) / len(df_candy["chocolate"])))


# # Do the same for the rest of the dichotomous IVs

# In[19]:


# Find out distribution of IV "fruity"

print(sum(df_candy["fruity"]))
print(len(df_candy["fruity"]))
print("The proportion of candy that is fruity is " + str(sum(df_candy["fruity"]) / len(df_candy["fruity"])))

# Etc. pp.


# # Do Multiple Linear Regression Analysis

# In[20]:


y = df_candy["winpercent"].values
print(y)


# In[21]:


X = df_candy[["chocolate", "fruity", "caramel", "peanutyalmondy", "nougat", "crispedricewafer", "hard", "bar", "pluribus", "sugarpercent", "pricepercent"]].values
print(X)


# In[22]:


from sklearn.linear_model import LinearRegression


# In[23]:


regr = LinearRegression()


# In[24]:


regr.fit(X, y)


# In[25]:


#predict the "winpercent" of a candy like "100 Grand" (real rating = 66.97) which contains "chocolate", "caramel", "crispedricewafer", "bar", and 0.73 "sugarpercent" and 0.86 "pricepercent":
predicted_winpercent = regr.predict([[1, 0, 1, 0, 0, 1, 0, 1, 0, 0.73, 0.86]])
print(predicted_winpercent)


# In[26]:


# Find out regression coefficients for regression equation, explanation: 1.00 increase in characteristic explains which increase in "winpercent"?
print(regr.coef_)


# In[27]:


# Note for me: it seems that "chocolate", "fruit", "caramel", "peanutyalmondy"=good; "nougat"=doesnt matter; "crispedricewafer"=good; "hard"=bad; "bar", "pluribus"=dont matter; 1.00 more sugar=better but sugar is measured in 0.xx; 1.00 higher price=worse but price is measured in 0.yy


# In[28]:


# I still havent calculated Rsquared,statistical significance, Confidence Intervalls, checked assumptions (work in progress...)


# In[ ]:




