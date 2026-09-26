#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("Social Media Productivity Analyzer")

df = pd.read_csv("archivee3.csv")

features = [
    "perceived_productivity_score",
    "stress_level",
    "sleep_hours",
    "screen_time_before_sleep",
    "breaks_during_work",
    "daily_social_media_time",
    "number_of_notifications",
    "work_hours_per_day",
    "coffee_consumption_per_day",
    "days_feeling_burnout_per_month",
    "weekly_offline_hours",
    "job_satisfaction_score",
    "age"
]

df = df[features + ["actual_productivity_score"]].dropna()

X = df[features]
y = df["actual_productivity_score"]

model = LinearRegression()
model.fit(X, y)

st.subheader("Enter Your Details")

values = []

for feature in features:
    value = st.number_input(feature.replace("_", " ").title())
    values.append(value)

if st.button("Predict Productivity"):
    prediction = model.predict([values])
    st.success(f"Predicted Productivity Score: {prediction[0]:.2f}")

