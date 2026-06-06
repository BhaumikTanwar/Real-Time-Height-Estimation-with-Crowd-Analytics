import streamlit as st
from streamlit_autorefresh import st_autorefresh
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

st_autorefresh(
    interval=3000,  
    key="refresh"
)

st.title(
    "Crowd Height Analytics"
)

conn = sqlite3.connect(
    "crowd.db"
)

df = pd.read_sql_query(
    "SELECT * FROM people",
    conn
)

conn.close()
avg_height = (
    df["height"].mean()
)

total_people = len(df)
min_height = (
    df["height"].min()
)
max_height = (
    df["height"].max()
)
st.subheader(
    "Crowd Statistics"
)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "People",
        total_people
    )

with col2:
    st.metric(
        "Average",
        f"{avg_height:.1f} cm"
    )

with col3:
    st.metric(
        "Min",
        f"{min_height:.1f} cm"
    )

with col4:
    st.metric(
        "Max",
        f"{max_height:.1f} cm"
    )

NATIONAL_AVG = 165
difference = avg_height - NATIONAL_AVG
if difference > 0:

    st.success(
        f"Crowd is {difference:.1f} cm taller than national average"
    )

else:

    st.info(
        f"Crowd is {abs(difference):.1f} cm shorter than national average"
    )
st.subheader(
    "Height Distribution"
)   
fig, ax = plt.subplots()

ax.hist(
    df["height"],
    bins=10
)

ax.set_title(
    "Height Distribution"
)

ax.set_xlabel(
    "Height (cm)"
)

ax.set_ylabel(
    "Number of People"
)

st.pyplot(fig)