import streamlit as st
import os
import cv2
import altair as alt
import pandas as pd
import numpy as np


chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
xrule = (alt.Chart(pd.DataFrame({'x': [1]})).mark_rule(color='cyan', strokeWidth=2).encode(x='x'))
yrule = (alt.Chart().mark_rule(color='cyan', strokeWidth=2).encode(y=alt.datum('12')))
chart = (alt.Chart(chart_data).mark_line().encode(
    x='a',
    y= 'b'
))

chart = alt.layer(
    chart, xrule
).properties(
    title='Factors that Contribute to Life Expectancy in Malaysia',
    width=500, height=300
)
st.altair_chart(chart)

