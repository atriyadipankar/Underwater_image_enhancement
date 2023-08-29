#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import torch
from PIL import Image
from model import PhysicalNN
from test import main as test_main

# Title and sidebar
st.title("Underwater Image Enhancement")
st.sidebar.title("Settings")
checkpoint = st.sidebar.file_uploader("Upload Model Checkpoint", type=["pth"])
image = st.file_uploader("Upload Test Image", type=["jpg", "jpeg", "png"])

if st.button("Enhance Image"):
    if checkpoint and image:
        # Save the uploaded image
        image_path = "./images/test_image.png"
        img = Image.open(image)
        img.save(image_path)

        # Run the test script with the uploaded checkpoint and image
        test_main(checkpoint.name, "./images", "./results")

        # Display enhanced image
        result_image = Image.open("./results/test_image/corrected.png")
        st.image(result_image, caption="Enhanced Image", use_column_width=True)


# In[ ]:




