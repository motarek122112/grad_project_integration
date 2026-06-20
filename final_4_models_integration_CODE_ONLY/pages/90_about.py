import streamlit as st

from app.ui import page_header


page_header(
    eyebrow="GRADUATION PROJECT",
    title="About Ishara",
    description="Online AI-Based Sign Language Translation Glasses.",
    icon="ℹ️",
)

st.markdown(
    """
    ### Project

    **Ishara: Online AI-Based Sign Language Translation Glasses**

    Ishara is an accessibility-focused AI platform that brings together sign
    language translation, speech processing, age prediction, and gender
    detection in one modular website.

    ### University

    Zewail City of Science and Technology  
    School of Computational Sciences and Artificial Intelligence

    ### Team

    - Amel Emad
    - Mohamed Osama
    - Mohamed Tarek
    - Seif Ahmed

    ### Supervisor

    Dr. Mohamed Ghalwash

    ### Website architecture

    The upgraded website separates authentication, user data, user-interface
    pages, and model inference adapters. This makes each model easier to test,
    maintain, replace, and deploy independently.
    """
)
