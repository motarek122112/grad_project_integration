import streamlit as st

from app.ui import model_card, page_header


user = st.session_state["user"]
first_name = user["full_name"].split()[0]

page_header(
    eyebrow="ISHARA AI PLATFORM",
    title=f"Welcome, {first_name}",
    description=(
        "One accessible workspace for sign language, speech, "
        "age prediction, and gender detection."
    ),
    icon="🤟",
)

if user.get("age") is None or not user.get("gender"):
    st.warning(
        "Your profile is not complete. Open **My Profile** and save your age and gender."
    )

st.markdown(
    """
    <section class="hero-panel">
        <div>
            <span class="eyebrow light">GRADUATION PROJECT</span>
            <h2>Communication without barriers</h2>
            <p>
                Ishara combines five AI modules in one clean interface.
                Choose a model from the sidebar, provide the required input,
                and review the result on its dedicated page.
            </p>
        </div>
        <div class="hero-symbol">AI</div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.subheader("Explore the AI models")
st.caption("Each model now has its own independent page and adapter module.")

row_one = st.columns(3)
with row_one[0]:
    model_card(
        icon="🤟",
        title="Sign Language to Text",
        description="Translate sign-language video into readable text.",
        input_label="Video",
        output_label="Text + confidence",
    )
with row_one[1]:
    model_card(
        icon="🔊",
        title="Text to Speech",
        description="Generate speech audio from written text.",
        input_label="Text",
        output_label="WAV audio",
    )
with row_one[2]:
    model_card(
        icon="🎙️",
        title="Speech to Text",
        description="Convert an uploaded voice recording into text.",
        input_label="Audio",
        output_label="Transcription",
    )

row_two = st.columns(2)
with row_two[0]:
    model_card(
        icon="🎂",
        title="Age Prediction",
        description="Estimate a person's age from a face image.",
        input_label="Image",
        output_label="Age + age group",
    )
with row_two[1]:
    model_card(
        icon="👤",
        title="Gender Detection",
        description="Predict the model's gender class from a face image.",
        input_label="Image",
        output_label="Class + confidence",
    )

st.info(
    "Use the sidebar to open a model. Complete your profile first so the app "
    "can associate age and gender information with your account."
)
