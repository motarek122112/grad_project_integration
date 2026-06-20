from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.database import update_user_profile
from app.model_utils import save_uploaded_file
from app.profile_image import save_profile_picture
from app.ui import page_header
from model_adapters.age_gender import predict_age_gender


RESULT_KEY = "combined_profile_prediction"
SUCCESS_KEY = "profile_prediction_saved"


def normalize_gender(label: str) -> str:
    """Normalize the gender label returned by the model."""
    normalized = str(label or "").strip().lower()

    if normalized == "female":
        return "Female"

    if normalized == "male":
        return "Male"

    return "Prefer not to say"


def convert_database_user(
    database_user: dict,
) -> dict:
    """Convert a database user into the session-state format."""
    return {
        "id": int(database_user["id"]),
        "full_name": database_user["full_name"],
        "email": database_user["email"],
        "age": database_user.get("age"),
        "gender": database_user.get("gender"),
        "profile_image_path": database_user.get(
            "profile_image_path"
        ),
        "created_at": database_user.get(
            "created_at"
        ),
        "last_login_at": database_user.get(
            "last_login_at"
        ),
    }


def save_prediction_to_profile(
    *,
    predicted_age: int,
    predicted_gender: str,
    profile_image_path: str,
) -> dict:
    """
    Save the predicted age, gender, and profile picture.

    The session-state user is updated so the sidebar refreshes
    immediately after prediction.
    """
    current_user = st.session_state["user"]

    updated_user = update_user_profile(
        user_id=int(current_user["id"]),
        full_name=current_user["full_name"],
        age=int(predicted_age),
        gender=predicted_gender,
        profile_image_path=profile_image_path,
    )

    if updated_user is None:
        raise RuntimeError(
            "The age, gender, and profile picture "
            "could not be saved."
        )

    session_user = convert_database_user(
        updated_user
    )

    st.session_state["user"] = session_user

    return session_user


user = st.session_state["user"]

profile_complete = (
    user.get("age") is not None
    and bool(user.get("gender"))
)


page_header(
    eyebrow=(
        "PROFILE SETUP"
        if not profile_complete
        else "MY PROFILE"
    ),
    title=(
        "Complete Your AI Profile"
        if not profile_complete
        else "Update Your AI Profile"
    ),
    description=(
        "Take or upload one clear face photo. "
        "The age and gender models will run together. "
        "The same photo will become your profile picture."
    ),
    icon="👤",
)


# ---------------------------------------------------------
# Success message
# ---------------------------------------------------------

if st.session_state.pop(SUCCESS_KEY, False):
    st.success(
        "Your age, gender, and profile picture "
        "were saved successfully."
    )


# ---------------------------------------------------------
# Existing profile information
# ---------------------------------------------------------

if profile_complete:
    profile_column, information_column = st.columns(
        [1, 2]
    )

    with profile_column:
        saved_profile_image = user.get(
            "profile_image_path"
        )

        if (
            saved_profile_image
            and Path(saved_profile_image).exists()
        ):
            st.image(
                saved_profile_image,
                caption="Profile picture",
                width=180,
            )
        else:
            st.info(
                "No profile picture has been saved yet."
            )

    with information_column:
        st.subheader(
            "Currently saved information"
        )

        age_column, gender_column = st.columns(2)

        age_column.metric(
            "Saved age",
            user["age"],
        )

        gender_column.metric(
            "Saved gender",
            user["gender"],
        )

    st.info(
        "Taking or uploading another photo will replace "
        "your saved age, gender, and profile picture."
    )

else:
    st.warning(
        "You must complete this step before accessing "
        "the rest of the Ishara website."
    )


# ---------------------------------------------------------
# Image input
# ---------------------------------------------------------

source = st.radio(
    "Choose the image source",
    [
        "Take a live photo",
        "Upload a photo",
    ],
    horizontal=True,
    key="profile_combined_source",
)

image_input = None

if source == "Take a live photo":
    image_input = st.camera_input(
        "Look directly at the camera and take a clear photo",
        key="combined_profile_camera",
    )

else:
    image_input = st.file_uploader(
        "Upload a clear face photo",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
        key="combined_profile_upload",
    )


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if image_input is not None:
    st.image(
        image_input,
        caption="Age, gender, and profile-picture input",
        width=420,
    )

    button_text = (
        "Predict age and gender and complete profile"
        if not profile_complete
        else "Predict age and gender and update profile"
    )

    if st.button(
        button_text,
        type="primary",
        use_container_width=True,
        key="combined_profile_prediction_button",
    ):
        temporary_image_path = save_uploaded_file(
            image_input
        )

        try:
            with st.spinner(
                "Detecting the face and running the "
                "age and gender models together..."
            ):
                result = predict_age_gender(
                    temporary_image_path
                )

            face_detected = result.get(
                "face_detected"
            )

            if face_detected is False:
                st.warning(
                    "The face detector did not clearly detect "
                    "a face. Try another front-facing photo "
                    "with better lighting."
                )

            predicted_age = int(
                round(float(result["age"]))
            )

            predicted_age = max(
                1,
                min(120, predicted_age),
            )

            predicted_gender = normalize_gender(
                result.get("gender", "")
            )

            profile_image_path = save_profile_picture(
                image_input,
                user_id=int(user["id"]),
                bbox=result.get("bbox"),
            )

            updated_session_user = (
                save_prediction_to_profile(
                    predicted_age=predicted_age,
                    predicted_gender=predicted_gender,
                    profile_image_path=profile_image_path,
                )
            )

            st.session_state[RESULT_KEY] = {
                "age": predicted_age,
                "gender": predicted_gender,
                "age_group": (
                    result.get("age_bin_label")
                    or result.get("age_group")
                    or "Unknown"
                ),
                "gender_confidence": result.get(
                    "gender_confidence"
                ),
                "face_detected": face_detected,
                "bbox": result.get("bbox"),
                "profile_image_path": (
                    updated_session_user.get(
                        "profile_image_path"
                    )
                ),
            }

            st.session_state[SUCCESS_KEY] = True

            # Refresh the profile page and sidebar.
            st.rerun()

        except FileNotFoundError as error:
            st.error(str(error))

        except ValueError as error:
            st.error(str(error))

        except Exception as error:
            st.error(
                "Age and gender prediction or profile-picture "
                "saving failed. Check the terminal for details."
            )
            st.exception(error)


# ---------------------------------------------------------
# Latest result
# ---------------------------------------------------------

prediction = st.session_state.get(
    RESULT_KEY
)

if prediction:
    st.divider()

    st.subheader(
        "Latest combined prediction"
    )

    age_column, group_column, gender_column = (
        st.columns(3)
    )

    age_column.metric(
        "Predicted age",
        prediction["age"],
    )

    group_column.metric(
        "Age group",
        prediction["age_group"],
    )

    gender_column.metric(
        "Predicted gender",
        prediction["gender"],
    )

    gender_confidence = prediction.get(
        "gender_confidence"
    )

    st.metric(
        "Gender confidence",
        (
            f"{float(gender_confidence) * 100:.1f}%"
            if gender_confidence is not None
            else "Not available"
        ),
    )

    latest_profile_image = prediction.get(
        "profile_image_path"
    )

    if (
        latest_profile_image
        and Path(latest_profile_image).exists()
    ):
        st.subheader(
            "Saved profile picture"
        )

        st.image(
            latest_profile_image,
            width=220,
        )
