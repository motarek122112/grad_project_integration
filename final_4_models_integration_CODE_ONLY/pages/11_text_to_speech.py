import json
import streamlit as st
import streamlit.components.v1 as components


st.title("🗣️ Text to Speech")
st.caption("Browser-based speech output for fast online demo.")

def get_profile_value(keys, default=None):
    for key in keys:
        if key in st.session_state and st.session_state[key] not in (None, ""):
            return st.session_state[key]

    for container_key in ["profile", "user_profile", "current_profile"]:
        profile = st.session_state.get(container_key)
        if isinstance(profile, dict):
            for key in keys:
                if key in profile and profile[key] not in (None, ""):
                    return profile[key]

    return default


gender = get_profile_value(["gender", "profile_gender", "predicted_gender"], "Unknown")
age = get_profile_value(["age", "profile_age", "predicted_age"], "Unknown")

st.info(f"Voice profile → Gender: {gender} | Age: {age}")

text = st.text_area("Enter text", value="hello", height=140)

with st.expander("Voice settings"):
    lang = st.selectbox(
        "Language",
        ["en-US", "en-GB", "ar-EG"],
        index=0,
        help="Use en-US for English text, ar-EG for Arabic text.",
    )

    try:
        age_num = int(float(str(age)))
    except Exception:
        age_num = 35

    default_rate = 1.05 if age_num <= 25 else (0.92 if age_num >= 51 else 1.0)

    rate = st.slider("Speed", 0.6, 1.5, float(default_rate), 0.05)
    pitch = st.slider("Pitch", 0.5, 1.5, 1.0, 0.05)


if st.button("Generate Speech", use_container_width=True):
    clean_text = " ".join(str(text or "").split())

    if not clean_text:
        st.warning("Please enter text first.")
    else:
        text_js = json.dumps(clean_text)
        lang_js = json.dumps(lang)
        rate_js = json.dumps(rate)
        pitch_js = json.dumps(pitch)
        gender_js = json.dumps(str(gender))

        html = f"""
        <div style="font-family:Arial, sans-serif; padding:14px; border-radius:12px; background:#111827; color:white;">
            <button id="playBtn" style="
                width:100%;
                padding:14px;
                border-radius:10px;
                border:0;
                background:#ff3333;
                color:white;
                font-weight:700;
                cursor:pointer;
                font-size:16px;
            ">▶ Play Speech</button>

            <p id="status" style="margin-top:10px;color:#cbd5e1;">
                If audio does not start automatically, press Play Speech.
            </p>

            <script>
            const text = {text_js};
            const lang = {lang_js};
            const rate = {rate_js};
            const pitch = {pitch_js};
            const gender = {gender_js}.toLowerCase();

            function pickVoice() {{
                const voices = window.speechSynthesis.getVoices() || [];
                const sameLang = voices.filter(v => (v.lang || "").toLowerCase().startsWith(lang.toLowerCase().slice(0,2)));

                let candidates = sameLang.length ? sameLang : voices;

                if (gender.includes("male")) {{
                    const male = candidates.find(v => /male|david|mark|daniel|alex|fred|google us english/i.test(v.name));
                    if (male) return male;
                }}

                if (gender.includes("female")) {{
                    const female = candidates.find(v => /female|zira|samantha|victoria|susan|karen|bella/i.test(v.name));
                    if (female) return female;
                }}

                return candidates[0] || null;
            }}

            function speakNow() {{
                const status = document.getElementById("status");

                if (!("speechSynthesis" in window)) {{
                    status.textContent = "Browser speech is not supported here.";
                    return;
                }}

                window.speechSynthesis.cancel();

                const utterance = new SpeechSynthesisUtterance(text);
                utterance.lang = lang;
                utterance.rate = rate;
                utterance.pitch = pitch;

                const voice = pickVoice();
                if (voice) utterance.voice = voice;

                utterance.onstart = () => status.textContent = "Speaking...";
                utterance.onend = () => status.textContent = "Done.";
                utterance.onerror = (e) => status.textContent = "Speech error: " + (e.error || "unknown");

                window.speechSynthesis.speak(utterance);
            }}

            document.getElementById("playBtn").addEventListener("click", speakNow);

            if (typeof speechSynthesis !== "undefined") {{
                speechSynthesis.onvoiceschanged = () => {{}};
            }}

            setTimeout(speakNow, 500);
            </script>
        </div>
        """

        components.html(html, height=130)
        st.success("Speech is ready. If it did not start automatically, press Play Speech.")
