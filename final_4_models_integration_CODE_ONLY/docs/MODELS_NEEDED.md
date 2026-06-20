# Models needed

Copy these model files into `models/`:

| File name | Required? | Purpose |
|---|---:|---|
| `asl_landmark_best_v3.pth` | Yes | Sign Language to Text |
| `best_tts_acoustic.zip` | Yes | Text to Mel / TTS acoustic |
| `best_speech_model.pth.zip` | Yes | Speech to Text |
| `yolov8n-face-lindevs.pt.zip` | Yes | Face detector |
| `best_age_efficientnet_b4_finetuned.pth.zip` | Yes | Age model |
| `best_gender_utkface.pth.zip` | Yes | Gender model |
| `vocoder_best.pt` | Optional | HiFi-GAN vocoder. If absent, TTS uses Griffin-Lim fallback. |
