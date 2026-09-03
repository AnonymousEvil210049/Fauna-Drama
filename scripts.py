import base64
import io
import os
import numpy as np
import soundfile as sf
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import butter, filtfilt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Set up page with a Forest aesthetic
st.set_page_config(
    page_title="Fauna-Drama | Multi-Species Translator",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOCAL BACKGROUND AUDIO (Safe HTML5 Version) ---
LOCAL_AUDIO_FILE = "forest_ambient.mp3" 

def get_audio_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

with st.sidebar:
    st.image("https://images.unsplash.com/photo-1542273917363-3b1817f69a2d?q=80&w=500&auto=format&fit=crop", use_container_width=True)
    st.markdown("### 🍃 Environment Controls")
    st.caption("Adjust or pause the ambient sound of the forest below.")
    
    if os.path.exists(LOCAL_AUDIO_FILE):
        audio_b64 = get_audio_base64(LOCAL_AUDIO_FILE)
        st.markdown(
            f"""
            <audio id="bg-forest-audio" controls loop style="width: 100%; border-radius: 8px;">
                <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
            </audio>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning(f"Audio '{LOCAL_AUDIO_FILE}' not found. Please place it in the same folder.")

# --- ORIGINAL ANIMAL DB (With Emojis Added) ---
ANIMAL_DB = {
    "Lion": {"emoji": "🦁", "features": [0.95, 0.15, 0.3, 0.2], "jokes": {"Quiet": "I am practicing my mighty roar in my head. Do not disturb the king.", "Moderate": "I just woke up from an 18-hour nap and I demand immediate snacks.", "Loud": "I AM THE APEX PREDATOR AND SOMEONE TOOK MY FAVORITE SUNBATHING ROCK!"}},
    "Elephant": {"emoji": "🐘", "features": [0.9, 0.05, 0.1, 0.1], "jokes": {"Quiet": "Just sending a subsonic rumble to my cousin three miles away.", "Moderate": "I never forget a face, and I distinctly remember you ignoring me in 2018.", "Loud": "MAKE WAY! THIS IS THE TRUNK-TO-TRUNK COMMUTE LANE AND YOU ARE IN IT!"}},
    "Dog": {"emoji": "🐶", "features": [0.6, 0.4, 0.6, 0.4], "jokes": {"Quiet": "I buried a sock. Do not tell the tall humans.", "Moderate": "I am requesting a brief belly rub. Please submit your application.", "Loud": "A LEAF JUST TOUCHED THE DRIVEWAY! I AM DEFENDING THE REALM!"}},
    "Cat": {"emoji": "🐱", "features": [0.4, 0.7, 0.7, 0.6], "jokes": {"Quiet": "I am judging your outfit from this shadow.", "Moderate": "The food bowl is 10% empty. Fix this tragedy.", "Loud": "I DEMAND TO GO OUTSIDE SO I CAN IMMEDIATELY ASK TO COME BACK INSIDE!"}},
    "Bear": {"emoji": "🐻", "features": [0.8, 0.1, 0.4, 0.2], "jokes": {"Quiet": "I am calculating the optimal trajectory to the nearest picnic basket.", "Moderate": "I am large, I am furry, and I am deeply annoyed by mosquitos.", "Loud": "WHO TOOK THE HONEY? I AM PREPARED TO OVERTURN THIS ENTIRE FOREST!"}},
    "Wolf": {"emoji": "🐺", "features": [0.7, 0.3, 0.8, 0.4], "jokes": {"Quiet": "Sneaking through the brush. Being a lone wolf is actually just very lonely.", "Moderate": "Has anyone seen Steve? We were supposed to hunt 10 minutes ago.", "Loud": "AWOOOOO! THE MOON IS OUT AND I JUST REMEMBERED SOMETHING EMBARRASSING!"}},
    "Monkey": {"emoji": "🐵", "features": [0.8, 0.8, 0.9, 0.7], "jokes": {"Quiet": "I am secretly analyzing the structural integrity of your camera.", "Moderate": "I have a banana and you do not. I win.", "Loud": "I HAVE DECIDED TO CAUSE ABSOLUTE CHAOS IN THE CANOPY TODAY!"}},
    "Horse": {"emoji": "🐴", "features": [0.7, 0.4, 0.5, 0.4], "jokes": {"Quiet": "I am majestic. I am grace. I am mildly startled by my own shadow.", "Moderate": "Could you please pass the premium oats? Thank you.", "Loud": "I AM GALLOPING AT MAXIMUM SPEED BECAUSE A PLASTIC BAG MOVED!"}},
    "Cow": {"emoji": "🐮", "features": [0.5, 0.2, 0.3, 0.2], "jokes": {"Quiet": "Chewing grass. Thinking about grass. Grass is good.", "Moderate": "Moo. That translates to: 'Please respect my personal grazing space.'", "Loud": "I SAID MOO! THE MILKING SCHEDULE IS LATE AND I DEMAND EXPLANATIONS!"}},
    "Crow": {"emoji": "🐦‍⬛", "features": [0.6, 0.7, 0.8, 0.6], "jokes": {"Quiet": "Plotting to steal shiny objects. Normal bird stuff.", "Moderate": "I remember your face. I will hold this grudge for five generations.", "Loud": "ATTENTION! THAT SHINY FOIL WRAPPER IS NOW MY ENTIRE NET WORTH!"}},
    "Owl": {"emoji": "🦉", "features": [0.3, 0.2, 0.2, 0.2], "jokes": {"Quiet": "I know all the secrets of this forest.", "Moderate": "Who? Exactly. None of your business.", "Loud": "I AM TRYING TO SLEEP BUT THESE SQUIRRELS ARE TOO LOUD!"}},
    "Duck": {"emoji": "🦆", "features": [0.5, 0.5, 0.7, 0.5], "jokes": {"Quiet": "Paddling silently. My feet are working incredibly hard down here.", "Moderate": "Quack. I demand bread crumbs. Gluten intolerance is a human construct.", "Loud": "THE BREAD IS MINE! ALL THE BREAD IS MINE!"}},
    "Rooster": {"emoji": "🐓", "features": [0.7, 0.7, 0.8, 0.6], "jokes": {"Quiet": "Clearing my throat for the morning announcements.", "Moderate": "It is 5:00 AM. Why are you not awake yet?", "Loud": "WAKE UP EVERYONE! THE SUN HAS RETURNED! I SAVED US ALL!"}},
    "Peacock": {"emoji": "🦚", "features": [0.8, 0.9, 0.7, 0.8], "jokes": {"Quiet": "Just fluffing my feathers. Maintaining this level of beauty is exhausting.", "Moderate": "Did you see my tail? Look at it. Look at it again.", "Loud": "I AM FABULOUS AND EVERYONE WITHIN A FIVE-MILE RADIUS NEEDS TO KNOW!"}},
    "Cuckoo": {"emoji": "🐦", "features": [0.4, 0.6, 0.2, 0.5], "jokes": {"Quiet": "I am practicing my harmonies for the spring forest festival.", "Moderate": "My vocal range is immaculate. Please enjoy this concert.", "Loud": "I AM THE MOST MAJESTIC BIRD IN THIS HEMISPHERE! HEAR MY SONG!"}},
    "Snake": {"emoji": "🐍", "features": [0.2, 0.9, 0.1, 0.9], "jokes": {"Quiet": "Sssssneaky sssssnake behavior.", "Moderate": "Sssssomeone turned off the heat lamp. Prepare for passive-aggressive slithering.", "Loud": "DO NOT TREAD ON ME OR MY CAREFULLY ARRANGED SUN ROCK!"}},
    "Frog": {"emoji": "🐸", "features": [0.3, 0.2, 0.2, 0.1], "jokes": {"Quiet": "Sitting on a lily pad. Thinking about flies.", "Moderate": "Ribbit. It is Wednesday. I am moist. Life is acceptable.", "Loud": "IT IS RAINING! THIS IS THE GREATEST DAY OF MY LIFE!"}},
    "Alligator": {"emoji": "🐊", "features": [0.7, 0.1, 0.6, 0.1], "jokes": {"Quiet": "I am basically a floating log. A very dangerous log.", "Moderate": "I live in swamp mud. Do not ask me about corporate goals.", "Loud": "THIS SWAMP IS MINE! GET OUT OF MY MUD!"}},
    "Dolphin": {"emoji": "🐬", "features": [0.5, 0.9, 0.4, 0.9], "jokes": {"Quiet": "Echolocating some tasty fish. Mind your business.", "Moderate": "I am significantly smarter than you, but I will do a flip for fish.", "Loud": "CLICK CLICK EEEEEE! THAT MEANS 'HELLO' IN OCEAN!"}},
    "Cricket": {"emoji": "🦗", "features": [0.2, 0.9, 0.8, 0.9], "jokes": {"Quiet": "Just warming up my legs for the evening concert.", "Moderate": "Chirp. You made a bad joke and I am here to highlight the silence.", "Loud": "I AM IN YOUR WALLS AND I WILL CHIRP UNTIL YOU LOSE YOUR MIND!"}}
}

@st.cache_resource
def init_classifier():
    X_train = [data["features"] for data in ANIMAL_DB.values()]
    y_train = list(ANIMAL_DB.keys())
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=1, metric='euclidean'))
    ])
    clf.fit(X_train, y_train)
    return clf

clf = init_classifier()

# --- AUDIO PROCESSING FUNCTIONS ---
def reduce_background_noise(signal, sr):
    nyquist = 0.5 * sr
    low = 500 / nyquist 
    high = min(7000 / nyquist, 0.99)
    if high > low:
        b, a = butter(4, [low, high], btype='band')
        signal = filtfilt(b, a, signal)
    threshold = 0.08 * np.max(np.abs(signal)) 
    signal = np.where(np.abs(signal) < threshold, 0, signal)
    return signal

def extract_acoustic_parameters(signal, sample_rate):
    rms = np.sqrt(np.mean(signal**2))
    loudness = min(float(rms * 4.0), 1.0) 
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1/sample_rate)
    
    valid_idx = np.where(freqs > 200)[0]
    if len(valid_idx) > 0:
        dominant_freq = freqs[valid_idx[np.argmax(spectrum[valid_idx])]] 
    else:
        dominant_freq = freqs[np.argmax(spectrum)]
        
    pitch = min(dominant_freq / 2500.0, 1.0) 
    chunks = np.array_split(signal, 10)
    chunk_energies = [np.sqrt(np.mean(c**2)) for c in chunks if len(c) > 0]
    volatility = min(float(np.std(chunk_energies) / (np.mean(chunk_energies) + 1e-6)), 1.0)
    
    centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-6)
    brightness = min(float(centroid / 4500.0), 1.0) 
    return loudness, pitch, volatility, brightness, dominant_freq

def hz_to_chord(hz):
    if hz < 65: return "Sub-bass (Percussive)"
    if hz > 4000: return "High-freq (Sibilance)"
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    h = round(12 * np.log2(hz / 440.0))
    n = int((h + 57) % 12)
    root = notes[n]
    third = notes[(n + 4) % 12]
    fifth = notes[(n + 7) % 12]
    return f"{root} Major ({root}-{third}-{fifth})"

def plot_waveform(signal):
    fig = go.Figure()
    subsample = signal[::max(1, len(signal)//1000)]
    fig.add_trace(go.Scatter(y=subsample, mode='lines', line=dict(color='#69f0ae', width=2)))
    fig.update_layout(
        title=dict(text="Acoustic Signature", font=dict(size=18)),
        margin=dict(l=10, r=10, t=40, b=10),
        height=140,
    )
    return fig

# --- MAIN UI ---
st.title("🌲 Fauna-Drama")
st.markdown("<p style='font-size:1.2rem; opacity:0.9;'><strong>Team Voice of the Forest</strong> | <em>AI Bio-Acoustic Multi-Species Translator</em></p>", unsafe_allow_html=True)
st.write("") 

# --- SESSION STATE FIXES FOR MIC/UPLOAD ---
if 'active_audio' not in st.session_state:
    st.session_state.active_audio = None

def process_mic():
    if st.session_state.mic_input is not None:
        st.session_state.active_audio = st.session_state.mic_input

def process_upload():
    if st.session_state.file_input is not None:
        st.session_state.active_audio = st.session_state.file_input

tab1, tab2 = st.tabs(["🎤 Live Microphone", "📁 Upload Audio File"])

with tab1:
    st.audio_input(
        "Record a forest sound or bird call:", 
        key="mic_input", 
        on_change=process_mic
    )
        
with tab2:
    st.file_uploader(
        "Upload Audio", 
        type=["wav", "mp3"], 
        label_visibility="collapsed", 
        key="file_input", 
        on_change=process_upload
    )

# --- EXECUTE TRANSLATION ---
if st.session_state.active_audio is not None:
    try:
        with st.spinner("Decoding biological frequencies..."):
            raw_data, sr = sf.read(io.BytesIO(st.session_state.active_audio.getvalue()))
            if len(raw_data.shape) > 1: 
                raw_data = raw_data.mean(axis=1)
            
            clean_data = reduce_background_noise(raw_data, sr)
            raw_float = clean_data.astype(float)
            
            if np.max(np.abs(raw_float)) > 0:
                processed_signal = raw_float / np.max(np.abs(raw_float))
                loudness, pitch, volatility, brightness, hz = extract_acoustic_parameters(processed_signal, sr)
                
                # ML PREDICTION
                features = np.array([[loudness, pitch, volatility, brightness]])
                secret_profile = clf.predict(features)[0]
                
                l_tag = "Quiet" if loudness < 0.35 else ("Loud" if loudness > 0.75 else "Moderate")
                joke = ANIMAL_DB[secret_profile]["jokes"][l_tag]
                emoji = ANIMAL_DB[secret_profile]["emoji"]
                chord = hz_to_chord(hz)
                
                st.write("") 
                st.plotly_chart(plot_waveform(processed_signal), use_container_width=True)
                
                # Translation HUD
                st.markdown(f"""
                <div style="padding: 20px; border-radius: 10px; background-color: rgba(100, 100, 100, 0.1);">
                    <h2 style="margin-bottom: 5px;">{emoji} Translated as: {secret_profile}</h2>
                    <p style="font-size: 1.2rem; font-style: italic;">"{joke}"</p>
                    <hr>
                    <p><strong>🎵 Detected Pitch Signature:</strong> {chord} ({int(hz)} Hz)</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.write("")
                st.write("")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Volume State", l_tag)
                col2.metric("Base Pitch", f"{int(hz)} Hz")
                col3.metric("Vocal Volatility", f"{int(volatility*100)}%")
                col4.metric("Tonal Brightness", f"{int(brightness*100)}%")
            else:
                st.warning("The audio was completely silent. Try making a louder sound!")
                
    except Exception as e:
        st.error(f"Error processing audio file. ({str(e)})")