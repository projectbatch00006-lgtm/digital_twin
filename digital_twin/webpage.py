Here's a concrete skeleton using Streamlit's modern multipage app structure (`st.navigation` + `st.Page`, available in Streamlit ≥1.36).

## File structure

```
project/
├── app.py                  # entry point, sets up navigation
├── pages_content/
│   ├── overview.py         # table of contents / landing page
│   ├── monitor.py          # input form + RF+IF prediction + color status
│   └── chat.py             # login-gated RAG chat page
├── models/
│   ├── rf_model.pkl
│   ├── isolation_forest.pkl
│   └── rag_pipeline.py     # your retrieval + generation logic
├── auth.py                 # simple login check
└── utils.py                # shared helpers (color logic, session state init)
```

## `app.py`

```python
import streamlit as st

st.set_page_config(page_title="Fault Monitoring System", layout="wide")

overview = st.Page("pages_content/overview.py", title="Overview", icon="🏠")
monitor = st.Page("pages_content/monitor.py", title="Monitor & Diagnose", icon="📊")
chat = st.Page("pages_content/chat.py", title="Instructions Chat", icon="💬")

pg = st.navigation([overview, monitor, chat])
pg.run()
```

This alone gives you the top/side nav with a table of contents — no manual sidebar wiring needed.

## `utils.py` — shared logic

```python
import streamlit as st

def init_session_state():
    if "authorized" not in st.session_state:
        st.session_state.authorized = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None
    if "reading_log" not in st.session_state:
        st.session_state.reading_log = []

def get_status(rf_label, if_anomaly_score):
    # Combine RF + Isolation Forest into one status
    if rf_label == "baseline" and if_anomaly_score > 0:
        return "green", "Healthy"
    elif rf_label != "baseline" and if_anomaly_score < 0:
        return "red", "Fault likely"
    else:
        return "yellow", "Needs consideration"  # disagreement case
```

## `pages_content/overview.py`

```python
import streamlit as st

st.title("Industrial Fault Monitoring & Diagnosis System")
st.markdown("""
### How this works
1. **Monitor & Diagnose** — enter live sensor readings (voltage, current, vibration, temperature)
2. The system runs a **Random Forest** classifier and an **Isolation Forest** anomaly detector together
3. Status is shown as 🟢 Healthy / 🟡 Needs consideration / 🔴 Fault likely
4. If a fault is flagged, authorized personnel can open **Instructions Chat** to get remediation steps from the RAG assistant
""")
# Optional: st.image("architecture_diagram.png")
```

## `pages_content/monitor.py`

```python
import streamlit as st
from utils import init_session_state, get_status
# from models.predict import run_rf, run_isolation_forest

init_session_state()
st.title("📊 Sensor Input & Diagnosis")

col1, col2 = st.columns(2)
with col1:
    volt = st.number_input("Voltage (V)", min_value=0.0)
    amp = st.number_input("Current (A)", min_value=0.0)
with col2:
    vibration = st.number_input("Vibration (g)", min_value=0.0)
    temp = st.number_input("Temperature (°C)", min_value=0.0)

if st.button("Run Diagnosis"):
    # rf_label, rf_probs = run_rf([volt, amp, vibration, temp])
    # if_score = run_isolation_forest([volt, amp, vibration, temp])
    rf_label, if_score = "imbalance", -0.1  # placeholder

    color, status_text = get_status(rf_label, if_score)
    st.session_state.last_prediction = {"label": rf_label, "status": status_text}
    st.session_state.reading_log.append(
        {"volt": volt, "amp": amp, "vibration": vibration, "temp": temp, "status": status_text}
    )

    badge_color = {"green": "🟢", "yellow": "🟡", "red": "🔴"}[color]
    st.markdown(f"## {badge_color} {status_text}")
    st.write(f"**Predicted fault type:** {rf_label}")

    # Show feature importance / explanation
    # st.bar_chart(feature_importance_dict)

    if color == "red":
        st.warning("Fault detected. Please proceed to **Instructions Chat** for remediation steps.")

# Trend log
if st.session_state.reading_log:
    st.subheader("Recent Readings")
    st.dataframe(st.session_state.reading_log)
```

## `auth.py`

```python
import streamlit as st

# For a mini-project: hardcoded or simple credential check
# Swap for streamlit-authenticator later if you want hashed passwords
VALID_USERS = {"engineer1": "pass123", "supervisor": "pass456"}

def login_form():
    st.subheader("🔒 Authorized Personnel Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if VALID_USERS.get(username) == password:
            st.session_state.authorized = True
            st.rerun()
        else:
            st.error("Invalid credentials")
```

## `pages_content/chat.py`

```python
import streamlit as st
from utils import init_session_state
from auth import login_form
# from models.rag_pipeline import get_rag_response

init_session_state()
st.title("💬 Instructions Chat")

if not st.session_state.authorized:
    login_form()
    st.stop()

if st.session_state.last_prediction is None:
    st.info("No active fault diagnosis. Go to Monitor & Diagnose first.")
    st.stop()

st.write(f"**Current fault context:** {st.session_state.last_prediction['label']}")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask about remediation steps..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # response = get_rag_response(prompt, context=st.session_state.last_prediction)
    response = "Placeholder instruction text based on RAG retrieval."

    st.session_state.chat_history.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)
```

## Notes on this skeleton

- `st.session_state` is what makes the chat persist across reruns and lets the fault context flow from the Monitor page into the Chat page — this solves the gotcha I mentioned earlier
- The yellow "disagreement" case is handled explicitly in `get_status()` rather than being an afterthought
- `st.chat_message`/`st.chat_input` are Streamlit's built-in chat UI primitives — they'll make your RAG interface look native instead of a plain text box
- Swap the placeholder `run_rf`, `run_isolation_forest`, and `get_rag_response` calls for your actual model code

Want help next with the actual RAG retrieval function (FAISS + embeddings) that `get_rag_response` would call?