import re
from pathlib import Path

import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

MODEL_DIR = str(Path(__file__).resolve().parent / "saved_summary_model")

# Must match training. If you trained with "summarize: " + dialogue, keep this.
# If you trained with the plain dialogue, set PREFIX = ""
PREFIX = "summarize: "

SAMPLE = """Amy: Hey, are you coming to the party tonight?
Ben: I'm not sure, I have an exam tomorrow.
Amy: Come on, just for an hour!
Ben: Okay, I'll come at 8 but leave by 9.
Amy: Great, see you then!"""

st.set_page_config(page_title="Chat Summarizer", page_icon="💬", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def clean_data(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n", "\n", text)
    return text.strip()


def summarize(text: str, num_beams: int, max_len: int) -> str:
    tokenizer, model, device = load_model()
    inputs = tokenizer(
        PREFIX + clean_data(text),
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_len,
            num_beams=num_beams,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
    return tokenizer.decode(ids[0], skip_special_tokens=True)


# ---------- UI ----------
st.title("Chat Summarizer")
st.write("Paste a conversation and get a short summary of what was said.")

with st.sidebar:
    st.header("Settings")
    num_beams = st.slider("Beams (higher = better, slower)", 1, 8, 4)
    max_len = st.slider("Max summary length (tokens)", 30, 150, 100)

if "dialogue" not in st.session_state:
    st.session_state.dialogue = ""

left, right = st.columns(2, gap="large")

with left:
    st.subheader("Conversation")
    st.text_area(
        "Conversation",
        key="dialogue",
        height=320,
        placeholder="Amy: Are you coming to the party tonight?\nBen: Okay, I'll come at 8 but leave by 9.",
        label_visibility="collapsed",
    )
    words = len(st.session_state.dialogue.split())
    st.caption(f"{words} words")

    b1, b2, b3 = st.columns([1, 1, 1])
    run = b1.button("Summarize", type="primary", use_container_width=True)
    b2.button(
        "Use sample",
        use_container_width=True,
        on_click=lambda: st.session_state.update(dialogue=SAMPLE),
    )
    b3.button(
        "Clear",
        use_container_width=True,
        on_click=lambda: st.session_state.update(dialogue=""),
    )

with right:
    st.subheader("Summary")
    if run:
        text = st.session_state.dialogue.strip()
        if not text:
            st.warning("Paste a conversation first, then select Summarize.")
        else:
            with st.spinner("Summarizing..."):
                result = summarize(text, num_beams, max_len)
            st.success(result)
            st.code(result, language=None)  # copy icon appears on hover
    else:
        st.info("Your summary will appear here.")