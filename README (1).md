# Chat Summarizer (Streamlit)

A web app that turns a chat conversation into a short summary. It uses a T5 model fine-tuned on the [SAMSum](https://huggingface.co/datasets/samsum) dialogue dataset, with a simple Streamlit interface.

**Example**

Input:

```
Amy: Hey, are you coming to the party tonight?
Ben: I'm not sure, I have an exam tomorrow.
Amy: Come on, just for an hour!
Ben: Okay, I'll come at 8 but leave by 9.
Amy: Great, see you then!
```

Output:

```
Ben will come to the party at 8 and leave by 9.
```

## Features

- Paste any conversation and get a summary with one click
- Live word counter
- **Use sample** button to try the app quickly
- Sidebar settings for beam search and summary length
- Copy button on the summary
- Model is loaded once and cached, so later summaries are fast
- Runs on GPU (CUDA), Apple Silicon (MPS), or CPU automatically

## Project structure

```
project/
├── streamlit_app.py        # the app
├── requirements.txt        # Python packages
├── README.md
└── saved_summary_model/    # your fine-tuned model (from training)
    ├── config.json
    ├── generation_config.json
    ├── model.safetensors
    ├── tokenizer_config.json
    └── ...
```

## Requirements

- Python 3.9 or newer (3.10 to 3.12 recommended)
- About 1 GB of free disk space
- No GPU needed, but a GPU makes summaries faster

## Setup

1. **Create and activate an environment** (conda example):

   ```bash
   conda create -n summarizer python=3.11 -y
   conda activate summarizer
   ```

2. **Install the packages:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Add your model.** Place the `saved_summary_model` folder next to `streamlit_app.py`.

## Run

```bash
python -m streamlit run streamlit_app.py
```

The app opens at **http://localhost:8501**. Keep the terminal open while you use it, and press `Ctrl + C` to stop.

On first launch the model takes a few seconds to load.

## How to use

1. Paste a conversation into the **Conversation** box, one message per line, in the form `Name: message`.
2. Click **Summarize**.
3. Read the result in the **Summary** panel, or copy it with the copy icon.

### Sidebar settings

| Setting | What it does |
|---|---|
| Beams | Number of candidate summaries the model compares. Higher is usually more accurate but slower. Default is 4. |
| Max summary length | Upper limit on summary length, in tokens. Default is 100. |

## How it works

1. The text is cleaned: line breaks are normalized, HTML tags and extra spaces are removed. Line breaks and capital letters are kept, because they help the model tell the speakers apart.
2. The prefix `summarize: ` is added in front of the text.
3. The T5 tokenizer converts the text to token ids (maximum 512 tokens, longer text is cut).
4. The model generates the summary with beam search.
5. The token ids are decoded back to text.

## Configuration

At the top of `streamlit_app.py`:

```python
MODEL_DIR = ".../saved_summary_model"   # where the model is loaded from
PREFIX = "summarize: "                  # must match your training setup
```

> **Important:** `PREFIX` must match how you trained the model. If you trained with `"summarize: " + dialogue`, keep it. If you trained on the plain dialogue, set `PREFIX = ""`. A mismatch lowers the summary quality.

## Model and training

- **Base model:** T5 (`t5-small`)
- **Dataset:** SAMSum (chat dialogues with human-written summaries)
- **Training:** fine-tuned with the Hugging Face `Trainer` on a sample of the training set, using a free Colab GPU
- **Metric:** ROUGE (ROUGE-1 around 40 or higher is decent for a small model)

## Known limitations

- A small model can **mix up who said what**, for example saying Amy will leave when Ben said it. Training on more data or for more epochs, or using a bigger model such as `flan-t5-base` or `facebook/bart-base`, improves this.
- It may add details that were not in the conversation (hallucination). Always check important summaries.
- It was trained on short English chats. Long conversations (more than about 400 words) are truncated, and other languages will not work well.

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'transformers'` | The wrong Python is active. Run `conda activate summarizer`, then `python -m streamlit run ...` |
| `Can't find ... saved_summary_model` | The model folder must be next to `streamlit_app.py` and named exactly `saved_summary_model` |
| `sentencepiece` error | `pip install sentencepiece` |
| Port 8501 is busy | `python -m streamlit run streamlit_app.py --server.port 8502` |
| Very slow summaries | Lower **Beams** in the sidebar, or run on a GPU |
| Page does not update after editing code | Click **Rerun** in the top right, or enable "Always rerun" |

## Deploy (optional)

The app can be hosted free on [Hugging Face Spaces](https://huggingface.co/spaces) with the Streamlit SDK:

1. Push your model to the Hub: `model.push_to_hub("your-username/samsum-t5")` and `tokenizer.push_to_hub(...)`.
2. Change `MODEL_DIR` in `streamlit_app.py` to `"your-username/samsum-t5"`.
3. Create a new Space (SDK: Streamlit) and upload `streamlit_app.py` and `requirements.txt`.

## Acknowledgements

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [SAMSum dataset](https://huggingface.co/datasets/samsum)
- [Streamlit](https://streamlit.io)
