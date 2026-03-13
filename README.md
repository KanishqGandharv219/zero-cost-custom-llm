# Building a Zero-Cost Custom LLM in 7 Days

This guide walks through designing and training an original (not fine-tuned) language model using only free tools (Google Colab, open libraries) and open datasets. We focus on an efficient "small" LLM architecture (e.g., a tiny GPT-like model) that can be trained within Colab's free-tier constraints and use primarily Indian-language data to position it as an "Indian-made" model. 

The outline encapsulates model architecture, tokenization, data selection, training pipeline, evaluation, deployment, and project setup, with practical tips and resources.

## 1. Architecture Design

Modern LLMs use transformer layers (self-attention + feed-forward) as their core. For a custom model, one might implement a small GPT-like transformer from scratch in PyTorch (embedding layer + positional encoding + N attention blocks + final output layer). The Raschka "LLMs from Scratch" code shows a step-by-step PyTorch implementation of a GPT-style model.

To make our model original, we can customize aspects like using fewer layers (e.g., 2–4 instead of 12+), modifying attention heads, or even mixing in a small recurrent or convolutional layer. However, the basic pattern – token embedding → positional encoding → Transformer blocks → softmax – provides a proven foundation. 

Our design should balance expressivity with Colab's limits: e.g., embedding size ~128–256, ~4–8 attention heads, and a handful of layers. This yields a "toy GPT" that still captures sequence structure. Using the free Colab GPU (~16 GB VRAM with a T4) is suited to "small-to-mid-sized Transformers", so we keep model size modest.

- **Custom elements:** To stand out, we could include features like RNN-based positional encoding or language-specific token embeddings, but core self-attention is recommended. Code can use PyTorch `nn.Module` to define the architecture (embedding → transformer blocks → softmax).
- **Libraries:** We can use open libraries (e.g., PyTorch or TensorFlow) for tensor operations, but write our own model classes instead of importing a pretrained model. This ensures true "from-scratch" training.
- **Initialization:** Randomly initialize all weights. We will not download any pretrained weights, only open-source code frameworks. The training process from scratch is essentially how big models are built internally (just on smaller data and compute).

## 2. Tokenization Strategy

We will use a subword tokenizer (e.g., Byte-Pair Encoding). A widely used choice is byte-level BPE, as in GPT-2/3, which avoids unknown tokens by using 256 byte values as base. In practice, we can use Hugging Face's `tokenizers` or the `tiktoken` library to build/train a BPE tokenizer on our corpus.

- **BPE basics:** Start with all single characters and iteratively merge the most frequent pairs until reaching the vocabulary size (e.g., 30K–50K tokens). GPT-2 used byte-level BPE with a 50K vocab. We can target a similar scale or even smaller (e.g., 10–20K) given limited data/GPU.
- **Implementation:** Use a Python library to train BPE. For example, the `tokenizers` library (HuggingFace) or OpenAI's `tiktoken` can create a GPT-compatible encoder. In Colab we could install it via `!pip install tiktoken` and then do:
  ```python
  import tiktoken
  enc = tiktoken.get_encoding("gpt2")  # or train your own
  tokens = enc.encode_ordinary("some input text")
  ```
  We can either load the GPT-2 BPE or train on our combined dataset (English + Indic). Training our own ensures it reflects Indian text patterns.
- **Vocabulary:** Aim for a vocabulary size that fits Colab memory. Byte-level BPE with ~30–50K tokens is common. If including Indic scripts, ensure the tokenizer covers Devanagari/Bengali scripts.
- **Storage:** Tokenize all text into integer IDs and save as binary (e.g., `.bin` files) for efficient loading. (e.g. using `numpy.memmap` to store token IDs to disk and handle large corpora).

## 3. Dataset Selection

We use entirely open, free text corpora. To emphasize the "Indian-made" aspect, we will include lots of Indian-language text alongside general data. Key sources:

- **OpenWebText and WikiText:** For general text, use the OpenWebText corpus (open-source version of GPT-2's WebText), and WikiText-103 (100M tokens from Wikipedia). These are CC0-licensed or equivalent and well-suited for pretraining.
- **OSCAR (multilingual):** The OSCAR corpus is a huge multilingual Common Crawl dataset. It contains 166 languages, with each language split. We can download the relevant languages (e.g., Hindi, Marathi, etc.) from the Hugging Face OSCAR dataset or its releases. OSCAR is explicitly intended to pretrain language models.
- **Indic corpora (monolingual):** The AI4Bharat IndicNLP resources list many Indian corpora. For example, IIT Bombay's Hindi monolingual corpus, Charles Univ. Hindi/Urdu corpora, and the EMILLE corpora (multi-language). Ensure license compliance (all above are CC0/CC-BY).
- **Wikipedia dumps:** Download latest Wikipedia dumps (English and Hindi, etc.) from dumps.wikimedia.org. These raw XML dumps can be cleaned easily.

Combine these to get tens of millions of lines, sampling or truncating corpora as needed for free GPU limits.

## 4. Tokenization & Data Pipeline

With text collected, we preprocess and tokenize in Colab:

- **Text cleaning:** Remove markup, convert to UTF-8, normalize whitespace. If mixing languages, consider normalizing scripts. Keep things in `.txt` or HuggingFace Datasets format.
- **Tokenizer training:** Train BPE on your combined text using one of the libraries mentioned. Save the learned vocab and merge rules.
- **Tokenize the corpora:** Run the tokenizer over all data splits. We should split off a validation set (e.g., 5–10%). Store token IDs for train/val as `train.bin`, `valid.bin` sequentially using `numpy.memmap`.
- **Batch preparation:** Prepare input-target pairs (next-token prediction). Split the sequence into chunks of fixed length (e.g., 128 tokens) and for each chunk, the target is the same tokens shifted by 1.

## 5. Training Pipeline

On Colab's free GPU (T4, ~16GB VRAM), training must be efficient:

- **Environment:** In Colab, install necessary Python libs (`torch`, `torchvision`, `tiktoken`, `numpy`) and use GPU runtime. Check VRAM with `!nvidia-smi`.
- **Model code:** Implement the model class (e.g., `TransformerLM`). Define the forward pass: token IDs → embeddings → transformer blocks → logits.
- **Training loop:** Iterate epochs, for each batch compute loss, backpropagate, and perform optimizer step. 
  - Due to GPU limits, use a small batch size (maybe 4–8) and gradient accumulation if needed. 
  - Use Adam optimizer. Optionally enable `torch.cuda.amp` (16-bit) for speed.
  ```python
  for epoch in range(num_epochs):
      model.train()
      for X, Y in train_loader:
          optimizer.zero_grad()
          logits = model(X)
          loss = criterion(logits.view(-1, vocab_size), Y.view(-1))
          loss.backward()
          optimizer.step()
      # evaluate on validation
  ```
- **Checkpoints:** Regularly save model weights to Google Drive to avoid losing progress on Colab timeout (~12h).
- **Monitoring:** Track training/validation loss (or perplexity) each epoch.

## 6. Evaluation

After training, evaluate the model's performance:

- **Perplexity:** Compute perplexity on held-out validation data (exp(average negative log-likelihood)). Lower is better. Calculate token-level loss on the validation set.
- **Qualitative tests:** Have the model generate text via a prompt (English or Hindi) and do greedy/beam sampling. See if generated text is grammatical or domain-appropriate.
- **Overfitting check:** Ensure the model isn't just memorizing by comparing train vs val loss. If overfitting, consider more data or regularization.
- **Downstream tasks (optional):** Fine-tune or test on a small task (e.g., text classification or QA) to gauge usefulness.

## 7. Deployment (Demo/API)

To showcase the model, we can create a simple interface:

- **Web UI with Gradio:** Gradio is an open-source library to make quick ML demos in seconds.
  ```python
  import gradio as gr

  def generate_text(prompt, length=50):
      tokens = model.tokenizer.encode(prompt)
      out = model.generate(tokens, max_new_tokens=length)
      return model.tokenizer.decode(out)

  demo = gr.Interface(fn=generate_text, inputs=["text", "slider"], outputs="text", api_name="predict")
  demo.launch() 
  ```
- **Alternative – Flask/FastAPI with ngrok:** Write a small Flask or FastAPI server that wraps the predict function, tunneling the local Colab server to a public URL with ngrok.

## 8. Project Structure & Documentation

Organize the code and docs clearly for GitHub display:

```
├── README.md              # Project overview, instructions, dataset links, model details
├── LICENSE                # Open license (e.g., MIT or CC0)
├── .gitignore             # Ignore data, models, etc.
├── requirements.txt       # Python libraries
├── data/                  # Scripts/instructions to download/process data (raw/, processed/)
├── src/                   # Source code (e.g., model.py, train.py, tokenize.py)
├── notebooks/             # Colab notebooks: 1_data_prep.ipynb, 2_train_model.ipynb, etc.
├── models/                # Saved checkpoints
├── demo/                  # Deployment code
└── docs/                  # Optional documentation
```

## 9. Day 2 MVP Command Flow

Day 2 is now wired as a runnable pipeline in `src/day2_data.py`:

```bash
# Run full Day 2 flow: ingest -> clean -> merge/split -> report
python -m src.day2_data run --max-samples-per-source 5000 --valid-ratio 0.05 --seed 42

# Optional staged runs
python -m src.day2_data ingest --max-samples-per-source 5000
python -m src.day2_data process --valid-ratio 0.05 --seed 42

# Offline/local validation run (no downloads)
python -m src.day2_data smoke --valid-ratio 0.05 --seed 42
```

Expected artifacts:

- `data/raw/dataset_manifest.json` (source metadata: source/license/language/split/sample_size)
- `data/raw/<source_id>/raw.txt` (raw text lines per source)
- `data/processed/intermediate/<source_id>.txt` (cleaned text per source)
- `data/processed/merged_corpus.txt` (all cleaned sources merged)
- `data/processed/train.txt` and `data/processed/valid.txt` (deterministic split)
- `data/processed/sample_100k_lines.txt` (smoke-test sample)
- `docs/day2_data_report.md` (quality report: counts, approx tokens, source mix, duplicate checks)

## 10. 7-Day Implementation Timeline

- **Day 1 – Setup & Architecture Design:** Set up Colab, environment, GitHub repo. Decide model hyperparameters and basic architecture. Implement a model class template.
- **Day 2 – Data Collection:** Download selected datasets. Write scripts to clean and merge text. Prepare a small sample dataset to test tokenization.
- **Day 3 – Tokenizer:** Train the BPE tokenizer on collected text. Save vocab and merge files. Generate `train.bin` and `valid.bin`.
- **Day 4 – Model & Training Code:** Finalize PyTorch model code. Write and verify the training loop on the sample data. Integrate data loading. Set up logging.
- **Day 5 – Train Small Model:** Launch training on full data for several epochs. Monitor and adjust hyperparameters. Save checkpoints.
- **Day 6 – Evaluation and Iteration:** Evaluate final model. If results are poor, iterate on data or training. Prepare output samples.
- **Day 7 – Deployment & Documentation:** Build the Gradio demo in Colab. Polish the GitHub repo (README, organization, license). Final review of the project.
