# 🩺 Physician Notetaker

An AI-powered NLP system that transforms physician–patient conversations into structured clinical documentation.
It performs **Named Entity Recognition**, **Medical Summarization**, **Sentiment & Intent Analysis**, and **SOAP Note Generation** — all in a clean **Streamlit** app.

---

## ⚙️ Setup

```bash
# 1️⃣ Clone the repository
git clone https://github.com/<your-username>/physician-notetaker.git
cd physician-notetaker

# 2️⃣ Create environment
conda create -n physician python=3.12 -y
conda activate physician

# 3️⃣ Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Then open: **[http://localhost:8501](http://localhost:8501)**

---

## 🧪 Example Input

```
Physician: Good morning, Ms. Jones. How are you feeling today?
Patient: I’m better, but still have some neck and back pain occasionally.
```

**Expected:**

* Extracts entities → “Ms. Jones”, “whiplash injury”
* Sentiment → *Reassured*
* Intent → *Reporting symptoms*
* Generates structured JSON + SOAP Note

---

## 👨‍💻 Author

**Shrikant Bawankule**
AI & Data Science Engineer | Healthcare NLP Enthusiast

---

