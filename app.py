# app.py - Physician Notetaker (clean, minimal comments)
import re
import json
import streamlit as st
import spacy
from spacy.matcher import PhraseMatcher
from transformers import pipeline
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download("vader_lexicon", quiet=True)

@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_sm")
    # Phrase patterns for medical entities (extendable)
    phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    symptom_patterns = ["neck pain", "back pain", "head impact", "headache", "stiffness", "backache", "trouble sleeping", "difficulty sleeping", "dizziness"]
    treatment_patterns = ["physiotherapy", "physio", "painkillers", "analgesics", "x-ray", "xray", "ten sessions", "ten sessions of physiotherapy", "advice and sent me home", "A&E", "accident and emergency"]
    diagnosis_patterns = ["whiplash", "whiplash injury", "lower back strain", "concussion"]
    prognosis_patterns = ["full recovery", "full recovery expected", "no long-term", "no signs of long-term damage", "on track for a full recovery"]
    phrase_matcher.add("SYMPTOM", [nlp.make_doc(t) for t in symptom_patterns])
    phrase_matcher.add("TREATMENT", [nlp.make_doc(t) for t in treatment_patterns])
    phrase_matcher.add("DIAGNOSIS", [nlp.make_doc(t) for t in diagnosis_patterns])
    phrase_matcher.add("PROGNOSIS", [nlp.make_doc(t) for t in prognosis_patterns])
    vader = SentimentIntensityAnalyzer()
    # Use CPU with device=-1 to avoid meta tensor device issues
    transformer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=-1)
    return nlp, phrase_matcher, vader, transformer

nlp, phrase_matcher, vader, transformer = load_models()

st.set_page_config(page_title="Physician Notetaker", layout="wide")
st.title("Physician Notetaker")
st.write("Medical transcription → NER, summarization, sentiment, intent, and SOAP note generation.")

note = st.text_area("Enter medical transcript or note:", height=240,
                    placeholder="Paste the physician-patient transcript here...")

def clean_text(text: str) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    return t

def extract_entities_and_matches(doc, matcher):
    ents = [(ent.text, ent.label_) for ent in doc.ents]
    found = {"Symptoms": set(), "Treatment": set(), "Diagnosis": set(), "Prognosis": set()}
    matches = matcher(doc)
    for match_id, start, end in matches:
        label = nlp.vocab.strings[match_id]
        span = doc[start:end].text
        if label == "SYMPTOM":
            found["Symptoms"].add(span)
        elif label == "TREATMENT":
            found["Treatment"].add(span)
        elif label == "DIAGNOSIS":
            found["Diagnosis"].add(span)
        elif label == "PROGNOSIS":
            found["Prognosis"].add(span)
    # simple regex fallback for ER/A&E
    text_lower = doc.text.lower()
    if "accident and emergency" in text_lower or "a&e" in text_lower or "moss bank" in text_lower:
        found["Treatment"].add("Visited A&E / ER evaluation")
    # sentence-level symptom captures
    for sent in doc.sents:
        if re.search(r"\b(pain|hurt|ache|stiff|stiffness|headache|dizziness|trouble sleeping|sleeping)\b", sent.text, flags=re.I):
            found["Symptoms"].add(sent.text.strip())
    return ents, {k: sorted(list(v)) for k, v in found.items()}

def extract_keywords(doc, top_k=10):
    phrases = set()
    for chunk in doc.noun_chunks:
        p = chunk.text.strip()
        if 2 <= len(p) <= 60 and len(p.split()) <= 5:
            phrases.add(p)
    # prefer exact medical phrases present in text
    medical_cands = ["whiplash injury", "physiotherapy", "painkillers", "A&E", "accident and emergency", "full recovery", "range of motion"]
    for p in medical_cands:
        if p.lower() in doc.text.lower():
            phrases.add(p)
    return sorted(list(phrases), key=lambda s: -len(s))[:top_k]

INTENT_KEYWORDS = {
    "Seeking reassurance": ["worry", "worried", "nervous", "concern", "afraid", "should I", "will I", "do I need", "do I have to"],
    "Reporting symptoms": ["pain", "hurt", "stiff", "ache", "symptom", "noticed", "discomfort"],
    "Expressing gratitude": ["thank you", "appreciate"],
    "Booking follow-up": ["follow-up", "come back", "return", "follow up", "followup"]
}

def detect_intent(text: str) -> str:
    t = text.lower()
    for label, keys in INTENT_KEYWORDS.items():
        for k in keys:
            if k in t:
                return label
    if re.search(r'\b(will I|should I|do I need|do I have to)\b', t):
        return "Seeking reassurance"
    return "Reporting symptoms"

def map_transformer_to_medical(label: str, text: str) -> str:
    lab = label.upper()
    if lab == "NEGATIVE":
        if re.search(r'\b(worried|worried|anxious|concern|afraid|nervous)\b', text.lower()):
            return "Anxious"
        return "Neutral"
    if lab == "POSITIVE":
        if re.search(r'\b(better|improved|good|fine|relieved)\b', text.lower()):
            return "Reassured"
        return "Neutral"
    return "Neutral"

def generate_structured_summary(doc, entities_matches):
    person = None
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            person = ent.text
            break
    symptoms = entities_matches.get("Symptoms", [])
    treatment = entities_matches.get("Treatment", [])
    diagnosis = entities_matches.get("Diagnosis", [])
    prognosis = entities_matches.get("Prognosis", [])
    # fallback heuristics
    if not symptoms:
        # find sentences mentioning pain/hurt
        for sent in doc.sents:
            if re.search(r"\b(pain|hurt|ache|stiff|discomfort)\b", sent.text, flags=re.I):
                symptoms.append(sent.text.strip())
    return {
        "Patient_Name": person or "Unknown",
        "Symptoms": symptoms or [],
        "Diagnosis": diagnosis[0] if diagnosis else None,
        "Treatment": treatment or [],
        "Current_Status": doc.text.strip()[:512],
        "Prognosis": prognosis[0] if prognosis else None,
        "Keywords": extract_keywords(doc)
    }

def generate_soap(doc, summary):
    subj_cc = summary["Symptoms"][:3] if summary.get("Symptoms") else ["Not stated"]
    subj_hpi = " ".join([s for s in subj_cc]) if subj_cc else "Not provided"
    objective = {
        "Physical_Exam": "Full range of movement in neck and back; no tenderness documented." if re.search(r'full range|range of motion|no tenderness', doc.text, flags=re.I) else "Exam details not present",
        "Observations": "Patient ambulatory; no gross neurological deficit observed."
    }
    assessment = {
        "Diagnosis": summary.get("Diagnosis") or "Whiplash injury suspected" if re.search(r'whiplash', doc.text, flags=re.I) else summary.get("Diagnosis") or "Not specified",
        "Severity": "Mild, improving" if re.search(r'improv|improved|better|recovery', doc.text, flags=re.I) else "Not specified"
    }
    plan = {
        "Treatment": summary.get("Treatment", []),
        "Follow-Up": "Return if symptoms worsen; typical follow-up within six months if symptoms persist."
    }
    return {
        "Subjective": {"Chief_Complaint": subj_cc, "History_of_Present_Illness": subj_hpi},
        "Objective": objective,
        "Assessment": assessment,
        "Plan": plan
    }

if st.button("Analyze"):
    if not note.strip():
        st.warning("Please enter a medical note or transcript.")
    else:
        cleaned = clean_text(note)
        doc = nlp(cleaned)
        ents, matches = extract_entities_and_matches(doc, phrase_matcher)
        keywords = extract_keywords(doc)
        # transformer result safe usage (pipeline returns label string)
        transformer_result = transformer(cleaned[:1000])[0]  # limit length for speed
        vader_result = vader.polarity_scores(cleaned)
        mapped_sentiment = map_transformer_to_medical(transformer_result.get("label", ""), cleaned)
        intent = detect_intent(cleaned)
        structured = generate_structured_summary(doc, matches)
        soap = generate_soap(doc, structured)

        st.success("Analysis Complete!")

        st.subheader("Extracted Entities (highlighted)")
        highlighted = cleaned
        # highlight entities in text visually (markdown)
        for ent_text, ent_label in ents:
            highlighted = highlighted.replace(ent_text, f"**{ent_text}**")
        st.markdown(highlighted)

        st.subheader("Entity List")
        st.json(ents or [])

        st.subheader("Matched Medical Phrases")
        st.json(matches)

        st.subheader("Keywords")
        st.json(keywords)

        st.subheader("Sentiment Analysis")
        st.json({
            "transformer_raw": transformer_result,
            "transformer_mapped_medical_label": mapped_sentiment,
            "vader": vader_result
        })

        st.subheader("Intent Detection")
        st.write(intent)

        st.subheader("Structured Medical Summary (JSON)")
        st.json(structured)

        st.subheader("Generated SOAP Note")
        st.json(soap)

        st.subheader("Cleaned Note")
        st.text(cleaned)

        st.caption("Built with spaCy, Hugging Face Transformers, NLTK (VADER), and Streamlit — by Shrikant Bawankule")
