### Q1. How to handle ambiguous or missing medical data?
→ Use confidence thresholds on entity extraction and replace missing fields with “Not Available.”

### Q2. Which pretrained NLP models are used?
→ spaCy for NER, DistilBERT for sentiment, VADER for rule-based scoring, and optionally T5 for summarization.

### Q3. How to fine-tune BERT for medical sentiment?
→ Collect domain-specific datasets like `MedDialog` or `ClinicalBERT`, tokenize, and fine-tune using cross-entropy loss.

### Q4. How to train NLP model for SOAP mapping?
→ Use a hybrid of rule-based keyword detection and summarization models trained on structured EHR datasets.
