# AHP-ISO

Semplice applicazione Streamlit per Analytic Hierarchy Process (AHP) con integrazione di validazione ISO/IEC 27005 (ex-post).

Funzionalità principali:

- Importa dataset alternative x criteri (CSV/Excel).
- Raccoglie confronti pairwise multi-esperto per i criteri.
- Aggrega giudizi esperti con media geometrica e calcola i pesi (metodo di Saaty).
- Valida risultati con un Risk Register ISO/IEC 27005 (opzionale) e combina ranking.
- Esporta risultati in Excel/CSV.

Requisiti

- Python 3.10+ (testato con 3.11/3.12)
- Raccomandati: creare un virtualenv e installare le dipendenze in `requirements.txt`.

Installazione

1. Creare un virtualenv e attivarlo:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Installare dipendenze:

```bash
pip install -r requirements.txt
```

Eseguire

```bash
streamlit run app.py
```

Licenza

Questo progetto è rilasciato sotto licenza MIT. Se usi questo software in lavori pubblici o in prodotti derivati, ti preghiamo di citare l'autore come indicato nella sezione seguente.

Citazione

Se usi questo software in una pubblicazione, includi la seguente citazione e contatta l'autore se necessario:

Christian Riccio (2026). AHP-ISO. Università degli Studi della Campania Luigi Vanvitelli. Disponibile online: https://ahpiso.streamlit.app/

Contatto email: christian.riccio@unicampania.it

