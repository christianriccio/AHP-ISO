import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import csv
import json
import math


# =========================================================
# FUNZIONI DI SUPPORTO I/O E STATO
# =========================================================

def init_session_state():
    """
    Inizializza lo stato di sessione se non è già presente.
    Questo serve per:
    - Backup/ripristino (JSON)
    """
    if "backup_state" not in st.session_state:
        st.session_state.backup_state = {
            "objective": "",
            "criteria": [],
            "alternatives": [],
            "performance_matrix": {},  # {alt: {crit: value}}
            "num_interviews": 1,
            "interviews": {},          # "0": {"pairwise": {...}}
        }


def load_alternatives_file(uploaded_file):
    """
    Caricamento generico di un file tabellare (CSV/Excel).
    Supporta CSV con separatore autodetect, e Excel.
    """
    if uploaded_file is None:
        return None

    filename = uploaded_file.name.lower()

    # Excel
    if filename.endswith('.xlsx') or filename.endswith('.xls'):
        try:
            df = pd.read_excel(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Errore lettura Excel: {e}")
            return None

    # CSV con separatore sconosciuto
    try:
        df = pd.read_csv(uploaded_file, sep=None, engine='python')
        return df
    except Exception as e:
        st.warning(f"Tentativo autodetect CSV fallito: {e}")
        try:
            uploaded_file.seek(0)
            rawdata = uploaded_file.read(2048).decode('utf-8', errors='replace')
            uploaded_file.seek(0)
            dialect = csv.Sniffer().sniff(rawdata, delimiters=[',',';','\t','|'])
            sep = dialect.delimiter
            df = pd.read_csv(uploaded_file, sep=sep)
            return df
        except Exception as e2:
            st.error(f"Tentativo csv.Sniffer fallito: {e2}")
            return None


def download_json_button(label, data_dict, filename):
    """
    Bottone per scaricare un backup JSON dell'intero stato dell'esperimento.
    """
    json_bytes = json.dumps(data_dict, indent=2).encode('utf-8')
    st.download_button(
        label=label,
        data=json_bytes,
        file_name=filename,
        mime='application/json'
    )


def try_load_backup_json(uploaded_json):
    """
    Carica un backup JSON precedentemente salvato.
    Restituisce il dict o None se fallisce.
    """
    try:
        raw = uploaded_json.read()
        data = json.loads(raw.decode('utf-8'))
        return data
    except Exception as e:
        st.error(f"Backup JSON non valido: {e}")
        return None


def matrix_to_dict(df):
    """Converte un DataFrame in dict serializzabile."""
    return {
        "index": list(df.index),
        "columns": list(df.columns),
        "data": df.values.tolist()
    }


def dict_to_matrix(d):
    """Converte dict serializzato da backup in DataFrame."""
    return pd.DataFrame(d["data"], index=d["index"], columns=d["columns"])


def save_current_state_to_session(objective,
                                  criteria_list,
                                  alternatives_list,
                                  perf_df,
                                  num_interviews,
                                  interview_matrices):
    """
    Aggiorna lo stato di sessione con:
    - obiettivo decisionale
    - lista criteri
    - lista alternative
    - matrice prestazioni alternative x criteri
    - tutte le matrici di confronto raccolte finora
    """
    st.session_state.backup_state["objective"] = objective
    st.session_state.backup_state["criteria"] = criteria_list
    st.session_state.backup_state["alternatives"] = alternatives_list

    perf_dict = {}
    for alt in alternatives_list:
        perf_dict[alt] = {}
        for crit in criteria_list:
            perf_dict[alt][crit] = float(perf_df.loc[alt, crit])
    st.session_state.backup_state["performance_matrix"] = perf_dict

    st.session_state.backup_state["num_interviews"] = num_interviews

    interviews_dict = {}
    for k, dfmat in interview_matrices.items():
        interviews_dict[str(k)] = {
            "pairwise": matrix_to_dict(dfmat)
        }
    st.session_state.backup_state["interviews"] = interviews_dict


def load_state_from_backup(backup_dict):
    """
    Ripristina i dati dal backup JSON.
    Restituisce tuple utili per popolazione dell'interfaccia.
    """
    try:
        objective = backup_dict["objective"]
        criteria_list = backup_dict["criteria"]
        alternatives_list = backup_dict["alternatives"]
        perf_mat_dict = backup_dict["performance_matrix"]
        num_interviews = backup_dict["num_interviews"]

        # ricostruisci performance DataFrame
        perf_df = pd.DataFrame.from_dict(perf_mat_dict, orient="index")
        perf_df = perf_df[criteria_list]  # garantisce l'ordine delle colonne

        # ricostruisci interviste
        interview_matrices = {}
        for key, sub in backup_dict["interviews"].items():
            interview_matrices[int(key)] = dict_to_matrix(sub["pairwise"])

        st.session_state.backup_state = backup_dict

        return (objective,
                criteria_list,
                alternatives_list,
                perf_df,
                num_interviews,
                interview_matrices)

    except Exception as e:
        st.error(f"Errore nel ripristino del backup: {e}")
        return None


# =========================================================
# FUNZIONI CORE AHP
# =========================================================

def create_empty_pairwise_matrix(elements):
    """
    Crea una matrice n x n con 1 sulla diagonale.
    """
    n = len(elements)
    mat = np.ones((n, n), dtype=float)
    return pd.DataFrame(mat, index=elements, columns=elements)


def saaty_scale_description():
    """
    Scala di Saaty 1..9 (descrizioni sintetiche).
    """
    return {
        1: "Uguale importanza",
        2: "Tra uguale e moderata",
        3: "Importanza moderata",
        4: "Tra moderata e forte",
        5: "Importanza forte",
        6: "Tra forte e molto forte",
        7: "Importanza molto forte",
        8: "Tra molto forte ed estrema",
        9: "Importanza estrema"
    }


def calculate_priority_vector(pairwise_matrix: pd.DataFrame):
    """
    Priority vector dei criteri:
    media delle righe dopo normalizzazione per colonna (metodo classico Saaty approx).
    """
    A = pairwise_matrix.values.astype(float)
    col_sum = A.sum(axis=0)
    norm = A / col_sum
    w = norm.mean(axis=1)
    return w  # numpy array di lunghezza n


def calculate_consistency_ratio(pairwise_matrix: pd.DataFrame, weights: np.ndarray):
    """
    Calcolo di λ_max, CI e CR per la matrice AHP.
    """
    A = pairwise_matrix.values.astype(float)
    n = A.shape[0]

    Aw = A @ weights
    lambda_max = np.mean(Aw / weights)

    CI = (lambda_max - n) / (n - 1) if n > 1 else 0.0

    RI_table = {
        1: 0.00,
        2: 0.00,
        3: 0.58,
        4: 0.90,
        5: 1.12,
        6: 1.24,
        7: 1.32,
        8: 1.41,
        9: 1.45,
        10: 1.49
    }
    RI = RI_table.get(n, 1.49)
    if n <= 2:
        CR = 0.0
    else:
        CR = CI / RI if RI != 0 else 0.0

    return CR, CI, lambda_max


def geometric_mean(values):
    """
    Media geometrica di valori positivi.
    """
    arr = np.array(values, dtype=float)
    if np.any(arr <= 0):
        return 0.0
    prod = np.prod(arr)
    return prod ** (1.0 / len(arr))


def aggregate_experts_geometric_mean(list_of_matrices):
    """
    Aggregazione delle matrici di confronto dei vari esperti
    tramite media geometrica elemento-per-elemento.
    (Group AHP standard)
    """
    if len(list_of_matrices) == 1:
        return list_of_matrices[0].copy()

    idx = list_of_matrices[0].index
    cols = list_of_matrices[0].columns
    n = len(idx)

    agg = np.ones((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            vals = [m.iloc[i, j] for m in list_of_matrices]
            agg[i, j] = geometric_mean(vals)

    return pd.DataFrame(agg, index=idx, columns=cols)


# =========================================================
# ISO/IEC 27005 - FUNZIONI (VALIDAZIONE EX-POST)
# =========================================================

def _pick_column(df_cols_lower_map, *candidates):
    for c in candidates:
        if c in df_cols_lower_map:
            return df_cols_lower_map[c]
    return None


def standardize_iso_risk_register(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizza colonne ISO risk register e valida i minimi necessari.

    Colonne supportate (case-insensitive):
    - Alternative: alternative | dlm | alt
    - Likelihood: likelihood | l
    - Impact: impact (opzionale se hai CIA)
    - Impact_C: impact_c | c | impact_confidentiality
    - Impact_I: impact_i | i | impact_integrity
    - Impact_A: impact_a | a | impact_availability
    - Control_Effectiveness: control_effectiveness | control_eff | mitigation | controls

    Restituisce un DF con colonne canonical:
    alternative, likelihood, impact_C, impact_I, impact_A, control_effectiveness
    (oppure impact se già presente; in tal caso CIA possono mancare).
    """
    if df_raw is None or df_raw.empty:
        return None

    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lower_map = {str(c).strip().lower(): str(c).strip() for c in df.columns}

    alt_col = _pick_column(lower_map, "alternative", "dlm", "alt")
    lik_col = _pick_column(lower_map, "likelihood", "l")

    impact_col = _pick_column(lower_map, "impact")
    ic_col = _pick_column(lower_map, "impact_c", "impact_confidentiality", "c")
    ii_col = _pick_column(lower_map, "impact_i", "impact_integrity", "i")
    ia_col = _pick_column(lower_map, "impact_a", "impact_availability", "a")

    ce_col = _pick_column(lower_map,
                          "control_effectiveness", "control_eff", "controls_effectiveness",
                          "mitigation", "controls")

    missing = []
    if alt_col is None: missing.append("Alternative (alternative|dlm|alt)")
    if lik_col is None: missing.append("Likelihood (likelihood|l)")
    if ce_col is None: missing.append("Control_Effectiveness (control_effectiveness|control_eff|mitigation|controls)")

    # Impact: o impact diretto o tripla CIA
    has_impact_direct = (impact_col is not None)
    has_cia = (ic_col is not None and ii_col is not None and ia_col is not None)

    if not has_impact_direct and not has_cia:
        missing.append("Impact: (impact) oppure (impact_C, impact_I, impact_A)")

    if missing:
        raise ValueError("Colonne mancanti nel risk register ISO: " + "; ".join(missing))

    out = pd.DataFrame()
    out["alternative"] = df[alt_col].astype(str).str.strip()
    out["likelihood"] = pd.to_numeric(df[lik_col], errors="coerce")

    if has_impact_direct:
        out["impact"] = pd.to_numeric(df[impact_col], errors="coerce")
        out["impact_C"] = np.nan
        out["impact_I"] = np.nan
        out["impact_A"] = np.nan
    else:
        out["impact_C"] = pd.to_numeric(df[ic_col], errors="coerce")
        out["impact_I"] = pd.to_numeric(df[ii_col], errors="coerce")
        out["impact_A"] = pd.to_numeric(df[ia_col], errors="coerce")
        out["impact"] = np.nan

    out["control_effectiveness"] = pd.to_numeric(df[ce_col], errors="coerce")

    out = out.dropna(subset=["alternative", "likelihood", "control_effectiveness"])

    return out


def compute_iso27005_metrics(iso_df: pd.DataFrame,
                            threshold_residual: float,
                            w_c: float, w_i: float, w_a: float,
                            risk_max_theoretical: float = 25.0) -> (pd.DataFrame, pd.DataFrame):
    """
    Calcola:
    - risk_inherent = likelihood * impact
    - risk_residual = risk_inherent * (1 - control_effectiveness)

    Aggrega per alternativa:
    residual_max, residual_mean, frac_above_threshold, iso_pass, iso_score (1 - norm)

    Restituisce:
    - iso_register_scored: DF scenario-level con risk_inherent/residual
    - iso_metrics: DF alternativa-level con metriche e ranking ISO
    """
    df = iso_df.copy()

    # Impact aggregato: usa impact diretto se presente, altrimenti CIA pesato
    if df["impact"].notna().any():
        df["impact_eff"] = df["impact"]
    else:
        denom = (w_c + w_i + w_a)
        if denom <= 0:
            raise ValueError("Pesi CIA non validi (somma <= 0).")
        df["impact_eff"] = (w_c * df["impact_C"] + w_i * df["impact_I"] + w_a * df["impact_A"]) / denom

    df["likelihood"] = pd.to_numeric(df["likelihood"], errors="coerce")
    df["impact_eff"] = pd.to_numeric(df["impact_eff"], errors="coerce")
    df["control_effectiveness"] = np.clip(pd.to_numeric(df["control_effectiveness"], errors="coerce"), 0.0, 1.0)

    df = df.dropna(subset=["likelihood", "impact_eff", "control_effectiveness"])

    df["risk_inherent"] = df["likelihood"] * df["impact_eff"]
    df["risk_residual"] = df["risk_inherent"] * (1.0 - df["control_effectiveness"])

    iso_metrics = df.groupby("alternative", as_index=False).agg(
        residual_max=("risk_residual", "max"),
        residual_mean=("risk_residual", "mean"),
        n_rows=("risk_residual", "count"),
        frac_above_thr=("risk_residual", lambda x: float((x > threshold_residual).mean()))
    )

    iso_metrics["risk_norm"] = np.clip(iso_metrics["residual_max"] / float(risk_max_theoretical), 0.0, 1.0)
    iso_metrics["iso_pass"] = iso_metrics["residual_max"] <= float(threshold_residual)
    iso_metrics["iso_score"] = 1.0 - iso_metrics["risk_norm"]

    iso_metrics = iso_metrics.sort_values("iso_score", ascending=False).reset_index(drop=True)
    iso_metrics["iso_rank"] = np.arange(1, len(iso_metrics) + 1)

    return df, iso_metrics


def combine_ahp_iso(result_ranked: pd.DataFrame,
                    iso_metrics: pd.DataFrame,
                    mode: str,
                    threshold_residual: float,
                    alpha: float) -> pd.DataFrame:
    """
    Unisce ranking AHP con metriche ISO e produce ranking combinato.

    mode:
      - "gate": score=FinalScore se iso_pass, altrimenti 0
      - "penalty": score=FinalScore*(1-alpha*risk_norm)
    """
    df = result_ranked.copy()
    df = df.merge(iso_metrics, left_on="Alternative", right_on="alternative", how="left")

    # se manca ISO per una alternativa, considerala "non pass" (conservative)
    df["iso_pass"] = df["iso_pass"].fillna(False)
    df["risk_norm"] = df["risk_norm"].fillna(1.0)
    df["iso_score"] = df["iso_score"].fillna(0.0)

    mode = str(mode).strip().lower()
    if mode == "gate":
        df["FinalScore_ISO"] = np.where(df["iso_pass"], df["FinalScore"], 0.0)
    elif mode == "penalty":
        df["FinalScore_ISO"] = df["FinalScore"] * np.clip(1.0 - float(alpha) * df["risk_norm"], 0.0, 1.0)
    else:
        raise ValueError("mode deve essere 'gate' o 'penalty'.")

    df = df.sort_values("FinalScore_ISO", ascending=False).reset_index(drop=True)
    df["Rank_ISO"] = np.arange(1, len(df) + 1)

    return df


# =========================================================
# FUNZIONI DI VISUALIZZAZIONE
# =========================================================

def plot_radar_chart(criteria_list, data_rows, labels):
    """
    Radar chart per confrontare alternative sui criteri normalizzati.
    """
    N = len(criteria_list)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))

    for i, rowvals in enumerate(data_rows):
        vals = list(rowvals) + [rowvals[0]]
        ax.plot(angles, vals, linewidth=2, label=labels[i])
        ax.fill(angles, vals, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(criteria_list, fontsize=9)
    ax.set_yticks([])
    ax.legend(loc="upper right", bbox_to_anchor=(1.2,1.1), fontsize=9)

    st.pyplot(fig)


def plot_bar_ranking(df_ranked, alt_col_name, score_col_name):
    """
    Bar chart delle alternative ordinate per punteggio finale.
    """
    fig, ax = plt.subplots(figsize=(7,4))
    names = df_ranked[alt_col_name].tolist()
    scores = df_ranked[score_col_name].tolist()

    bars = ax.bar(names, scores)
    ax.set_ylabel("Score")
    ax.set_title(f"Ranking ({score_col_name})")
    ax.grid(axis='y', alpha=0.3)

    for b, s in zip(bars, scores):
        h = b.get_height()
        ax.text(b.get_x()+b.get_width()/2., h, f"{s:.3f}",
                ha='center', va='bottom', fontsize=8)

    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    st.pyplot(fig)


# =========================================================
# APP STREAMLIT
# =========================================================

def main():
    st.set_page_config(
        page_title="AHP-ISO",
        layout="wide",
        page_icon="📊"
    )

    init_session_state()

    title_col, info_col = st.columns([0.7, 0.3])
    with title_col:
        st.markdown("### AHP - ISO")
        st.caption("Valutazione multi-criterio con supporto multi-esperto + validazione ISO/IEC 27005 (ex-post)")

    

    st.divider()

    # =====================================================
    # STEP 1. OBIETTIVO
    # =====================================================
    st.markdown("#### 1. Obiettivo")
    objective = st.text_input(
        "Obiettivo della decisione",
        value=st.session_state.backup_state.get("objective", ""),
        placeholder="Esempio: Selezionare il miglior DLM"
    )

    st.divider()

    # =====================================================
    # STEP 2. INPUT: AHP (alternatives/criteria) + ISO risk register (opzionale)
    # =====================================================
    st.markdown("#### 2. Dati di input (Alternative/Criteri + ISO Risk Register opzionale)")

    col_upload, col_backup, col_iso = st.columns(3)

    with col_upload:
        st.write("Carica file con alternative e criteri (CSV o Excel).")
        st.caption(
            "- Prima colonna: nome alternativa (DLM)\n"
            "- Colonne successive: criteri numerici"
        )
        uploaded_data = st.file_uploader(
            "Carica dati AHP",
            type=["csv", "xlsx", "xls"],
            key="uploaded_data_file"
        )

    with col_backup:
        st.write("Oppure riprendi un backup JSON completo.")
        uploaded_backup = st.file_uploader(
            "Carica backup JSON",
            type=["json"],
            key="uploaded_backup_file"
        )

    with col_iso:
        st.write("Carica ISO Risk Register (opzionale).")
        st.caption(
            "Colonne minime: Alternative, Likelihood, Impact_C, Impact_I, Impact_A, Control_Effectiveness.\n"
            "In alternativa: Alternative, Likelihood, Impact, Control_Effectiveness."
        )
        uploaded_iso = st.file_uploader(
            "Carica Risk Register ISO",
            type=["csv", "xlsx", "xls"],
            key="uploaded_iso_file"
        )

    restored = None
    if uploaded_backup is not None:
        backup_dict = try_load_backup_json(uploaded_backup)
        if backup_dict is not None:
            restored = load_state_from_backup(backup_dict)
            st.success("Backup caricato. Interfaccia precompilata con i dati salvati.")

    if restored is not None:
        (objective_rest,
         criteria_list_rest,
         alternatives_list_rest,
         perf_df_rest,
         num_interviews_rest,
         interview_matrices_rest) = restored

        if not objective:
            objective = objective_rest
    else:
        criteria_list_rest = []
        alternatives_list_rest = []
        perf_df_rest = None
        num_interviews_rest = st.session_state.backup_state.get("num_interviews", 1)
        interview_matrices_rest = {}

    # AHP data
    if uploaded_data is not None:
        df_raw = load_alternatives_file(uploaded_data)
        if df_raw is None or df_raw.empty:
            st.error("File AHP vuoto/non valido.")
            st.stop()
    elif perf_df_rest is not None:
        df_raw = perf_df_rest.reset_index().rename(columns={"index": "Alternative"})
    else:
        df_raw = None

    # ISO risk register
    iso_df_std = None
    iso_df_raw = None
    if uploaded_iso is not None:
        iso_df_raw = load_alternatives_file(uploaded_iso)
        if iso_df_raw is None or iso_df_raw.empty:
            st.error("File ISO Risk Register vuoto/non valido.")
            st.stop()
        try:
            iso_df_std = standardize_iso_risk_register(iso_df_raw)
        except Exception as e:
            st.error(f"Formato ISO Risk Register non valido: {e}")
            iso_df_std = None

    if df_raw is not None:
        st.markdown("**Anteprima dati AHP caricati**")
        st.dataframe(df_raw.head(), use_container_width=True)

        alt_col_name = df_raw.columns[0]
        alternatives_list = df_raw[alt_col_name].astype(str).tolist()
        criteria_list = list(df_raw.columns[1:])
        perf_df = df_raw.set_index(alt_col_name).astype(float)

        st.caption(f"Alternative rilevate: {alternatives_list}")
        st.caption(f"Criteri rilevati: {criteria_list}")
    else:
        alternatives_list = alternatives_list_rest
        criteria_list = criteria_list_rest
        perf_df = perf_df_rest if perf_df_rest is not None else pd.DataFrame()

    if iso_df_std is not None:
        st.markdown("**Anteprima ISO Risk Register (standardizzato)**")
        st.dataframe(iso_df_std.head(), use_container_width=True)

        # sanity check: alternative ISO ⊆ alternative AHP (non obbligatorio, ma utile)
        missing_iso_alts = sorted(set(alternatives_list) - set(iso_df_std["alternative"].unique().tolist()))
        if missing_iso_alts:
            st.warning(
                "Nel risk register ISO mancano alcune alternative presenti in AHP. "
                "Quelle alternative verranno trattate come 'non pass' (conservative): "
                + ", ".join(missing_iso_alts[:10]) + ("..." if len(missing_iso_alts) > 10 else "")
            )

    if len(criteria_list) == 0 or len(alternatives_list) == 0:
        st.warning("Carica un file dati o un backup per continuare con i passi successivi.")
        return

    st.divider()

    # =====================================================
    # STEP 3. NUMERO DI ESPERTI
    # =====================================================
    st.markdown("#### 3. Esperti / Interviste")
    num_interviews = st.number_input(
        "Numero di esperti/interviste",
        min_value=1,
        max_value=20,
        step=1,
        value=num_interviews_rest,
        help="Per ogni esperto raccoglieremo una matrice di confronto tra i criteri."
    )

    st.divider()

    # =====================================================
    # STEP 4. PAIRWISE PER ESPERTO
    # =====================================================
    st.markdown("#### 4. Confronti tra criteri (pairwise)")

    n_crit = len(criteria_list)

    interview_matrices = {}
    if len(interview_matrices_rest) > 0:
        for k, dfmat in interview_matrices_rest.items():
            interview_matrices[k] = dfmat.copy()

    for interview_id in range(num_interviews):
        st.markdown(f"**Esperto #{interview_id + 1}**")

        if interview_id in interview_matrices:
            pairwise_df = interview_matrices[interview_id]
        else:
            pairwise_df = create_empty_pairwise_matrix(criteria_list)

        with st.expander(f"Confronti criteri – Esperto #{interview_id + 1}", expanded=True):
            for i in range(n_crit):
                for j in range(i+1, n_crit):
                    left = criteria_list[i]
                    right = criteria_list[j]

                    st.write(f"{left} ↔ {right}")

                    pref = st.radio(
                        "Quale criterio è più importante?",
                        options=[left, "Uguali", right],
                        index=1,
                        key=f"pref-{interview_id}-{i}-{j}"
                    )

                    default_val = float(pairwise_df.loc[left, right])
                    if default_val < 1:
                        default_slider_val = 1
                    else:
                        default_slider_val = int(np.clip(round(default_val), 1, 9))

                    intensity = st.slider(
                        "Intensità importanza (1 = uguale, 9 = estremamente più importante)",
                        min_value=1,
                        max_value=9,
                        value=default_slider_val,
                        step=1,
                        key=f"intensity-{interview_id}-{i}-{j}"
                    )

                    if pref == "Uguali" or intensity == 1:
                        val = 1.0
                    elif pref == left:
                        val = float(intensity)
                    else:
                        val = 1.0 / float(intensity)

                    pairwise_df.loc[left, right] = val
                    pairwise_df.loc[right, left] = 1.0 / val
                    pairwise_df.loc[left, left] = 1.0
                    pairwise_df.loc[right, right] = 1.0

            st.caption("Matrice di confronto (criteri) per questo esperto")
            st.dataframe(pairwise_df.style.format("{:.4f}"), use_container_width=True)

        interview_matrices[interview_id] = pairwise_df

    st.divider()

    # =====================================================
    # STEP 5. BACKUP
    # =====================================================
    st.markdown("#### 5. Backup stato (facoltativo)")
    save_current_state_to_session(
        objective,
        criteria_list,
        alternatives_list,
        perf_df,
        num_interviews,
        interview_matrices
    )

    download_json_button(
        "💾 Scarica backup (.json)",
        st.session_state.backup_state,
        "AHP_backup.json"
    )

    st.divider()

    # =====================================================
    # STEP 6. CALCOLO AHP + (opzionale) VALIDAZIONE ISO/IEC 27005
    # =====================================================
    st.markdown("#### 6. Calcolo finale e ranking + validazione ISO/IEC 27005 (opzionale)")

    # parametri ISO (UI)
    with st.expander("Parametri ISO/IEC 27005 (solo se carichi il Risk Register)", expanded=(iso_df_std is not None)):
        col_mode, col_thr, col_alpha = st.columns(3)
        with col_mode:
            iso_mode = st.selectbox(
                "Modalità integrazione AHP→ISO",
                options=["gate", "penalty"],
                index=0,
                help="gate: esclude alternative non accettabili; penalty: penalizza il punteggio AHP."
            )
        with col_thr:
            iso_threshold = st.number_input(
                "Soglia rischio residuo (accettazione)",
                min_value=0.0,
                max_value=25.0,
                value=9.0,
                step=0.5,
                help="Se Likelihood e Impact sono 1..5, il massimo teorico è 25."
            )
        with col_alpha:
            iso_alpha = st.number_input(
                "Alpha (solo penalty)",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05,
                help="Intensità penalizzazione: FinalScore*(1 - alpha*risk_norm)."
            )

        st.markdown("**Pesi CIA per Impact aggregato (se non usi 'Impact' diretto)**")
        col_c, col_i, col_a = st.columns(3)
        with col_c:
            w_c = st.number_input("Peso C", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
        with col_i:
            w_i = st.number_input("Peso I", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
        with col_a:
            w_a = st.number_input("Peso A", min_value=0.0, max_value=10.0, value=1.0, step=0.5)

    if st.button("Esegui calcolo AHP", type="primary"):
        # 1) aggregazione
        matrices_list = [interview_matrices[i] for i in range(num_interviews)]
        final_criteria_matrix = aggregate_experts_geometric_mean(matrices_list)

        st.markdown("**Matrice criteri aggregata (media geometrica tra esperti)**")
        st.dataframe(final_criteria_matrix.style.format("{:.4f}"), use_container_width=True)

        # 2) pesi
        w = calculate_priority_vector(final_criteria_matrix)
        weights_criteria = pd.Series(w, index=criteria_list, name="Weight")

        st.markdown("**Pesi dei criteri (priority vector)**")
        st.dataframe(weights_criteria.to_frame(), use_container_width=True)

        # 3) consistenza
        CR, CI, lambda_max = calculate_consistency_ratio(final_criteria_matrix, w)

        st.markdown("**Consistenza (Saaty)**")
        col_ci, col_cr, col_lambda = st.columns(3)
        with col_lambda:
            st.metric("λ_max", f"{lambda_max:.4f}")
        with col_ci:
            st.metric("CI", f"{CI:.4f}")
        with col_cr:
            st.metric("CR", f"{CR:.4f}")

        if CR > 0.10:
            st.warning("CR > 0.10 → giudizi poco consistenti, considerare revisione.")
        else:
            st.success("CR ≤ 0.10 → giudizi coerenti.")

        # 4) ranking AHP
        scores = {}
        for alt in alternatives_list:
            row_vals = perf_df.loc[alt, criteria_list].values.astype(float)
            scores[alt] = np.dot(row_vals, weights_criteria.values)

        scores_series = pd.Series(scores, name="FinalScore")
        result_df = perf_df.copy()
        result_df["FinalScore"] = scores_series

        result_ranked = result_df.sort_values(by="FinalScore", ascending=False).reset_index()
        result_ranked.rename(columns={result_ranked.columns[0]: "Alternative"}, inplace=True)
        result_ranked["Rank"] = np.arange(1, len(result_ranked)+1)

        st.markdown("**Ranking AHP delle alternative**")
        st.dataframe(
            result_ranked[["Rank","Alternative","FinalScore"] + criteria_list]
                .style.format({"FinalScore":"{:.4f}", **{c:"{:.2f}" for c in criteria_list}})
                .background_gradient(subset=["FinalScore"], cmap="viridis"),
            use_container_width=True
        )

        best_alt = result_ranked.iloc[0]["Alternative"]
        best_score = result_ranked.iloc[0]["FinalScore"]
        st.success(f"Alternativa migliore (AHP): {best_alt} (score {best_score:.4f})")

        # =====================================================
        # ISO/IEC 27005 VALIDATION (se presente iso_df_std)
        # =====================================================
        iso_register_scored = None
        iso_metrics = None
        combined_ranked = None
        spearman = None

        if iso_df_std is not None:
            try:
                iso_register_scored, iso_metrics = compute_iso27005_metrics(
                    iso_df_std,
                    threshold_residual=float(iso_threshold),
                    w_c=float(w_c), w_i=float(w_i), w_a=float(w_a),
                    risk_max_theoretical=25.0
                )
                combined_ranked = combine_ahp_iso(
                    result_ranked=result_ranked,
                    iso_metrics=iso_metrics,
                    mode=iso_mode,
                    threshold_residual=float(iso_threshold),
                    alpha=float(iso_alpha)
                )

                # Spearman rank correlation AHP vs ISO (validazione di concordanza)
                tmp = combined_ranked.merge(
                    iso_metrics[["alternative", "iso_rank"]],
                    left_on="Alternative", right_on="alternative", how="left"
                )
                spearman = tmp["Rank"].corr(tmp["iso_rank"], method="spearman")

            except Exception as e:
                st.error(f"Errore nel calcolo ISO/IEC 27005: {e}")
                iso_register_scored, iso_metrics, combined_ranked = None, None, None

        st.divider()

        # =====================================================
        # VISUALIZZAZIONI
        # =====================================================
        st.markdown("#### Visualizzazioni")
        tabs = ["Ranking AHP", "Radar", "Pesi criteri"]
        if combined_ranked is not None:
            tabs.append("ISO/IEC 27005")

        tab_objs = st.tabs(tabs)

        with tab_objs[0]:
            plot_bar_ranking(result_ranked, "Alternative", "FinalScore")

        with tab_objs[1]:
            radar_df = perf_df.copy()
            for c in criteria_list:
                col = radar_df[c].astype(float)
                maxv = col.max()
                radar_df[c] = col / maxv if maxv > 0 else 0.0

            radar_rows = [
                radar_df.loc[alt, criteria_list].values.astype(float)
                for alt in alternatives_list
            ]
            plot_radar_chart(criteria_list, radar_rows, alternatives_list)

        with tab_objs[2]:
            fig, ax = plt.subplots(figsize=(5,3))
            bars = ax.bar(criteria_list, weights_criteria.values)
            ax.set_ylabel("Peso")
            ax.set_title("Pesi dei criteri")
            ax.grid(axis='y', alpha=0.3)
            for b, s in zip(bars, weights_criteria.values):
                h = b.get_height()
                ax.text(b.get_x()+b.get_width()/2., h, f"{s:.3f}", ha='center', va='bottom', fontsize=8)
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()
            st.pyplot(fig)

        if combined_ranked is not None:
            with tab_objs[3]:
                st.markdown("**Metriche ISO per alternativa (risk residual)**")
                st.dataframe(
                    iso_metrics.style.format({
                        "residual_max": "{:.3f}",
                        "residual_mean": "{:.3f}",
                        "frac_above_thr": "{:.2%}",
                        "risk_norm": "{:.3f}",
                        "iso_score": "{:.3f}"
                    }),
                    use_container_width=True
                )

                st.markdown("**Ranking combinato (AHP + ISO)**")
                cols_show = [
                    "Rank_ISO", "Alternative", "FinalScore", "FinalScore_ISO",
                    "iso_pass", "residual_max", "residual_mean", "frac_above_thr", "iso_score"
                ]
                st.dataframe(
                    combined_ranked[cols_show]
                        .style.format({
                            "FinalScore": "{:.4f}",
                            "FinalScore_ISO": "{:.4f}",
                            "residual_max": "{:.3f}",
                            "residual_mean": "{:.3f}",
                            "frac_above_thr": "{:.2%}",
                            "iso_score": "{:.3f}"
                        })
                        .background_gradient(subset=["FinalScore_ISO"], cmap="viridis"),
                    use_container_width=True
                )

                if spearman is not None and not np.isnan(spearman):
                    st.caption(f"Spearman corr (Rank AHP vs Rank ISO): {spearman:.3f}")

                st.markdown("**Bar chart ranking combinato**")
                plot_bar_ranking(combined_ranked, "Alternative", "FinalScore_ISO")

                st.markdown("**Risk register (scenario-level) con risk_inherent e risk_residual**")
                st.dataframe(
                    iso_register_scored[["alternative", "likelihood", "impact_eff", "control_effectiveness", "risk_inherent", "risk_residual"]]
                        .head(200)
                        .style.format({
                            "impact_eff": "{:.3f}",
                            "control_effectiveness": "{:.3f}",
                            "risk_inherent": "{:.3f}",
                            "risk_residual": "{:.3f}"
                        }),
                    use_container_width=True
                )

        st.divider()

        # =====================================================
        # EXPORT
        # =====================================================
        st.markdown("#### Esporta risultati")

        output_excel = io.BytesIO()
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            result_ranked.to_excel(writer, sheet_name='AHP_Ranking', index=False)
            weights_criteria.to_frame().to_excel(writer, sheet_name='CriteriaWeights', index=True)
            final_criteria_matrix.to_excel(writer, sheet_name='CriteriaMatrix')

            if combined_ranked is not None:
                combined_ranked.to_excel(writer, sheet_name='AHP_ISO_Ranking', index=False)
                iso_metrics.to_excel(writer, sheet_name='ISO_Metrics', index=False)
                iso_register_scored.to_excel(writer, sheet_name='ISO_RiskRegister', index=False)

        output_excel.seek(0)

        st.download_button(
            label="📥 Scarica risultati (Excel)",
            data=output_excel,
            file_name="AHP_ISO_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        csv_bytes = result_ranked.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Scarica ranking AHP (CSV)",
            data=csv_bytes,
            file_name="AHP_ranking.csv",
            mime="text/csv"
        )

        if combined_ranked is not None:
            csv_bytes2 = combined_ranked.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Scarica ranking combinato AHP+ISO (CSV)",
                data=csv_bytes2,
                file_name="AHP_ISO_ranking.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()
