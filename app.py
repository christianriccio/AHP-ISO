import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import csv
import json


# =========================================================
# I/O AND SESSION STATE HELPERS
# =========================================================

def init_session_state():
    """
    Initializes the Streamlit session state if not already present.
    Used for:
    - Full experiment backup/restore (JSON)
    """
    if "backup_state" not in st.session_state:
        st.session_state.backup_state = {
            "objective": "",
            "criteria": [],
            "alternatives": [],
            "performance_matrix": {},   # {alt: {crit: value}}
            "num_interviews": 1,
            "interviews": {},           # "0": {"pairwise": {...}}
            # AHP Express support (backward compatible with older backups)
            "ahp_variant": "standard",  # "standard" | "express"
            "reference_factor": None    # criterion name
        }


def load_alternatives_file(uploaded_file):
    """
    Generic loader for tabular input files (CSV/Excel).
    Supports CSV with auto-detected delimiter and Excel files.
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
            st.error(f"Excel read error: {e}")
            return None

    # CSV with unknown delimiter
    try:
        df = pd.read_csv(uploaded_file, sep=None, engine='python')
        return df
    except Exception as e:
        st.warning(f"CSV auto-detect attempt failed: {e}")
        try:
            uploaded_file.seek(0)
            rawdata = uploaded_file.read(2048).decode('utf-8', errors='replace')
            uploaded_file.seek(0)
            dialect = csv.Sniffer().sniff(rawdata, delimiters=[',', ';', '\t', '|'])
            sep = dialect.delimiter
            df = pd.read_csv(uploaded_file, sep=sep)
            return df
        except Exception as e2:
            st.error(f"csv.Sniffer fallback failed: {e2}")
            return None


def download_json_button(label, data_dict, filename):
    """
    Button to download a JSON backup of the entire experiment state.
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
    Loads a previously saved JSON backup.
    Returns dict or None if invalid.
    """
    try:
        raw = uploaded_json.read()
        data = json.loads(raw.decode('utf-8'))
        return data
    except Exception as e:
        st.error(f"Invalid JSON backup: {e}")
        return None


def matrix_to_dict(df):
    """Converts a DataFrame to a JSON-serializable dict."""
    return {
        "index": list(df.index),
        "columns": list(df.columns),
        "data": df.values.tolist()
    }


def dict_to_matrix(d):
    """Converts a serialized dict back into a DataFrame."""
    return pd.DataFrame(d["data"], index=d["index"], columns=d["columns"])


def save_current_state_to_session(objective,
                                  criteria_list,
                                  alternatives_list,
                                  perf_df,
                                  num_interviews,
                                  interview_matrices,
                                  ahp_variant,
                                  reference_factor):
    """
    Updates session state with:
    - decision objective
    - criteria list
    - alternatives list
    - performance matrix alternatives x criteria
    - all collected pairwise matrices
    - AHP variant and reference factor (if express)
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

    st.session_state.backup_state["num_interviews"] = int(num_interviews)

    interviews_dict = {}
    for k, dfmat in interview_matrices.items():
        interviews_dict[str(k)] = {"pairwise": matrix_to_dict(dfmat)}
    st.session_state.backup_state["interviews"] = interviews_dict

    st.session_state.backup_state["ahp_variant"] = str(ahp_variant)
    st.session_state.backup_state["reference_factor"] = reference_factor


def load_state_from_backup(backup_dict):
    """
    Restores data from a JSON backup.
    Returns a tuple for pre-filling the UI.
    """
    try:
        objective = backup_dict["objective"]
        criteria_list = backup_dict["criteria"]
        alternatives_list = backup_dict["alternatives"]
        perf_mat_dict = backup_dict["performance_matrix"]
        num_interviews = backup_dict["num_interviews"]

        perf_df = pd.DataFrame.from_dict(perf_mat_dict, orient="index")
        perf_df = perf_df[criteria_list]  # preserve column order

        interview_matrices = {}
        for key, sub in backup_dict["interviews"].items():
            interview_matrices[int(key)] = dict_to_matrix(sub["pairwise"])

        ahp_variant = backup_dict.get("ahp_variant", "standard")
        reference_factor = backup_dict.get("reference_factor", None)

        st.session_state.backup_state = backup_dict

        return (objective,
                criteria_list,
                alternatives_list,
                perf_df,
                int(num_interviews),
                interview_matrices,
                ahp_variant,
                reference_factor)

    except Exception as e:
        st.error(f"Backup restore error: {e}")
        return None


# =========================================================
# AHP CORE
# =========================================================

def create_empty_pairwise_matrix(elements):
    """
    Creates an n x n matrix with 1s on the diagonal.
    """
    n = len(elements)
    mat = np.ones((n, n), dtype=float)
    return pd.DataFrame(mat, index=elements, columns=elements)


def calculate_priority_vector(pairwise_matrix: pd.DataFrame):
    """
    Criteria priority vector (Saaty approx):
    column-normalize, then average each row.
    """
    A = pairwise_matrix.values.astype(float)
    col_sum = A.sum(axis=0)
    norm = A / col_sum
    w = norm.mean(axis=1)
    return w


def calculate_consistency_ratio(pairwise_matrix: pd.DataFrame, weights: np.ndarray):
    """
    Computes λ_max, CI, and CR for an AHP matrix.
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
    CR = 0.0 if n <= 2 else (CI / RI if RI != 0 else 0.0)

    return CR, CI, lambda_max


def geometric_mean(values):
    """
    Geometric mean of positive values.
    """
    arr = np.array(values, dtype=float)
    if np.any(arr <= 0):
        return 0.0
    prod = np.prod(arr)
    return prod ** (1.0 / len(arr))


def aggregate_experts_geometric_mean(list_of_matrices):
    """
    Group-AHP aggregation: element-wise geometric mean across experts.
    """
    if len(list_of_matrices) == 1:
        return list_of_matrices[0].copy()

    idx = list_of_matrices[0].index
    n = len(idx)
    agg = np.ones((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            vals = [m.iloc[i, j] for m in list_of_matrices]
            agg[i, j] = geometric_mean(vals)

    return pd.DataFrame(agg, index=idx, columns=idx)


def reconstruct_full_matrix_from_reference(partial_matrix: pd.DataFrame, reference_factor: str) -> pd.DataFrame:
    """
    Reconstructs the full AHP matrix A (n x n) using only comparisons vs a reference factor r:
      a_ij = a_{i,r} / a_{j,r}

    Assumes partial_matrix contains (at least) all a_{i,r} and a_{r,i} entries filled, and diagonal = 1.
    """
    elements = list(partial_matrix.index)
    if reference_factor not in elements:
        raise ValueError("Reference factor is not in the criteria list.")

    A = create_empty_pairwise_matrix(elements)

    a_ir = {e: float(partial_matrix.loc[e, reference_factor]) for e in elements}
    if any(v <= 0 for v in a_ir.values()):
        raise ValueError("Invalid values in reference-factor comparisons (<= 0).")

    for i in elements:
        for j in elements:
            A.loc[i, j] = 1.0 if i == j else (a_ir[i] / a_ir[j])

    return A


# =========================================================
# ISO/IEC 27005 - EX-POST VALIDATION
# =========================================================

def _pick_column(df_cols_lower_map, *candidates):
    for c in candidates:
        if c in df_cols_lower_map:
            return df_cols_lower_map[c]
    return None


def standardize_iso_risk_register(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes ISO risk register columns and validates minimum requirements.

    Supported columns (case-insensitive):
    - Alternative: alternative | dlm | alt
    - Likelihood: likelihood | l
    - Impact: impact (optional if CIA is present)
    - Impact_C: impact_c | c | impact_confidentiality
    - Impact_I: impact_i | i | impact_integrity
    - Impact_A: impact_a | a | impact_availability
    - Control_Effectiveness: control_effectiveness | control_eff | mitigation | controls

    Returns a DF with canonical columns:
      alternative, likelihood, impact_C, impact_I, impact_A, control_effectiveness
    (or 'impact' if present; in that case CIA can be missing).
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
    if alt_col is None:
        missing.append("Alternative (alternative|dlm|alt)")
    if lik_col is None:
        missing.append("Likelihood (likelihood|l)")
    if ce_col is None:
        missing.append("Control_Effectiveness (control_effectiveness|control_eff|mitigation|controls)")

    has_impact_direct = (impact_col is not None)
    has_cia = (ic_col is not None and ii_col is not None and ia_col is not None)

    if not has_impact_direct and not has_cia:
        missing.append("Impact: (impact) OR (impact_C, impact_I, impact_A)")

    if missing:
        raise ValueError("Missing columns in ISO risk register: " + "; ".join(missing))

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
    Computes:
    - inherent risk = likelihood * impact
    - residual risk = inherent risk * (1 - control_effectiveness)

    Aggregates per alternative:
    residual_max, residual_mean, frac_above_threshold, iso_pass, iso_score (1 - normalized)

    Returns:
    - iso_register_scored: scenario-level DF with inherent/residual risk
    - iso_metrics: alternative-level DF with ISO metrics and ISO ranking
    """
    df = iso_df.copy()

    # Effective impact: direct impact if available, else weighted CIA
    if df["impact"].notna().any():
        df["impact_eff"] = df["impact"]
    else:
        denom = (w_c + w_i + w_a)
        if denom <= 0:
            raise ValueError("Invalid CIA weights (sum <= 0).")
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
    Merges AHP ranking with ISO metrics and produces a combined ranking.

    mode:
      - "gate": score=FinalScore if iso_pass else 0
      - "penalty": score=FinalScore*(1 - alpha*risk_norm)
    """
    df = result_ranked.copy()
    df = df.merge(iso_metrics, left_on="Alternative", right_on="alternative", how="left")

    # If ISO is missing for an alternative -> conservative assumption: not pass
    df["iso_pass"] = df["iso_pass"].fillna(False)
    df["risk_norm"] = df["risk_norm"].fillna(1.0)
    df["iso_score"] = df["iso_score"].fillna(0.0)

    mode = str(mode).strip().lower()
    if mode == "gate":
        df["FinalScore_ISO"] = np.where(df["iso_pass"], df["FinalScore"], 0.0)
    elif mode == "penalty":
        df["FinalScore_ISO"] = df["FinalScore"] * np.clip(1.0 - float(alpha) * df["risk_norm"], 0.0, 1.0)
    else:
        raise ValueError("mode must be 'gate' or 'penalty'.")

    df = df.sort_values("FinalScore_ISO", ascending=False).reset_index(drop=True)
    df["Rank_ISO"] = np.arange(1, len(df) + 1)

    return df


# =========================================================
# VISUALIZATION
# =========================================================

def plot_radar_chart(criteria_list, data_rows, labels):
    """
    Radar chart to compare alternatives on normalized criteria.
    """
    N = len(criteria_list)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for i, rowvals in enumerate(data_rows):
        vals = list(rowvals) + [rowvals[0]]
        ax.plot(angles, vals, linewidth=2, label=labels[i])
        ax.fill(angles, vals, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(criteria_list, fontsize=9)
    ax.set_yticks([])
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), fontsize=9)

    st.pyplot(fig)


def plot_bar_ranking(df_ranked, alt_col_name, score_col_name):
    """
    Bar chart of alternatives sorted by the selected score column.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    names = df_ranked[alt_col_name].tolist()
    scores = df_ranked[score_col_name].tolist()

    bars = ax.bar(names, scores)
    ax.set_ylabel("Score")
    ax.set_title(f"Ranking ({score_col_name})")
    ax.grid(axis='y', alpha=0.3)

    for b, s in zip(bars, scores):
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2., h, f"{s:.3f}",
                ha='center', va='bottom', fontsize=8)

    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    st.pyplot(fig)


# =========================================================
# STREAMLIT APP
# =========================================================

def main():
    st.set_page_config(
        page_title="AHP-ISO",
        layout="wide",
        page_icon="📊"
    )

    init_session_state()

    title_col, _ = st.columns([0.75, 0.25])
    with title_col:
        st.markdown("### AHP + ISO/IEC 27005")
        st.caption("Multi-criteria decision support with multi-expert AHP and optional ISO/IEC 27005 ex-post validation.")

    # =====================================================
    # TUTORIAL (INITIAL SECTION)
    # =====================================================
    with st.expander("Tutorial (read first) — How to use this app", expanded=True):
        st.markdown(
            """
This tool computes an AHP-based ranking of alternatives and, if you provide an ISO/IEC 27005 risk register, it can validate or adjust the ranking using residual-risk metrics.

The workflow is the following. First, you provide a dataset containing alternatives (first column) and numerical criteria values (subsequent columns). Second, you choose how to elicit the criteria weights: either Standard AHP (full pairwise comparisons) or AHP Express (reduced comparisons via a reference factor). Third, you collect judgments from one or more experts (interviews) and the app aggregates them via element-wise geometric mean (Group AHP). Finally, the app computes the criteria weights, checks consistency (CR), and ranks alternatives with a weighted sum using your performance matrix.

If you also upload an ISO risk register, the app computes inherent and residual risk per scenario and then aggregates these metrics per alternative. You can choose how ISO affects the final ranking:
- Gate mode: alternatives that do not pass the residual-risk threshold are excluded (their ISO-adjusted score becomes 0).
- Penalty mode: alternatives are penalized proportionally to normalized residual risk.

AHP Express details: you select one criterion as the reference factor. For each expert, the app asks only the comparisons between each criterion and the reference factor (n−1 judgments instead of n(n−1)/2). The full pairwise matrix is reconstructed as a_ij = a_{i,r} / a_{j,r}. This reconstruction is coherent by construction, so the Consistency Ratio is typically very low (often ~0, up to rounding).

Recommended practical usage: start with 1 expert to validate the pipeline and the ISO file format, then increase the number of experts. If CR is above 0.10 in Standard AHP, revisit the most conflicting judgments. In Express mode, CR is not a diagnostic of judgment quality because the matrix is derived from a constrained structure.

Data format notes. The AHP dataset expects numeric criteria columns. The ISO risk register expects at least Alternative + Likelihood + Control_Effectiveness and either Impact (single column) or the CIA triplet (Impact_C, Impact_I, Impact_A). Control_Effectiveness must be in [0,1]. Likelihood/Impact are typically in [1,5] (so maximum theoretical risk is 25), but you can adapt the threshold accordingly.

At any point, you can download a full JSON backup and restore it later to avoid re-entering judgments.
            """
        )

    st.divider()

    # =====================================================
    # STEP 1. OBJECTIVE
    # =====================================================
    st.markdown("#### 1. Objective")
    objective = st.text_input(
        "Decision objective",
        value=st.session_state.backup_state.get("objective", ""),
        placeholder="Example: Select the best alternative"
    )

    st.divider()

    # =====================================================
    # STEP 2. INPUTS: AHP DATA + OPTIONAL ISO RISK REGISTER
    # =====================================================
    st.markdown("#### 2. Inputs (Alternatives/Criteria + optional ISO Risk Register)")

    col_upload, col_backup, col_iso = st.columns(3)

    with col_upload:
        st.write("Upload the AHP dataset (CSV or Excel).")
        st.caption("Expected format: first column = alternative name; next columns = numeric criteria.")
        uploaded_data = st.file_uploader(
            "Upload AHP data",
            type=["csv", "xlsx", "xls"],
            key="uploaded_data_file"
        )

    with col_backup:
        st.write("Or restore a full JSON backup.")
        uploaded_backup = st.file_uploader(
            "Upload JSON backup",
            type=["json"],
            key="uploaded_backup_file"
        )

    with col_iso:
        st.write("Upload an ISO risk register (optional).")
        st.caption(
            "Minimum columns: Alternative, Likelihood, Control_Effectiveness and either\n"
            "Impact (single column) OR (Impact_C, Impact_I, Impact_A)."
        )
        uploaded_iso = st.file_uploader(
            "Upload ISO Risk Register",
            type=["csv", "xlsx", "xls"],
            key="uploaded_iso_file"
        )

    restored = None
    if uploaded_backup is not None:
        backup_dict = try_load_backup_json(uploaded_backup)
        if backup_dict is not None:
            restored = load_state_from_backup(backup_dict)
            st.success("Backup loaded. The interface has been pre-filled with saved data.")

    if restored is not None:
        (objective_rest,
         criteria_list_rest,
         alternatives_list_rest,
         perf_df_rest,
         num_interviews_rest,
         interview_matrices_rest,
         ahp_variant_rest,
         reference_factor_rest) = restored

        if not objective:
            objective = objective_rest
    else:
        criteria_list_rest = []
        alternatives_list_rest = []
        perf_df_rest = None
        num_interviews_rest = int(st.session_state.backup_state.get("num_interviews", 1))
        interview_matrices_rest = {}
        ahp_variant_rest = st.session_state.backup_state.get("ahp_variant", "standard")
        reference_factor_rest = st.session_state.backup_state.get("reference_factor", None)

    # AHP dataset
    if uploaded_data is not None:
        df_raw = load_alternatives_file(uploaded_data)
        if df_raw is None or df_raw.empty:
            st.error("The AHP file is empty or invalid.")
            st.stop()
    elif perf_df_rest is not None:
        df_raw = perf_df_rest.reset_index().rename(columns={"index": "Alternative"})
    else:
        df_raw = None

    # ISO dataset
    iso_df_std = None
    iso_df_raw = None
    if uploaded_iso is not None:
        iso_df_raw = load_alternatives_file(uploaded_iso)
        if iso_df_raw is None or iso_df_raw.empty:
            st.error("The ISO Risk Register file is empty or invalid.")
            st.stop()
        try:
            iso_df_std = standardize_iso_risk_register(iso_df_raw)
        except Exception as e:
            st.error(f"Invalid ISO Risk Register format: {e}")
            iso_df_std = None

    if df_raw is not None:
        st.markdown("**AHP data preview**")
        st.dataframe(df_raw.head(), use_container_width=True)

        alt_col_name = df_raw.columns[0]
        alternatives_list = df_raw[alt_col_name].astype(str).tolist()
        criteria_list = list(df_raw.columns[1:])
        perf_df = df_raw.set_index(alt_col_name).astype(float)

        st.caption(f"Detected alternatives: {alternatives_list}")
        st.caption(f"Detected criteria: {criteria_list}")
    else:
        alternatives_list = alternatives_list_rest
        criteria_list = criteria_list_rest
        perf_df = perf_df_rest if perf_df_rest is not None else pd.DataFrame()

    if iso_df_std is not None:
        st.markdown("**ISO Risk Register preview (standardized)**")
        st.dataframe(iso_df_std.head(), use_container_width=True)

        missing_iso_alts = sorted(set(alternatives_list) - set(iso_df_std["alternative"].unique().tolist()))
        if missing_iso_alts:
            st.warning(
                "Some AHP alternatives are missing in the ISO Risk Register. "
                "These alternatives will be treated conservatively as 'not pass': "
                + ", ".join(missing_iso_alts[:10]) + ("..." if len(missing_iso_alts) > 10 else "")
            )

    if len(criteria_list) == 0 or len(alternatives_list) == 0:
        st.warning("Please upload an AHP dataset or a JSON backup to proceed.")
        return

    st.divider()

    # =====================================================
    # STEP 3. NUMBER OF EXPERTS
    # =====================================================
    st.markdown("#### 3. Experts / Interviews")
    num_interviews = st.number_input(
        "Number of experts/interviews",
        min_value=1,
        max_value=20,
        step=1,
        value=int(num_interviews_rest),
        help="A criteria pairwise matrix is collected for each expert."
    )

    st.divider()

    # =====================================================
    # STEP 4. PAIRWISE INPUT (STANDARD OR EXPRESS)
    # =====================================================
    st.markdown("#### 4. Criteria comparisons (pairwise)")

    n_crit = len(criteria_list)

    ahp_variant_ui = st.selectbox(
        "AHP variant",
        options=["standard", "express"],
        index=0 if str(ahp_variant_rest).lower() != "express" else 1,
        help="Standard: full pairwise comparisons. Express: n-1 comparisons vs a reference factor and matrix reconstruction."
    )

    reference_factor_ui = None
    if ahp_variant_ui == "express":
        default_ref = reference_factor_rest if reference_factor_rest in criteria_list else criteria_list[0]
        reference_factor_ui = st.selectbox(
            "Reference factor (reference criterion)",
            options=criteria_list,
            index=criteria_list.index(default_ref),
            help="In Express AHP, you only compare each criterion against this reference factor."
        )
        st.caption(f"Comparisons per expert: {n_crit - 1} (Express) instead of {n_crit * (n_crit - 1) // 2} (Standard).")
    else:
        st.caption(f"Comparisons per expert: {n_crit * (n_crit - 1) // 2} (Standard).")

    interview_matrices = {}
    if len(interview_matrices_rest) > 0:
        for k, dfmat in interview_matrices_rest.items():
            interview_matrices[k] = dfmat.copy()

    for interview_id in range(int(num_interviews)):
        st.markdown(f"**Expert #{interview_id + 1}**")

        if interview_id in interview_matrices:
            pairwise_df = interview_matrices[interview_id]
            try:
                pairwise_df = pairwise_df.loc[criteria_list, criteria_list].copy()
            except Exception:
                pairwise_df = create_empty_pairwise_matrix(criteria_list)
        else:
            pairwise_df = create_empty_pairwise_matrix(criteria_list)

        with st.expander(f"Criteria comparisons — Expert #{interview_id + 1}", expanded=True):

            if ahp_variant_ui == "standard":
                # Full comparisons
                for i in range(n_crit):
                    for j in range(i + 1, n_crit):
                        left = criteria_list[i]
                        right = criteria_list[j]

                        st.write(f"{left} ↔ {right}")

                        pref = st.radio(
                            "Which criterion is more important?",
                            options=[left, "Equal", right],
                            index=1,
                            key=f"pref-std-{interview_id}-{i}-{j}"
                        )

                        default_val = float(pairwise_df.loc[left, right])
                        default_slider_val = 1 if default_val < 1 else int(np.clip(round(default_val), 1, 9))

                        intensity = st.slider(
                            "Strength of importance (1 = equal, 9 = extremely more important)",
                            min_value=1,
                            max_value=9,
                            value=default_slider_val,
                            step=1,
                            key=f"intensity-std-{interview_id}-{i}-{j}"
                        )

                        if pref == "Equal" or intensity == 1:
                            val = 1.0
                        elif pref == left:
                            val = float(intensity)
                        else:
                            val = 1.0 / float(intensity)

                        pairwise_df.loc[left, right] = val
                        pairwise_df.loc[right, left] = 1.0 / val
                        pairwise_df.loc[left, left] = 1.0
                        pairwise_df.loc[right, right] = 1.0

                st.caption("Pairwise matrix for this expert")
                st.dataframe(pairwise_df.style.format("{:.4f}"), use_container_width=True)

            else:
                # Express comparisons vs reference factor
                r = reference_factor_ui
                r_idx = criteria_list.index(r)

                for k in range(n_crit):
                    crit_k = criteria_list[k]
                    if crit_k == r:
                        continue

                    left = crit_k
                    right = r

                    st.write(f"{left} ↔ {right}")

                    pref = st.radio(
                        "Which criterion is more important?",
                        options=[left, "Equal", right],
                        index=1,
                        key=f"pref-exp-{interview_id}-{k}-{r_idx}"
                    )

                    default_val = float(pairwise_df.loc[left, right])
                    default_slider_val = 1 if default_val < 1 else int(np.clip(round(default_val), 1, 9))

                    intensity = st.slider(
                        "Strength of importance (1 = equal, 9 = extremely more important)",
                        min_value=1,
                        max_value=9,
                        value=default_slider_val,
                        step=1,
                        key=f"intensity-exp-{interview_id}-{k}-{r_idx}"
                    )

                    if pref == "Equal" or intensity == 1:
                        val = 1.0
                    elif pref == left:
                        val = float(intensity)
                    else:
                        val = 1.0 / float(intensity)

                    pairwise_df.loc[left, right] = val
                    pairwise_df.loc[right, left] = 1.0 / val
                    pairwise_df.loc[left, left] = 1.0
                    pairwise_df.loc[right, right] = 1.0

                try:
                    full_df = reconstruct_full_matrix_from_reference(pairwise_df, r)
                except Exception as e:
                    st.error(f"Express reconstruction error: {e}")
                    full_df = pairwise_df.copy()

                st.caption("Reconstructed full matrix (AHP Express; coherent by construction)")
                st.dataframe(full_df.style.format("{:.4f}"), use_container_width=True)

                pairwise_df = full_df

        interview_matrices[interview_id] = pairwise_df

    st.divider()

    # =====================================================
    # STEP 5. BACKUP
    # =====================================================
    st.markdown("#### 5. Backup (optional)")
    save_current_state_to_session(
        objective=objective,
        criteria_list=criteria_list,
        alternatives_list=alternatives_list,
        perf_df=perf_df,
        num_interviews=num_interviews,
        interview_matrices=interview_matrices,
        ahp_variant=ahp_variant_ui,
        reference_factor=reference_factor_ui
    )

    download_json_button(
        label="💾 Download backup (.json)",
        data_dict=st.session_state.backup_state,
        filename="AHP_backup.json"
    )

    st.divider()

    # =====================================================
    # STEP 6. COMPUTE AHP + OPTIONAL ISO VALIDATION
    # =====================================================
    st.markdown("#### 6. Final computation and ranking + optional ISO/IEC 27005 validation")

    with st.expander("ISO/IEC 27005 parameters (only if you upload a Risk Register)", expanded=(iso_df_std is not None)):
        col_mode, col_thr, col_alpha = st.columns(3)
        with col_mode:
            iso_mode = st.selectbox(
                "AHP → ISO integration mode",
                options=["gate", "penalty"],
                index=0,
                help="Gate: excludes non-acceptable alternatives. Penalty: reduces AHP score by residual-risk magnitude."
            )
        with col_thr:
            iso_threshold = st.number_input(
                "Residual-risk acceptance threshold",
                min_value=0.0,
                max_value=25.0,
                value=9.0,
                step=0.5,
                help="If Likelihood and Impact are in [1..5], the theoretical maximum is 25."
            )
        with col_alpha:
            iso_alpha = st.number_input(
                "Alpha (penalty mode only)",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05,
                help="Penalty intensity: FinalScore*(1 - alpha*risk_norm)."
            )

        st.markdown("**CIA weights for effective impact (if you do not provide a direct 'Impact' column)**")
        col_c, col_i, col_a = st.columns(3)
        with col_c:
            w_c = st.number_input("Weight C", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
        with col_i:
            w_i = st.number_input("Weight I", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
        with col_a:
            w_a = st.number_input("Weight A", min_value=0.0, max_value=10.0, value=1.0, step=0.5)

    if st.button("Run AHP computation", type="primary"):
        # 1) Aggregate expert matrices
        matrices_list = [interview_matrices[i] for i in range(int(num_interviews))]
        final_criteria_matrix = aggregate_experts_geometric_mean(matrices_list)

        st.markdown("**Aggregated criteria matrix (element-wise geometric mean across experts)**")
        st.dataframe(final_criteria_matrix.style.format("{:.4f}"), use_container_width=True)

        # 2) Weights
        w = calculate_priority_vector(final_criteria_matrix)
        weights_criteria = pd.Series(w, index=criteria_list, name="Weight")

        st.markdown("**Criteria weights (priority vector)**")
        st.dataframe(weights_criteria.to_frame(), use_container_width=True)

        # 3) Consistency
        CR, CI, lambda_max = calculate_consistency_ratio(final_criteria_matrix, w)

        st.markdown("**Consistency (Saaty)**")
        col_lambda, col_ci, col_cr = st.columns(3)
        with col_lambda:
            st.metric("λ_max", f"{lambda_max:.4f}")
        with col_ci:
            st.metric("CI", f"{CI:.4f}")
        with col_cr:
            st.metric("CR", f"{CR:.4f}")

        if CR > 0.10:
            st.warning("CR > 0.10 → low consistency. Consider revising judgments (Standard AHP).")
        else:
            st.success("CR ≤ 0.10 → acceptable consistency (Standard AHP).")

        # 4) AHP ranking of alternatives (weighted sum of performance values)
        scores = {}
        for alt in alternatives_list:
            row_vals = perf_df.loc[alt, criteria_list].values.astype(float)
            scores[alt] = float(np.dot(row_vals, weights_criteria.values))

        scores_series = pd.Series(scores, name="FinalScore")
        result_df = perf_df.copy()
        result_df["FinalScore"] = scores_series

        result_ranked = result_df.sort_values(by="FinalScore", ascending=False).reset_index()
        result_ranked.rename(columns={result_ranked.columns[0]: "Alternative"}, inplace=True)
        result_ranked["Rank"] = np.arange(1, len(result_ranked) + 1)

        st.markdown("**AHP ranking of alternatives**")
        st.dataframe(
            result_ranked[["Rank", "Alternative", "FinalScore"] + criteria_list]
            .style.format({"FinalScore": "{:.4f}", **{c: "{:.2f}" for c in criteria_list}})
            .background_gradient(subset=["FinalScore"], cmap="viridis"),
            use_container_width=True
        )

        best_alt = result_ranked.iloc[0]["Alternative"]
        best_score = result_ranked.iloc[0]["FinalScore"]
        st.success(f"Top alternative (AHP): {best_alt} (score {best_score:.4f})")

        # ISO/IEC 27005 validation (optional)
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

                tmp = combined_ranked.merge(
                    iso_metrics[["alternative", "iso_rank"]],
                    left_on="Alternative", right_on="alternative", how="left"
                )
                spearman = tmp["Rank"].corr(tmp["iso_rank"], method="spearman")

            except Exception as e:
                st.error(f"ISO/IEC 27005 computation error: {e}")
                iso_register_scored, iso_metrics, combined_ranked = None, None, None

        st.divider()

        # =====================================================
        # VISUALIZATIONS
        # =====================================================
        st.markdown("#### Visualizations")
        tabs = ["AHP ranking", "Radar chart", "Criteria weights"]
        if combined_ranked is not None:
            tabs.append("ISO/IEC 27005")

        tab_objs = st.tabs(tabs)

        with tab_objs[0]:
            plot_bar_ranking(result_ranked, "Alternative", "FinalScore")

        with tab_objs[1]:
            radar_df = perf_df.copy()
            for c in criteria_list:
                col = radar_df[c].astype(float)
                maxv = float(col.max())
                radar_df[c] = (col / maxv) if maxv > 0 else 0.0

            radar_rows = [radar_df.loc[alt, criteria_list].values.astype(float) for alt in alternatives_list]
            plot_radar_chart(criteria_list, radar_rows, alternatives_list)

        with tab_objs[2]:
            fig, ax = plt.subplots(figsize=(5, 3))
            bars = ax.bar(criteria_list, weights_criteria.values)
            ax.set_ylabel("Weight")
            ax.set_title("Criteria weights")
            ax.grid(axis='y', alpha=0.3)
            for b, s in zip(bars, weights_criteria.values):
                h = b.get_height()
                ax.text(b.get_x() + b.get_width()/2., h, f"{s:.3f}",
                        ha='center', va='bottom', fontsize=8)
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()
            st.pyplot(fig)

        if combined_ranked is not None:
            with tab_objs[3]:
                st.markdown("**ISO metrics per alternative (residual risk)**")
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

                st.markdown("**Combined ranking (AHP + ISO)**")
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
                    st.caption(f"Spearman correlation (AHP rank vs ISO rank): {spearman:.3f}")

                st.markdown("**Combined ranking bar chart**")
                plot_bar_ranking(combined_ranked, "Alternative", "FinalScore_ISO")

                st.markdown("**Scored risk register (scenario-level) with inherent and residual risk**")
                st.dataframe(
                    iso_register_scored[["alternative", "likelihood", "impact_eff", "control_effectiveness",
                                         "risk_inherent", "risk_residual"]]
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
        # EXPORTS
        # =====================================================
        st.markdown("#### Export results")

        output_excel = io.BytesIO()
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            result_ranked.to_excel(writer, sheet_name='AHP_Ranking', index=False)
            weights_criteria.to_frame().to_excel(writer, sheet_name='Criteria_Weights', index=True)
            final_criteria_matrix.to_excel(writer, sheet_name='Criteria_Matrix')

            if combined_ranked is not None:
                combined_ranked.to_excel(writer, sheet_name='AHP_ISO_Ranking', index=False)
                iso_metrics.to_excel(writer, sheet_name='ISO_Metrics', index=False)
                iso_register_scored.to_excel(writer, sheet_name='ISO_RiskRegister', index=False)

        output_excel.seek(0)

        st.download_button(
            label="📥 Download results (Excel)",
            data=output_excel,
            file_name="AHP_ISO_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        csv_bytes = result_ranked.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download AHP ranking (CSV)",
            data=csv_bytes,
            file_name="AHP_ranking.csv",
            mime="text/csv"
        )

        if combined_ranked is not None:
            csv_bytes2 = combined_ranked.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download combined ranking AHP+ISO (CSV)",
                data=csv_bytes2,
                file_name="AHP_ISO_ranking.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()
