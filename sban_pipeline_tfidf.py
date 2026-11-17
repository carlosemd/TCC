"""
Pipeline do TCC (VERSAO TF-IDF): dataset SBAN, somente codigo-fonte.

  1. subconjunto (5% do SBAN) + holdout estratificado 80/20
  2. treino do modelo base (Random Forest)
  3. avaliacao (F1 + curva ROC/AUC) + SHAP do modelo base
  4. analise exploratoria (SOMENTE treino) -> relatorios HTML gerados pelo Sweetviz

Uso:
  python sban_pipeline_tfidf.py prepare    # etapa 1 (+ checagem de nulos; grava a FONTE de cada amostra)
  python sban_pipeline_tfidf.py features   # aprende o vocabulario no treino e o lista
  python sban_pipeline_tfidf.py base       # etapas 2 e 3 (vocabulario completo)
  python sban_pipeline_tfidf.py eda        # etapa 4: relatorios HTML (Sweetviz) em outputs_tfidf/eda/
  # iteracao SHAP: exclude / restore / iterate (controlada por ITERATION_ENABLED)
"""
import argparse
import fnmatch
import json
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

# =====================================================================
# CONFIGURACAO: parametros do experimento
# =====================================================================
SEED = 2026
SAMPLE_FRACTION = 0.05
ITERATION_ENABLED = True
TEST_SIZE = 0.20
SHAP_SAMPLE_SIZE = 800
SHAP_BATCH = 100
SHAP_ADDITIVITY_TOL = 0.035 #0.005
SHAP_SPLIT = "train"

EDA_PAIRWISE = "auto"

SBAN_ROOT = Path(r"E:\.Codigo\.TCC\SBAN")
FIELD_ID = "ID"
FIELD_CODE = "source_code"

# Fontes de dados com o rotulo de cada uma (0 = benigno, 1 = malicioso).
SOURCES = [
    (SBAN_ROOT / "Benign" / "1.CodeNet" / "Source code" / "cn_source code.json", 0),
    (SBAN_ROOT / "Benign" / "2.ExeBench" / "Source code" / "eb_source_code.json", 0),
    (SBAN_ROOT / "Malware" / "1.BODMAS" / "Source code" / "b_source_code.json", 1),
    (SBAN_ROOT / "Malware" / "2.MalwareBazzar" / "Source code" / "m_source_code2.json", 1),
    (SBAN_ROOT / "Malware" / "3.Sorel20m" / "Source code" / "a_source_code.json", 1),
    (SBAN_ROOT / "Malware" / "4.Dike" / "Source code" / "d_source_code.json", 1),
    (SBAN_ROOT / "Malware" / "5.xLangKode" / "Source code" / "sub275" / "filtered_sc_sub275.json", 1),
    (SBAN_ROOT / "Malware" / "5.xLangKode" / "Source code" / "sub300" / "filtered_Dataset3_6jun_AI_output.json", 1),
]

OUT = Path("outputs_tfidf")

# Hiperparametros do TF-IDF: tokens sao identificadores, com diferenca entre maiusculas e minusculas.
TFIDF_PARAMS = dict(
    token_pattern=r"[A-Za-z_]\w*",
    lowercase=False,
    max_features=2000,
    min_df=5,
    sublinear_tf=False,
    use_idf=True,
    smooth_idf=True,
    norm="l2",
    dtype=np.float32,
)

# Hiperparametros do Random Forest (sem tratamento de desbalanceamento).
RF_PARAMS = dict(
    n_estimators=100,
    criterion="gini",
    max_depth=60,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=True,
    class_weight=None,
    random_state=SEED,
    n_jobs=-1,
    verbose=1,
)


# Log com horario, usado para acompanhar etapas demoradas.
def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =====================================================================
# ETAPA 1: amostragem, holdout estratificado e checagem de nulos
# =====================================================================
# Relatorio da proporcao de maliciosos no subconjunto, treino e teste.
def report_balance(y_all, y_train, y_test):
    print("\n=== MEDIA DE MALICIOSOS (rotulo 1 = malicioso; media do rotulo = proporcao) ===")
    for name, y in ((f"subconjunto ({SAMPLE_FRACTION:.0%})", y_all), ("treino (80%)", y_train),
                    ("teste (20%)", y_test)):
        print(f"  {name:<18} n={len(y):>9,}  maliciosos={int(y.sum()):>8,}  "
              f"media de maliciosos = {y.mean():.4%}")


# Conta valores que nao sao texto (nulos ou outros tipos).
def count_non_text(s: pd.Series) -> int:
    return int((~s.map(lambda x: isinstance(x, str))).sum())


# Extrai o nome da fonte a partir da pasta numerada do caminho.
def source_name(path: Path) -> str:
    """Nome curto da fonte, tirado da pasta numerada (ex.: '1.CodeNet', '3.Sorel20m')."""
    for part in reversed(path.parts):
        if re.match(r"^\d+\.", part):
            return part
    return path.parent.name


# Leitura do JSON item a item, sem carregar o arquivo inteiro na memoria.
def iter_json_array(path):
    """Le um JSON cujo topo e uma LISTA, item a item (ijson), sem carregar o arquivo na memoria.
    json.load() le o arquivo inteiro e estoura a RAM nos arquivos grandes (MemoryError)."""
    try:
        import ijson
    except ImportError:
        sys.exit("Biblioteca ausente. Instale com: pip install ijson")
    with open(path, "rb") as fh:
        yield from ijson.items(fh, "item", use_float=True)


# Amostragem aleatoria em duas passagens: primeiro conta os itens, depois guarda so os sorteados.
def sample_json(path, frac, seed, items=iter_json_array) -> pd.DataFrame:
    """Equivale a pd.DataFrame(json.load(...))[[ID, CODE]].sample(frac, random_state=seed), mas em fluxo:
    passo 1 conta os itens; passo 2 guarda SO as amostras sorteadas (mesmos indices e mesma ordem do
    DataFrame.sample do pandas: RandomState(seed).choice com n = round(frac * N)). Custo: le o arquivo 2 vezes."""
    n_total = 0
    for item in items(path):
        if n_total == 0:
            if not isinstance(item, dict):
                sys.exit(f"ERRO em {path}: os itens nao sao objetos JSON (tipo {type(item).__name__}).")
            missing = [c for c in (FIELD_ID, FIELD_CODE) if c not in item]
            if missing:
                sys.exit(f"ERRO em {path}: campos ausentes {missing}. "
                         f"Campos encontrados: {list(item.keys())}. "
                         f"Ajuste FIELD_ID/FIELD_CODE (ou me informe).")
        n_total += 1
    if n_total == 0:
        sys.exit(f"ERRO em {path}: nenhum item encontrado. O topo do JSON precisa ser uma lista "
                 "([ {...}, {...} ]); se for outra estrutura, me informe.")
    n = int(round(frac * n_total))
    log(f"  {n_total:,} itens no arquivo; sorteando {n:,}")
    chosen = np.random.RandomState(seed).choice(n_total, size=n, replace=False)
    rank = {int(i): r for r, i in enumerate(chosen)}
    rows = [None] * n
    for i, item in enumerate(items(path)):
        r = rank.get(i)
        if r is not None:
            rows[r] = {FIELD_ID: item[FIELD_ID], FIELD_CODE: item[FIELD_CODE]}
    return pd.DataFrame(rows, columns=[FIELD_ID, FIELD_CODE])


# Comando prepare: amostra cada fonte, divide em treino/teste e grava os arquivos intermediarios.
def cmd_prepare():
    if not SOURCES or not any(l == 1 for _, l in SOURCES) or not any(l == 0 for _, l in SOURCES):
        sys.exit("ERRO: SOURCES precisa ter ao menos um JSON benigno (0) e um malicioso (1).")
    OUT.mkdir(exist_ok=True)

    parts = []
    for path, label in SOURCES:
        log(f"Lendo {path} (rotulo={label}) ...")
        df = sample_json(path, SAMPLE_FRACTION, SEED)
        df["label"] = label
        df["source"] = source_name(path)
        print(f"  amostrados {len(df):,} ({SAMPLE_FRACTION:.0%} deste arquivo)")
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)

    # Remocao de duplicatas exatas (dentro e entre fontes) ANTES do split, para nao haver o mesmo codigo em treino e teste.
    # Textos iguais com rotulos diferentes sao removidos por completo (rotulo ambiguo).
    n_before = len(df)
    df["_hash"] = pd.util.hash_pandas_object(df[FIELD_CODE].astype(str), index=False).values
    conflict = df.groupby("_hash")["label"].transform("nunique") > 1
    n_conflict = int(conflict.sum())
    df = df[~conflict]
    n_dup = int(df["_hash"].duplicated(keep="first").sum())
    df = df.drop_duplicates(subset="_hash", keep="first").drop(columns="_hash").reset_index(drop=True)
    print(f"\n=== Deduplicacao: {n_before:,} -> {len(df):,} amostras "
          f"({n_dup:,} duplicatas removidas; {n_conflict:,} com rotulos conflitantes removidas) ===")

    # Split estratificado por rotulo + fonte, mantendo a proporcao de cada fonte em treino e teste.
    strat = df["label"].astype(str) + "_" + df["source"]
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=strat, random_state=SEED)
    report_balance(df["label"], train_df["label"], test_df["label"])

    print("\n=== Proporcao por fonte (treino x teste) ===")
    comp = pd.DataFrame({"treino": train_df["source"].value_counts(normalize=True),
                         "teste": test_df["source"].value_counts(normalize=True)})
    print((comp * 100).round(3).to_string())

    print("\n=== Checagem de nulos: campo de codigo (somente relato) ===")
    for name, d in (("treino", train_df), ("teste", test_df)):
        print(f"  [{name}] {FIELD_CODE} nulo/nao-texto: {count_non_text(d[FIELD_CODE])}")

    train_df[FIELD_CODE].reset_index(drop=True).to_pickle(OUT / "text_train.pkl")
    test_df[FIELD_CODE].reset_index(drop=True).to_pickle(OUT / "text_test.pkl")
    train_df["label"].reset_index(drop=True).to_pickle(OUT / "y_train.pkl")
    test_df["label"].reset_index(drop=True).to_pickle(OUT / "y_test.pkl")
    train_df["source"].reset_index(drop=True).to_pickle(OUT / "source_train.pkl")
    test_df["source"].reset_index(drop=True).to_pickle(OUT / "source_test.pkl")
    train_df[[FIELD_ID, "label", "source"]].to_csv(OUT / "ids_train.csv", index=False)
    test_df[[FIELD_ID, "label", "source"]].to_csv(OUT / "ids_test.csv", index=False)
    print(f"\nSalvo em {OUT}/ (o TF-IDF so e ajustado depois, no 'features'/'base').")


# =====================================================================
# TF-IDF: vocabulario e IDF aprendidos somente no treino
# =====================================================================
# Carrega os dados preparados pelo comando prepare.
def load_prepared():
    try:
        return (pd.read_pickle(OUT / "text_train.pkl"), pd.read_pickle(OUT / "y_train.pkl"),
                pd.read_pickle(OUT / "text_test.pkl"), pd.read_pickle(OUT / "y_test.pkl"))
    except FileNotFoundError:
        sys.exit("Dados nao encontrados. Rode primeiro: python sban_pipeline_tfidf.py prepare")


# Interrompe a execucao se houver codigo nulo ou nao textual.
def abort_if_nulls(text_train, text_test):
    bad = count_non_text(text_train) + count_non_text(text_test)
    if bad:
        sys.exit(f"ABORTADO: {bad} amostra(s) com codigo nulo/nao-texto (veja a saida do "
                 "'prepare'). O tratamento nao esta definido; decida antes de treinar.")


# Ajusta o TF-IDF no treino e apenas transforma o teste; palavras excluidas saem via stop_words.
def fit_tfidf(text_train, text_test, excluded=()):
    """Aprende vocabulario e IDF no TREINO; o teste so passa por transform.
    Palavras excluidas saem do texto antes da contagem (stop_words), entao IDF,
    normalizacao e max_features sao recalculados sem elas."""
    vec = TfidfVectorizer(stop_words=sorted(excluded) if excluded else None, **TFIDF_PARAMS)
    t0 = time.time()
    log(f"TF-IDF: contando palavras de {len(text_train):,} amostras de treino "
        "(etapa pesada em memoria e tempo; sem barra de progresso) ...")
    X_train = vec.fit_transform(text_train)
    log(f"TF-IDF: treino pronto em {time.time() - t0:.0f}s; transformando {len(text_test):,} de teste ...")
    X_test = vec.transform(text_test)
    log(f"TF-IDF: concluido em {time.time() - t0:.0f}s")
    vocab = pd.DataFrame({"token": vec.get_feature_names_out(), "idf": vec.idf_})
    return vec, X_train, X_test, vocab


# =====================================================================
# ETAPAS 2 e 3: treino, avaliacao (F1, ROC/AUC) e SHAP
# =====================================================================
# Treina o Random Forest.
def train_rf(X, y):
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X, y)
    return model


# Avalia no conjunto de teste: F1 da classe maliciosa, AUC e curva ROC.
def evaluate(model, X_test, y_test, outdir: Path) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    metrics = {
        "f1_malicioso": float(f1_score(y_test, pred, pos_label=1)),
        "auc": float(roc_auc_score(y_test, proba)),
    }
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {metrics['auc']:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="aleatorio")
    plt.xlabel("Taxa de falsos positivos")
    plt.ylabel("Taxa de verdadeiros positivos")
    plt.title("Curva ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outdir / "roc_curve.png", dpi=150)
    plt.close()
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"  F1 (classe maliciosa) = {metrics['f1_malicioso']:.4f} | AUC = {metrics['auc']:.4f}")
    return metrics


# Calcula SHAP em lote, valida a aditividade, gera o ranking e os graficos de barras e enxame.
def shap_analysis(model, X, feature_names, outdir: Path) -> pd.DataFrame:
    """X: matriz esparsa (TF-IDF). So a amostra do SHAP e convertida para densa."""
    rng = np.random.RandomState(SEED)
    idx = rng.choice(X.shape[0], size=min(SHAP_SAMPLE_SIZE, X.shape[0]), replace=False)
    Xs = pd.DataFrame(X[idx].toarray(), columns=feature_names)

    import shap
    depth = max(e.get_depth() for e in model.estimators_)
    log(f"Versoes: shap {shap.__version__} | scikit-learn {sklearn.__version__} | "
        f"numpy {np.__version__} | profundidade maxima das arvores = {depth}")
    explainer = shap.TreeExplainer(model)
    base = float(np.atleast_1d(explainer.expected_value)[-1])
    parts = []
    all_gaps = []
    t0 = time.time()
    for start in range(0, len(Xs), SHAP_BATCH):
        batch = Xs.iloc[start:start + SHAP_BATCH]
        part = explainer.shap_values(batch, check_additivity=False)
        if isinstance(part, list):
            part = part[1]
        elif part.ndim == 3:
            part = part[:, :, 1]

        gap = np.abs(part.sum(axis=1) + base - model.predict_proba(batch.values)[:, 1])
        all_gaps.append(gap)
        bad = int((~np.isfinite(gap) | (gap > SHAP_ADDITIVITY_TOL)).sum())
        log(f"SHAP aditividade (lote {start // SHAP_BATCH + 1}): "
            f"diferenca max = {np.nanmax(gap):.3e} | media = {np.nanmean(gap):.3e} | "
            f"mediana = {np.nanmedian(gap):.3e} | acima da tol. ({SHAP_ADDITIVITY_TOL:g}) = "
            f"{bad}/{len(batch)}")
        if bad:
            sys.exit(
                f"\nABORTADO: o SHAP NAO FECHA com a probabilidade prevista em {bad}/{len(batch)} "
                f"amostras do lote (tolerancia = {SHAP_ADDITIVITY_TOL:g}; maior diferenca = {np.nanmax(gap):.3g}; maior |SHAP| = "
                f"{np.abs(part).max():.3g}).\nO ranking seria lixo numerico e NAO foi gravado. "
                f"As metricas (F1/ROC) ja foram salvas.\nVersoes: shap {shap.__version__}, "
                f"scikit-learn {sklearn.__version__}, numpy {np.__version__}.\n"
                "Causa provavel: versao do shap incompativel com a do scikit-learn/numpy "
                "(tente: pip install -U shap).")
        parts.append(part)
        done = min(start + SHAP_BATCH, len(Xs))
        el = time.time() - t0
        log(f"SHAP: {done}/{len(Xs)} amostras | {el:.0f}s decorridos | "
            f"ETA ~{el / done * (len(Xs) - done):.0f}s")
    sv = np.vstack(parts)
    g = np.concatenate(all_gaps)
    gf = g[np.isfinite(g)]
    log("SHAP aditividade (resumo, todas as %d amostras): max = %.3e | p99 = %.3e | p95 = %.3e | "
        "media = %.3e | mediana = %.3e | tolerancia atual = %g | nao finitos = %d"
        % (len(g), gf.max(), np.percentile(gf, 99), np.percentile(gf, 95),
           gf.mean(), np.median(gf), SHAP_ADDITIVITY_TOL, len(g) - len(gf)))
    log(f"SHAP: maior |SHAP| = {np.abs(sv).max():.3g}")

    ranking = (pd.DataFrame({"feature": Xs.columns,
                             "mean_abs_shap": np.abs(sv).mean(axis=0)})
               .sort_values("mean_abs_shap", ascending=False)
               .reset_index(drop=True))
    ranking.index += 1
    ranking.to_csv(outdir / "shap_ranking.csv", index_label="rank")

    plt.figure()
    shap.summary_plot(sv, Xs, plot_type="bar", max_display=30, show=False)
    plt.tight_layout()
    plt.savefig(outdir / "shap_bar.png", dpi=150)
    plt.close()
    plt.figure()
    shap.summary_plot(sv, Xs, max_display=30, show=False)
    plt.tight_layout()
    plt.savefig(outdir / "shap_beeswarm.png", dpi=150)
    plt.close()

    print(f"\n  Ranking SHAP (top 30 de {len(ranking)}; impacto na classe MALICIOSO):")
    print(ranking.head(30).to_string())
    return ranking


# Executa uma rodada completa (TF-IDF, treino, avaliacao e SHAP) e grava os resultados em OUT/tag.
def run_stage(tag: str, excluded: list):
    outdir = OUT / tag
    outdir.mkdir(parents=True, exist_ok=True)
    text_train, y_train, text_test, y_test = load_prepared()
    abort_if_nulls(text_train, text_test)

    print(f"\n=== {tag}: ajustando TF-IDF no treino ({len(excluded)} palavras excluidas) ===")
    vec, X_train, X_test, vocab = fit_tfidf(text_train, text_test, excluded)
    print(f"  vocabulario: {len(vocab)} palavras | treino {X_train.shape} | teste {X_test.shape}")
    vocab.to_csv(outdir / "vocabulary.csv", index=False)
    if tag == "base":
        vocab.to_csv(OUT / "vocabulary_base.csv", index=False)

    t0 = time.time()
    log(f"Random Forest: treinando {RF_PARAMS['n_estimators']} arvores em {X_train.shape[0]:,} "
        f"amostras x {X_train.shape[1]} palavras (esparsa) ...")
    model = train_rf(X_train, y_train)
    log(f"Random Forest: treino concluido em {time.time() - t0:.0f}s")
    evaluate(model, X_test, y_test, outdir)
    log("Calculando SHAP ...")
    X_shap = X_test if SHAP_SPLIT == "test" else X_train
    shap_analysis(model, X_shap, vocab["token"].tolist(), outdir)
    (outdir / "excluded_tokens.json").write_text(json.dumps(sorted(excluded), indent=2))
    print(f"\nArquivos em {outdir}/")


# Comando features: aprende e lista o vocabulario base.
def cmd_features():
    text_train, _, text_test, _ = load_prepared()
    abort_if_nulls(text_train, text_test)
    _, _, _, vocab = fit_tfidf(text_train, text_test)
    OUT.mkdir(exist_ok=True)
    vocab.to_csv(OUT / "vocabulary_base.csv", index=False)
    print(f"Vocabulario base: {len(vocab)} palavras (aprendido SO no treino).")
    print("Lista completa: " + str(OUT / "vocabulary_base.csv"))
    print("\nPalavras mais comuns (menor IDF):")
    print(vocab.sort_values("idf").head(30).to_string(index=False))
    print("\nPalavras mais raras (maior IDF):")
    print(vocab.sort_values("idf", ascending=False).head(30).to_string(index=False))


# =====================================================================
# ETAPA 4: analise exploratoria (somente treino) com Sweetviz
# =====================================================================
# Carrega apenas os dados de treino (texto, rotulo e fonte).
def load_train():
    try:
        text = pd.read_pickle(OUT / "text_train.pkl")
        y = pd.read_pickle(OUT / "y_train.pkl")
    except FileNotFoundError:
        sys.exit("Dados nao encontrados. Rode primeiro: python sban_pipeline_tfidf.py prepare")
    try:
        src = pd.read_pickle(OUT / "source_train.pkl")
    except FileNotFoundError:
        sys.exit("Falta a coluna de FONTE do treino. Rode 'prepare' de novo: com a mesma semente a "
                 "divisao treino/teste e identica a anterior, so se acrescenta a fonte.")
    return text, y, src


# Monta a tabela de treino e gera os relatorios Sweetviz (geral e benigno x malicioso).
def cmd_eda():
    """EDA feita 100% pela biblioteca Sweetviz (nenhuma estatistica calculada a mao).
    O script so monta a TABELA que a biblioteca analisa: uma linha por amostra do TREINO, com
    fonte, rotulo, tamanho do texto e flags de nulo/duplicata. O texto bruto NAO entra na tabela."""
    text_raw, y, src = load_train()
    if y.nunique() < 2:
        sys.exit("ERRO: o treino precisa ter as duas classes.")
    try:
        import sweetviz as sv
    except ImportError:
        sys.exit("Biblioteca ausente. Instale com: pip install sweetviz")
    outdir = OUT / "eda"
    outdir.mkdir(parents=True, exist_ok=True)
    import logging
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    log("EDA 1/3: montando a tabela do treino ...")
    is_str = text_raw.map(lambda x: isinstance(x, str))
    text = text_raw.where(is_str, "")
    df = pd.DataFrame({
        "fonte": src.astype(object).values,
        "rotulo": y.values,
        "n_chars": text.str.len().values,
        "n_lines": (text.str.count("\n") + 1).values,
        "texto_nulo_ou_vazio": np.where(~is_str | (text.str.strip() == ""), "sim", "nao"),
        "texto_duplicado": np.where(
            pd.util.hash_pandas_object(text, index=False).duplicated(keep=False), "sim", "nao"),
    })
    for c in ("fonte", "texto_nulo_ou_vazio", "texto_duplicado"):
        df[c] = df[c].astype(object)
    cfg = sv.FeatureConfig(force_cat=["fonte", "texto_nulo_ou_vazio", "texto_duplicado"])

    log(f"EDA 2/3: relatorio geral (Sweetviz, {len(df):,} amostras) ...")
    try:
        rep_geral = sv.analyze(df, target_feat="rotulo", feat_cfg=cfg, pairwise_analysis=EDA_PAIRWISE)
        rep_geral.show_html(filepath=str(outdir / "eda_geral.html"), open_browser=False)

        log("EDA 3/3: relatorio benigno x malicioso (Sweetviz) ...")
        rep_cmp = sv.compare_intra(df.drop(columns="rotulo"), df["rotulo"] == 1,
                                   ["Maliciosa", "Benigna"], feat_cfg=cfg,
                                   pairwise_analysis=EDA_PAIRWISE)
        rep_cmp.show_html(filepath=str(outdir / "eda_benigno_vs_malicioso.html"), open_browser=False)
    except Exception as e:
        sys.exit(f"ERRO do Sweetviz ({type(e).__name__}): {e}\n"
                 f"Versoes: sweetviz {getattr(sv, '__version__', '?')}, pandas {pd.__version__}, "
                 f"numpy {np.__version__}. Se for incompatibilidade de versao, tente "
                 "'pip install -U sweetviz' ou ajuste pandas/numpy.")

    print(f"\n=== EDA (treino, n={len(df):,}) ===")
    print(f"Relatorios: {outdir / 'eda_geral.html'}")
    print(f"            {outdir / 'eda_benigno_vs_malicioso.html'}")


# =====================================================================
# Gerenciamento das palavras excluidas (iteracao SHAP, nao destrutivo)
# =====================================================================
EXCL_FILE = OUT / "excluded_tokens.json"


# Le a lista de palavras excluidas (vazia se o arquivo nao existir).
def load_excluded() -> list:
    return json.loads(EXCL_FILE.read_text()) if EXCL_FILE.exists() else []


# Grava a lista de palavras excluidas, sem repeticoes e ordenada.
def save_excluded(lst):
    OUT.mkdir(exist_ok=True)
    EXCL_FILE.write_text(json.dumps(sorted(set(lst)), indent=2))


# Resolve curingas contra o vocabulario base, respeitando maiusculas e minusculas.
def match(patterns):
    """Casa curingas (*, ?) com o vocabulario BASE, diferenciando maiusculas."""
    path = OUT / "vocabulary_base.csv"
    if not path.exists():
        sys.exit("Vocabulario base nao encontrado. Rode 'features' (ou 'base') primeiro.")
    vocab = pd.read_csv(path, keep_default_na=False)["token"].tolist()
    found = set()
    for p in patterns:
        hits = [t for t in vocab if fnmatch.fnmatchcase(t, p)]
        if not hits:
            print(f"  aviso: '{p}' nao corresponde a nenhuma palavra do vocabulario base")
        found.update(hits)
    return found


# Adiciona palavras a lista de excluidas.
def cmd_exclude(patterns):
    cur = set(load_excluded()) | match(patterns)
    save_excluded(cur)
    print(f"Excluidas agora ({len(cur)}): {sorted(cur)}")


# Remove palavras da lista de excluidas.
def cmd_restore(patterns):
    cur = set(load_excluded()) - match(patterns)
    save_excluded(cur)
    print(f"Excluidas agora ({len(cur)}): {sorted(cur)}")


# Executa uma nova iteracao numerada com a lista atual de exclusoes.
def cmd_iterate():
    OUT.mkdir(exist_ok=True)
    k = len(list(OUT.glob("iter_*"))) + 1
    run_stage(f"iter_{k:02d}", load_excluded())


# =====================================================================
# Ponto de entrada: interpreta os subcomandos da linha de comando
# =====================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    sub.add_parser("features")
    sub.add_parser("base")
    sub.add_parser("eda")
    sub.add_parser("iterate")
    sub.add_parser("exclude").add_argument("patterns", nargs="+")
    sub.add_parser("restore").add_argument("patterns", nargs="+")
    a = p.parse_args()

    if a.cmd in ("exclude", "restore", "iterate") and not ITERATION_ENABLED:
        sys.exit("Iteracao SHAP INATIVA (ITERATION_ENABLED = False). "
                 "Ative a flag no topo do arquivo quando quiser usar.")

    if a.cmd == "prepare":
        cmd_prepare()
    elif a.cmd == "features":
        cmd_features()
    elif a.cmd == "base":
        run_stage("base", [])
    elif a.cmd == "eda":
        cmd_eda()
    elif a.cmd == "iterate":
        cmd_iterate()
    elif a.cmd == "exclude":
        cmd_exclude(a.patterns)
    elif a.cmd == "restore":
        cmd_restore(a.patterns)


if __name__ == "__main__":
    main()