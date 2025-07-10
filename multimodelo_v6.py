import streamlit as st
import pandas as pd
import numpy as np
import shap
import lime.lime_tabular
import pickle
import base64
from io import BytesIO
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report, 
    roc_curve, auc, confusion_matrix, log_loss, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression, SGDClassifier, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.multiclass import unique_labels
from yattag import Doc
import datetime
from scipy.stats import shapiro
from streamlit.components.v1 import html as components_html
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================
# Configuração inicial
# =============================================
def setup_page():
    st.set_page_config(layout="wide")
    st.title("AutoML Interativo com Streamlit")

# =============================================
# Classes customizadas
# =============================================
class StringConverter(BaseEstimator, TransformerMixin):
    """Transformer para converter colunas para string mantendo os nomes das features"""
    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns if hasattr(X, 'columns') else None
        return self
    
    def transform(self, X):
        return X.astype(str)
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, 'feature_names_in_', None)
        return input_features

# =============================================
# Funções utilitárias
# =============================================
def detect_problem_type(y, threshold=20, max_fraction_unique=0.1):
    """
    Versão corrigida com detecção mais precisa de variáveis contínuas.
    """
    # 1. Verificação para tipos não-numéricos (sempre classificação)
    if pd.api.types.is_string_dtype(y) or pd.api.types.is_categorical_dtype(y):
        return 'classification'
    
    # 2. Verificação para booleanos (sempre classificação)
    if pd.api.types.is_bool_dtype(y):
        return 'classification'
    
    # 3. Se for numérico, faz verificações mais inteligentes
    if pd.api.types.is_numeric_dtype(y):
        unique_values = y.nunique()
        total_values = len(y)
        
        # 3.1. Verifica se parece ser categórico (rótulos codificados)
        if pd.api.types.is_integer_dtype(y):
            # Caso especial: se for 0 e 1 (binário)
            if set(y.dropna().unique()).issubset({0, 1}):
                return 'classification'
            
            # Verifica se os valores são sequenciais começando em 0 (como rótulos)
            unique_sorted = np.sort(y.dropna().unique())
            if np.array_equal(unique_sorted, np.arange(len(unique_sorted))):
                return 'classification'
        
        # 3.2. Verifica se tem poucos valores únicos (potencial categórico)
        if (unique_values <= threshold) and (unique_values / total_values <= max_fraction_unique):
            return 'classification'
    
    # 4. Todos os outros casos são regressão
    return 'regression'

def build_pipeline(model, num_cols, cat_cols, sampler=None):
    model_name = type(model).__name__
    is_classifier = any(keyword in model_name for keyword in ['Classifier', 'SVC', 'NB'])
    is_regressor = any(keyword in model_name for keyword in ['Regressor', 'Regression', 'SVR'])
    
    if not is_classifier and not is_regressor:
        raise ValueError(f"Não foi possível determinar o tipo do modelo {model_name}")
    
    transformers = []
    no_scaling_needed = [
        'DecisionTree', 'RandomForest', 'GradientBoosting',
        'XGB', 'LGBM', 'GaussianNB'
    ]
    
    normalize = not any(keyword in model_name for keyword in no_scaling_needed)
    
    # Processamento normal para outros modelos
    if normalize and num_cols:
        transformers.append(('num', StandardScaler(), num_cols))
    elif num_cols:
        transformers.append(('num', 'passthrough', num_cols))
    
    if cat_cols:
        drop_param = 'first' if is_classifier else None
        transformers.append(('cat', Pipeline([
            ('to_string', StringConverter()),
            ('encoder', OneHotEncoder(
                handle_unknown='infrequent_if_exist',
                sparse_output=False,
                drop=drop_param
            ))
        ]), cat_cols))
    
    preprocessor = ColumnTransformer(transformers, remainder='passthrough')
    
    steps = [('pre', preprocessor)]
    if is_classifier and sampler:
        steps.append(('sampler', sampler))
    steps.append(('clf', model))
    
    if is_classifier and sampler:
        pipeline = ImbPipeline(steps)
    else:
        pipeline = Pipeline(steps)
    
    return pipeline

def download_link(obj, filename, text):
    buffer = BytesIO()
    pickle.dump(obj, buffer)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:file/output_model;base64,{b64}" download="{filename}">{text}</a>'
    return href

def download_text_file(content, filename, text):
    buffer = BytesIO()
    buffer.write(content.encode())
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:file/text;base64,{b64}" download="{filename}">{text}</a>'
    return href

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def plot_metric_comparison(resultados, metric_name):
    df = pd.DataFrame(resultados)
    df = df.sort_values(by=metric_name, ascending=False)

    # Definir tamanho fixo e DPI para consistência
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    sns.barplot(data=df, x='Modelo', y=metric_name, palette="Blues_d", ax=ax)
    ax.set_title(f"Comparação de {metric_name} entre Modelos", pad=20)
    ax.set_ylim(0, 1 if metric_name == 'R²' else None)
    ax.set_xlabel("Modelos")
    plt.xticks(rotation=45)
    
    # Ajustar layout para evitar cortes
    plt.tight_layout()

    for p in ax.patches:
        value = p.get_height()
        if not pd.isna(value):
            ax.annotate(f"{value:.4f}", 
                       (p.get_x() + p.get_width() / 2., value),
                       ha='center', va='center', 
                       xytext=(0, 8), 
                       textcoords='offset points',
                       fontsize=10, color='black')
    return fig

def plot_error_comparison(resultados, metric_name):
    df = pd.DataFrame(resultados)
    df = df.sort_values(by=metric_name, ascending=True)
    df_melted = df.melt(id_vars=['Modelo'], value_vars=['RMSE', 'MAE'], 
                        var_name='Erro', value_name='Valor')

    # Usar o mesmo tamanho e DPI que a outra função
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    sns.barplot(data=df_melted, x='Modelo', y='Valor', hue='Erro', 
                palette=['#3498db', '#e74c3c'], ax=ax)
    ax.set_title("Comparação de Erros entre Modelos", pad=20)
    ax.set_ylabel("Valor do Erro")
    ax.set_xlabel("Modelos")
    plt.xticks(rotation=45)

    for p in ax.patches:
        value = p.get_height()
        if not pd.isna(value):
            ax.annotate(f"{value:.4f}", 
                       (p.get_x() + p.get_width() / 2., value),
                       ha='center', va='center', 
                       xytext=(0, 8), 
                       textcoords='offset points',
                       fontsize=10, color='black')
    
    ax.legend(title="Tipo de Erro", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig

# =============================================
# Funções de relatório
# =============================================
def generate_regression_html_report(resultados, X_test, y_test):
    doc, tag, text = Doc().tagtext()
    
    with tag('html'):
        with tag('head'):
            doc.stag('meta', charset="utf-8")
            with tag('title'): text("Relatório de Modelos de Regressão")
            with tag('style'):
                text("""
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1, h2 { color: #333; }
                    table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                    .metric { font-weight: bold; color: #2c3e50; }
                    img { max-width: 100%; height: auto; margin: 10px 0; }
                    .image-pair-container {
                        display: flex;
                        flex-wrap: wrap;
                        justify-content: space-around;
                        gap: 20px;
                        margin-top: 20px;
                    }
                    .image-item {
                        flex: 1;
                        min-width: 300px;
                        max-width: 48%;
                        box-sizing: border-box;
                        text-align: center;
                    }
                    @media (max-width: 768px) {
                        .image-item {
                            max-width: 100%;
                        }
                    }
                """)
        
        with tag('body'):
            with tag('h1'): text("Relatório Comparativo de Modelos de Regressão")
            with tag('p'): text(f"Gerado em: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            with tag('h2'): text("Comparação de Desempenho")
            with tag('table'):
                with tag('thead'):
                    with tag('tr'):
                        with tag('th'): text("Modelo")
                        with tag('th'): text("R²")
                        with tag('th'): text("RMSE")
                        with tag('th'): text("MAE")
                        with tag('th'): text("MSE")
                with tag('tbody'):
                    for resultado in sorted(resultados, key=lambda x: x['R²'], reverse=True):
                        with tag('tr'):
                            with tag('td'): text(resultado['Modelo'])
                            with tag('td', klass="metric"): text(f"{resultado['R²']:.4f}")
                            with tag('td'): text(f"{resultado['RMSE']:.4f}")
                            with tag('td'): text(f"{resultado['MAE']:.4f}")
                            with tag('td'): text(f"{resultado['MSE']:.4f}")
            
            with tag('h2'): text("Visualizações")
            with tag('div', klass="image-pair-container"):
                with tag('div', klass="image-item"):
                    with tag('h3'): text("Comparação de R² entre Modelos")
                    fig_r2 = plot_metric_comparison(resultados, 'R²')
                    doc.stag('img', src=f"data:image/png;base64,{fig_to_base64(fig_r2)}")
                    plt.close(fig_r2)
                with tag('div', klass="image-item"):
                    with tag('h3'): text("Comparação de Métricas de Erro")
                    fig_errors = plot_error_comparison(resultados, 'RMSE')
                    doc.stag('img', src=f"data:image/png;base64,{fig_to_base64(fig_errors)}")
                    plt.close(fig_errors)
            
            with tag('h2'): text("Detalhes por Modelo")
            for resultado in resultados:
                with tag('div', klass="model-section"):
                    with tag('h3'): text(resultado['Modelo'])
                    with tag('p'):
                        with tag('span', klass="metric"): text("R²: ")
                        text(f"{resultado['R²']:.4f} | ")
                        with tag('span', klass="metric"): text("RMSE: ")
                        text(f"{resultado['RMSE']:.4f} | ")
                        with tag('span', klass="metric"): text("MAE: ")
                        text(f"{resultado['MAE']:.4f}")
                    
                    with tag('h4'): text("Gráfico de Dispersão")
                    doc.stag('img', src=f"data:image/png;base64,{fig_to_base64(resultado['Gráfico'])}")
                    
                    with tag('h4'): text("Melhores Parâmetros")
                    with tag('pre'): text(str(resultado['Melhores Params']))
    
    return doc.getvalue()

# Função auxiliar para exibir gráficos com layout responsivo
def display_plots(plots):
    num_plots = len(plots)
    
    if num_plots == 1:
        # 1 gráfico: exibir em 2 colunas (ocupando metade da largura)
        col1, col2 = st.columns(2)
        with col1:
            st.image(BytesIO(base64.b64decode(plots[0][1])), 
                   caption=plots[0][0], use_column_width=True)
        with col2:
            st.empty()  # Espaço vazio para balancear
    
    elif num_plots == 2:
        # 2 gráficos: exibir em 2 colunas
        cols = st.columns(2)
        for idx, (title, plot) in enumerate(plots):
            with cols[idx]:
                st.image(BytesIO(base64.b64decode(plot)), 
                       caption=title, use_column_width=True)
    
    elif num_plots >= 3:
        # 3+ gráficos: exibir os 3 primeiros em 3 colunas
        cols = st.columns(3)
        for idx, (title, plot) in enumerate(plots[:3]):
            with cols[idx]:
                st.image(BytesIO(base64.b64decode(plot)), 
                       caption=title, use_column_width=True)
        
        # Se houver mais de 3, exibir os restantes em linhas abaixo
        for title, plot in plots[3:]:
            st.image(BytesIO(base64.b64decode(plot)), 
                   caption=title, use_column_width=True)

def generate_classification_html_report(resultados, roc_curves_data, le, interest_classes_selected, X_test, y_test):
    doc, tag, text = Doc().tagtext()
    doc.asis('<!DOCTYPE html>')
    
    with tag('html'):
        with tag('head'):
            doc.stag('meta', charset="utf-8")
            with tag('title'): text("Relatório de Modelos - Machine Learning")
            doc.asis('''
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                h2 { color: #555; border-bottom: 1px solid #eee; padding-bottom: 5px; margin-top: 20px; }
                h3 { color: #777; margin-top: 15px; }
                pre { background-color: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }
                table { width: 100%; border-collapse: collapse; margin-top: 20px; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                
                /* Layout principal */
                .container {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                }
                
                /* Container para a curva ROC (50% da largura) */
                .roc-container {
                    flex: 0 0 calc(50% - 20px);
                    max-width: calc(50% - 20px);
                }
                
                /* Container para os modelos */
                .model-container {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    margin-bottom: 30px;
                }
                
                /* Cada modelo */
                .model-box {
                    flex: 1 1 100%;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 15px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }
                
                /* Container para os 3 gráficos do modelo */
                .model-plots {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 15px;
                    margin: 15px 0;
                }
                
                /* Cada gráfico do modelo (3 por linha) */
                .model-plot {
                    flex: 0 0 calc(33.33% - 15px);
                    max-width: calc(33.33% - 15px);
                    box-sizing: border-box;
                }
                
                /* Imagens responsivas */
                img {
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #eee;
                }
                
                /* Destaques para tabela */
                .max-value { background-color: #e6f7e6; }
                .min-value { background-color: #ffe6e6; }
                
                /* Responsividade */
                @media (max-width: 992px) {
                    .roc-container {
                        flex: 0 0 100%;
                        max-width: 100%;
                    }
                    .model-plot {
                        flex: 0 0 calc(50% - 15px);
                        max-width: calc(50% - 15px);
                    }
                }
                
                @media (max-width: 768px) {
                    .model-plot {
                        flex: 0 0 100%;
                        max-width: 100%;
                    }
                }
            </style>
            ''')

        with tag('body'):
            with tag('h1'): text("Relatório de Avaliação de Modelos")
            with tag('p'): text(f"Gerado em: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            # Seção de comparação geral
            with tag('h2'): text("Comparação Geral dos Modelos")
            
            # Tabela comparativa
            df_resultados = pd.DataFrame(resultados)
            # Ordenar por Acurácia (descendente) ANTES de gerar a tabela HTML
            df_resultados = df_resultados.sort_values('Acurácia', ascending=False)
            cols_para_exibir = ['Modelo', 'Acurácia', 'AUC', 'PR-AUC', 'F1-Score', 'Precisão', 'Recall', 'Optimal Threshold']
            cols_presentes = [col for col in cols_para_exibir if col in df_resultados.columns]
            
            with tag('table'):
                with tag('thead'):
                    with tag('tr'):
                        for col in cols_presentes:
                            with tag('th'): text(col)
                
                with tag('tbody'):
                    for _, row in df_resultados.iterrows():
                        with tag('tr'):
                            for col in cols_presentes:
                                cell_value = row[col]
                                if isinstance(cell_value, float):
                                    cell_text = f"{cell_value:.4f}"
                                else:
                                    cell_text = str(cell_value)
                                
                                cell_class = ""
                                if col != 'Modelo':
                                    if cell_value == df_resultados[col].max():
                                        cell_class = "max-value"
                                    elif cell_value == df_resultados[col].min():
                                        cell_class = "min-value"
                                
                                with tag('td', klass=cell_class): 
                                    text(cell_text)
            
            # Container principal para a seção de gráficos
            with tag('div', klass="container"):
                # Curva ROC (50% da largura)
                if roc_curves_data:
                    with tag('div', klass="roc-container"):
                        with tag('h2'): text("Comparação de Curvas ROC")
                        fig_combined, ax_combined = plt.subplots(figsize=(10, 8))
                        for curve in roc_curves_data:
                            roc_plot_label = f"{curve['model']} (AUC = {curve['auc']:.3f})"
                            if curve['plot_type'] == 'custom_binary':
                                int_lbls = ", ".join(map(str, curve['interest_classes'])) 
                                comp_lbls = ", ".join(map(str, curve['complementary_classes'])) 
                                roc_plot_label = f"{curve['model']} (AUC = {curve['auc']:.3f}) - Interesse: [{int_lbls}], Compl.: [{comp_lbls}]" 
                            ax_combined.plot(curve['fpr'], curve['tpr'], lw=2, label=roc_plot_label)
                        
                        ax_combined.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance')
                        ax_combined.set_xlim([0.0, 1.0])
                        ax_combined.set_ylim([0.0, 1.05])
                        ax_combined.set_xlabel('Taxa de Falsos Positivos')
                        ax_combined.set_ylabel('Taxa de Verdadeiros Positivos')
                        ax_combined.set_title('Comparação de Curvas ROC dos Modelos')
                        ax_combined.legend(loc='lower right', fontsize='small') 
                        plt.tight_layout() 
                        ax_combined.grid(alpha=0.3)
                        
                        buf = BytesIO()
                        fig_combined.savefig(buf, format="png", bbox_inches="tight")
                        buf.seek(0)
                        b64 = base64.b64encode(buf.read()).decode()
                        plt.close(fig_combined)
                        
                        doc.stag('img', src=f"data:image/png;base64,{b64}")
            
            # Detalhes por modelo
            with tag('h2'): text("Detalhes por Modelo")
            for resultado in resultados:
                with tag('div', klass="model-box"):
                    with tag('h3'): text(resultado['Modelo'])
                    
                    # Métricas resumidas
                    with tag('div', klass="metrics"):
                        with tag('p'):
                            try:
                                text(f"Acurácia: {resultado.get('Acurácia', 'N/A'):.4f} | "
                                     f"Precisão: {resultado.get('Precisão', 'N/A'):.4f} | "
                                     f"Recall: {resultado.get('Recall', 'N/A'):.4f} | "
                                     f"F1-Score: {resultado.get('F1-Score', 'N/A'):.4f}")
                            except (KeyError, ValueError) as e:
                                text(f"Erro ao exibir métricas: {str(e)}")
                        
                        if not pd.isna(resultado.get('AUC', np.nan)):
                            with tag('p'): text(f"AUC: {resultado['AUC']:.4f}")
                        
                        if not pd.isna(resultado.get('PR-AUC', np.nan)):
                            with tag('p'): text(f"PR-AUC: {resultado['PR-AUC']:.4f}")
                    
                    # Container para os 3 gráficos
                    with tag('div', klass="model-plots"):
                        # Matriz de Confusão
                        if 'Confusion Matrix' in resultado:
                            with tag('div', klass="model-plot"):
                                with tag('h4'): text("Matriz de Confusão")
                                doc.stag('img', src=f"data:image/png;base64,{resultado['Confusion Matrix']}")
                        
                        # Curva ROC
                        if 'ROC Curve' in resultado:
                            with tag('div', klass="model-plot"):
                                with tag('h4'): text("Curva ROC")
                                doc.stag('img', src=f"data:image/png;base64,{resultado['ROC Curve']}")
                        
                        # Curva Precision-Recall
                        if 'PR Curve' in resultado:
                            with tag('div', klass="model-plot"):
                                with tag('h4'): text("Curva Precision-Recall")
                                doc.stag('img', src=f"data:image/png;base64,{resultado['PR Curve']}")
                    
                    # Parâmetros
                    with tag('h4'): text("Melhores Parâmetros")
                    with tag('pre'): text(str(resultado['Melhores Params']))
    
    return doc.getvalue()
    
# =============================================
# Funções de análise de dados
# =============================================
def perform_eda(df_to_analyze):

    st.markdown("""
    <style>
    h3 {
        font-size: 24px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.subheader("Distribuição de Variáveis Numéricas")
    with st.expander("Clique para ver os Histogramas das Variáveis Numéricas"):
        num_cols_auto = df_to_analyze.select_dtypes(include=np.number).columns
        
        if len(num_cols_auto) == 0:
            st.warning("Nenhuma coluna numérica encontrada para análise.")
            return
            
        if not num_cols_auto.empty:
            current_cols = st.columns(3)
            for i, col in enumerate(num_cols_auto):
                with current_cols[i % 3]:
                    try:
                        # Converter para float e remover NaNs
                        clean_data = df_to_analyze[col].dropna().astype(float)
                        
                        if len(clean_data) == 0:
                            st.warning(f"Dados vazios na coluna {col}")
                            continue
                            
                        fig, ax = plt.subplots(figsize=(6, 4))
                        
                        # Verificar amplitude dos dados
                        data_range = clean_data.max() - clean_data.min()
                        
                        if data_range > 1e6:  # Para dados com grande variação
                            # Usar escala logarítmica
                            log_data = np.log1p(clean_data)
                            sns.histplot(log_data, kde=True, ax=ax)
                            ax.set_title(f'Distribuição (log) - {col}', fontsize=10)
                        else:
                            # Usar escala normal com limite de bins
                            sns.histplot(clean_data, kde=True, ax=ax, bins=50)
                            ax.set_title(f'Distribuição - {col}', fontsize=10)
                        
                        ax.tick_params(axis='x', labelsize=8)
                        ax.tick_params(axis='y', labelsize=8)
                        st.pyplot(fig)
                        plt.close(fig)
                        
                    except Exception as e:
                        st.error(f"Erro ao plotar {col}: {str(e)}")
                        continue
        else:
            st.info("Nenhuma coluna numérica encontrada para plotar histogramas.")

    st.subheader("Contagem de Variáveis Categóricas (Top 20 categorias)")
    with st.expander("Clique para ver os Gráficos de Contagem das Variáveis Categóricas"):
        cat_cols_auto = df_to_analyze.select_dtypes(exclude=np.number).columns
        if not cat_cols_auto.empty:
            current_cols = st.columns(3)
            for i, col in enumerate(cat_cols_auto):
                with current_cols[i % 3]:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    top_values = df_to_analyze[col].value_counts().nlargest(20)
                    if not top_values.empty:
                        sns.barplot(x=top_values.index.astype(str), y=top_values.values, ax=ax)
                        ax.set_title(f'Contagem - {col} (Top {min(20, len(top_values))})', fontsize=10)
                        ax.set_ylabel('Frequência', fontsize=8)
                        ax.set_xlabel(col, fontsize=8)
                        ax.tick_params(axis='x', rotation=45, labelsize=7) 
                        ax.tick_params(axis='y', labelsize=8)
                        for j, v in enumerate(top_values.values):
                            ax.text(j, v + max(top_values.values)*0.01, str(v), ha='center', va='bottom', fontsize=6)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.info(f"Coluna '{col}' não possui valores categóricos para plotar contagem.")
        else:
            st.info("Nenhuma coluna categórica encontrada para plotar contagens.")
    
    st.subheader("Matriz de Correlação")
    with st.expander("Clique para ver a Matriz de Correlação"):
        num_cols_auto = df_to_analyze.select_dtypes(include=np.number).columns
        
        if len(num_cols_auto) >= 2:
            col1, col2 = st.columns(2) 
            with col1: 
                corr = df_to_analyze[num_cols_auto].corr()
                if not corr.empty:
                    fig_width = max(8, len(num_cols_auto) * 0.6) 
                    fig_height = max(6, len(num_cols_auto) * 0.5) 
                    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax, annot_kws={"size": 6}) 
                    ax.set_title("Matriz de Correlação", fontsize=12)
                    ax.tick_params(axis='x', labelsize=8, rotation=45) 
                    ax.tick_params(axis='y', labelsize=8, rotation=0)
                    plt.tight_layout() 
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("Não foi possível calcular a matriz de correlação.")
        else:
            st.info("São necessárias pelo menos duas colunas numéricas para gerar a matriz de correlação.")
    
    # Análise de Multicolinearidade (VIF)
    st.subheader("Análise de Multicolinearidade (VIF)")
    with st.expander("Clique para ver a análise de multicolinearidade"):
        num_cols = df_to_analyze.select_dtypes(include=np.number).columns.tolist()
        
        # Verificação robusta para cálculo do VIF
        if len(num_cols) < 2:
            st.warning("São necessárias pelo menos duas colunas numéricas para calcular o VIF.")
        else:
            try:
                # Pré-processamento mais rigoroso
                numeric_df = df_to_analyze[num_cols].dropna()
                
                # Remove colunas com zero variância ou constantes
                numeric_df = numeric_df.loc[:, numeric_df.nunique() > 1]
                numeric_df = numeric_df.loc[:, numeric_df.std() > 0]
                
                if len(numeric_df.columns) < 2:
                    st.warning("Não há colunas numéricas com variação suficiente para cálculo do VIF.")
                    return
                
                # Verificação adicional para garantir dados válidos
                if numeric_df.empty:
                    st.warning("Dados numéricos insuficientes após pré-processamento.")
                    return
                
                # Cálculo seguro do VIF com tratamento de erro específico
                vif_data = pd.DataFrame(columns=["Variável", "VIF"])
                
                for i, col in enumerate(numeric_df.columns):
                    try:
                        # Calcula VIF para cada coluna individualmente
                        vif = variance_inflation_factor(numeric_df.values, i)
                        vif_data.loc[i] = [col, vif]
                    except Exception as e:
                        st.warning(f"Não foi possível calcular VIF para {col}: {str(e)}")
                        continue
                
                if not vif_data.empty:
                    # Ordena por VIF
                    vif_data = vif_data.sort_values(by="VIF", ascending=False)
                    
                    # Formatação condicional
                    def color_vif(val):
                        if val > 10:
                            return 'background-color: #ffcccc'
                        elif val > 5:
                            return 'background-color: #fff3cd'
                        return ''
                    
                    st.write("""**Interpretação do VIF (Variance Inflation Factor):**
                    - **VIF < 5**: Multicolinearidade baixa (geralmente aceitável)
                    - **5 ≤ VIF ≤ 10**: Multicolinearidade moderada (pode ser problemática)
                    - **VIF > 10**: Multicolinearidade alta (deve ser tratada)
                    """)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(
                            vif_data.style.applymap(color_vif, subset=['VIF'])
                            .format({'VIF': '{:.2f}'})
                        )
                        
                    with col2:
                        st.empty()  # Espaço vazio para balancear
                                            
                    # Verifica multicolinearidade alta
                    high_vif_cols = vif_data[vif_data["VIF"] > 10]["Variável"].tolist()
                    if high_vif_cols:
                        st.warning(f"⚠️ **Atenção:** As seguintes variáveis apresentam alta multicolinearidade (VIF > 10): {', '.join(high_vif_cols)}")
                        st.info("""**Sugestões para tratar multicolinearidade:**
                    1. Remover uma das variáveis altamente correlacionadas
                    2. Criar uma nova variável combinando as correlacionadas
                    3. Aplicar técnicas de redução de dimensionalidade (PCA)
                    """)
                    else:
                        st.success("✅ Nenhuma variável com multicolinearidade alta (VIF > 10) detectada.")
                else:
                    st.info("São necessárias pelo menos duas colunas numéricas para calcular o VIF.")
                    
            except Exception as e:
                st.error(f"Erro ao calcular VIF: {str(e)}")
                st.error("Recomendações:")
                st.error("1. Verifique se há colunas com valores constantes")
                st.error("2. Verifique se há valores NaN/Inf nos dados")
                st.error("3. Tente reduzir o número de colunas analisadas")

    # Análise Fatorial (PCA simplificado)
    st.subheader("Análise Fatorial (Componentes Principais)")
    with st.expander("Clique para ver a análise fatorial"):
        num_cols = df_to_analyze.select_dtypes(include=np.number).columns
        
        # Criar cópia do dataframe apenas com colunas numéricas
        df_num = df_to_analyze[num_cols].copy()
        
        # 1. Remover colunas constantes
        constant_cols = [col for col in df_num.columns if df_num[col].nunique() == 1]
        if constant_cols:
            st.warning(f"⚠️ Removendo colunas constantes: {', '.join(constant_cols)}")
            df_num = df_num.drop(columns=constant_cols)
        
        # 2. Remover colunas altamente correlacionadas
        corr_matrix = df_num.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        if to_drop:
            st.warning(f"⚠️ Removendo colunas com alta correlação (>0.95): {', '.join(to_drop)}")
            df_num = df_num.drop(columns=to_drop)
        
        # Atualizar lista de colunas numéricas após tratamento
        num_cols = df_num.columns
        
        if len(num_cols) >= 3:  # Mínimo de 3 variáveis para análise fatorial
            # Testes de adequação
            st.write("**Testes de Adequação para Análise Fatorial:**")
            
            try:
                # Teste de esfericidade de Bartlett
                bartlett, p_value = calculate_bartlett_sphericity(df_to_analyze[num_cols].dropna())
                st.write(f"- **Teste de Esfericidade de Bartlett:** p-valor = {p_value:.4f}")
                if p_value < 0.05:
                    st.success("✅ O teste é significativo (p < 0.05), indicando que as variáveis estão correlacionadas e adequadas para análise fatorial.")
                else:
                    st.warning("⚠️ O teste não é significativo (p ≥ 0.05), indicando que as variáveis podem não estar suficientemente correlacionadas para análise fatorial.")
                
                # KMO
                kmo_all, kmo_model = calculate_kmo(df_to_analyze[num_cols].dropna())
                st.write(f"- **Medida de Adequação de Kaiser-Meyer-Olkin (KMO):** {kmo_model:.4f}")
                if kmo_model >= 0.8:
                    st.success("✅ Excelente adequação para análise fatorial (KMO ≥ 0.8).")
                elif kmo_model >= 0.7:
                    st.info("💡 Adequação razoável para análise fatorial (0.7 ≤ KMO < 0.8).")
                elif kmo_model >= 0.6:
                    st.warning("⚠️ Adequação medíocre para análise fatorial (0.6 ≤ KMO < 0.7).")
                else:
                    st.error("❌ Adequação inaceitável para análise fatorial (KMO < 0.6).")
                
                # Verificar se a matriz de correlação é singular
                corr_matrix = df_to_analyze[num_cols].corr()
                try:
                    # Tentar inverter a matriz para verificar singularidade
                    np.linalg.inv(corr_matrix)
                except np.linalg.LinAlgError:
                    st.error("❌ A matriz de correlação é singular (não invertível). Isso pode ocorrer por:")
                    st.error("- Colunas perfeitamente correlacionadas (correlação = 1 ou -1)")
                    st.error("- Colunas com variância zero (constantes)")
                    st.error("- Mais colunas do que linhas")
                    st.error("Por favor, verifique seus dados e remova colunas problemáticas antes de prosseguir.")
                    return
                
                # Executar análise fatorial se os testes forem adequados
                if p_value < 0.05 and kmo_model >= 0.6:
                    # Determinar número de fatores
                    fa = FactorAnalyzer(rotation=None, impute="drop", n_factors=len(num_cols))
                    fa.fit(df_to_analyze[num_cols].dropna())
                    
                    # Plotar scree plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ev, v = fa.get_eigenvalues()
                    ax.scatter(range(1, df_to_analyze[num_cols].shape[1]+1), ev)
                    ax.plot(range(1, df_to_analyze[num_cols].shape[1]+1), ev)
                    ax.set_title('Scree Plot')
                    ax.set_xlabel('Fatores')
                    ax.set_ylabel('Autovalor')
                    ax.axhline(y=1, color='r', linestyle='--')
                    ax.grid()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.pyplot(fig)
                    with col2:
                        st.empty()  # Espaço vazio para balancear
                    
                    plt.close(fig)
                    
                    st.write("""
                    **Interpretação do Scree Plot:**
                    - Fatores acima da linha vermelha (autovalor > 1) são considerados significativos
                    - O ponto onde a curva 'dobra' (elbow) indica o número ideal de fatores
                    """)
                    
                    # Sugerir número de fatores
                    n_factors_suggested = sum(ev > 1)
                    st.info(f"💡 Sugestão: Utilizar {n_factors_suggested} fatores principais (autovalor > 1).")
                    
                    # Substituir o number_input por um slider
                    with col1:
                        n_factors = st.slider(
                            "Selecione o número de fatores a serem extraídos:",
                            min_value=1,
                            max_value=len(num_cols),
                            value=n_factors_suggested,
                            step=1,
                            key="n_factors_slider"
                        )
                    
                    # Mostrar visualização do slider com marcações
                    st.write(f"Fatores selecionados: {n_factors}")
                    
                    # Executar análise com número selecionado de fatores
                    if st.button("Executar Análise Fatorial"):
                        with st.spinner("Executando análise fatorial..."):
                            fa = FactorAnalyzer(rotation="varimax", n_factors=n_factors)
                            fa.fit(df_to_analyze[num_cols].dropna())
                            
                            # Obter cargas fatoriais
                            loadings = pd.DataFrame(fa.loadings_, 
                                                   index=num_cols,
                                                   columns=[f"Fator {i+1}" for i in range(n_factors)])
                            
                            # Exibir cargas fatoriais com cores
                            def color_loading(val):
                                color = 'red' if abs(val) > 0.5 else 'black'
                                return f'color: {color}'
                            
                            st.write("**Cargas Fatoriais (Rotação Varimax):**")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.dataframe(
                                loadings.style.applymap(color_loading)
                                .format("{:.2f}")
                                .background_gradient(cmap='coolwarm', axis=None, vmin=-1, vmax=1)
                            )
                            with col2:
                                st.empty()  # Espaço vazio para balancear
                            
                            st.write("""
                            **Interpretação das Cargas Fatoriais:**
                            - Valores absolutos > 0.5 (em vermelho) indicam forte associação com o fator
                            - Cada variável geralmente carrega mais fortemente em um único fator
                            """)
                            
                            # Obter todas as métricas de variância
                            variance, proportion_variance, cumulative_variance = fa.get_factor_variance()
                            
                            # Criar DataFrame com todas as informações
                            variance_df = pd.DataFrame({
                                'Fator': [f"Fator {i+1}" for i in range(n_factors)],
                                'Variância': variance[:n_factors],  # Variância individual
                                '% Variância': proportion_variance[:n_factors] * 100,  # Em porcentagem
                                '% Variância Acumulada': cumulative_variance[:n_factors] * 100  # Acumulada em porcentagem
                            })

                            # Exibir a tabela formatada
                            st.write("**Variância Explicada por Cada Fator:**")

                            col1, col2 = st.columns(2)  # Ajuste a proporção conforme necessário
                            with col1:
                                st.dataframe(
                                    variance_df.style
                                    .format({
                                        'Variância': '{:.2f}',
                                        '% Variância': '{:.2f}%',
                                        '% Variância Acumulada': '{:.2f}%'
                                    })
                                    .background_gradient(cmap='Blues', subset=['% Variância', '% Variância Acumulada'])
                                    .set_properties(**{'text-align': 'center'})
                                    .set_table_styles([{
                                        'selector': 'th',
                                        'props': [('text-align', 'center')]
                                    }])
                                )
                            with col2:
                                st.empty()  # Espaço vazio para balancear

                            # Adicionar explicação
                            st.caption("""
                            **Legenda:**
                            - **Variância**: Valor absoluto da variância explicada por cada fator
                            - **% Variância**: Porcentagem da variância total explicada por cada fator
                            - **% Variância Acumulada**: Porcentagem cumulativa da variância explicada
                            """)

                            # Mostrar resumo final
                            total_variance = cumulative_variance[n_factors-1] * 100  # Variância total explicada
                            st.success(f"**Total explicado pelos {n_factors} fatores selecionados:** {total_variance:.1f}% da variância total")
                            
                            # Transformar os dados nos fatores
                            factors = fa.transform(df_to_analyze[num_cols].dropna())
                            factor_cols = [f"Fator_{i+1}" for i in range(n_factors)]
                            factors_df = pd.DataFrame(factors, columns=factor_cols, index=df_to_analyze[num_cols].dropna().index)
                            
                            # Armazenar os fatores na sessão para uso posterior
                            st.session_state['factors_df'] = factors_df
                            st.session_state['use_factors'] = True
                            
                            st.success("✅ Fatores calculados e prontos para uso na modelagem!")
                            
                            # Mostrar prévia dos fatores
                            st.write("**Prévia dos Fatores Calculados:**")
                            with col1:
                                st.dataframe(factors_df.head())
                            
                            st.info("""
                            **Próximos Passos:**
                            1. Os fatores serão automaticamente usados no lugar das variáveis originais na modelagem
                            2. Você pode desativar o uso de fatores na seção de configuração de modelagem
                            """)
                else:
                    st.warning("""
                    ⚠️ Os testes indicam que os dados podem não ser adequados para análise fatorial.
                    Considere outras técnicas de redução de dimensionalidade ou trabalhe com as variáveis originais.
                    """)
                    
            except Exception as e:
                st.error(f"❌ Erro durante a análise fatorial: {str(e)}")
                st.error("Isso geralmente ocorre quando há problemas com a matriz de correlação.")
                st.error("Sugestões:")
                st.error("1. Verifique se há colunas constantes (com todos os valores iguais)")
                st.error("2. Verifique se há colunas perfeitamente correlacionadas")
                st.error("3. Remova colunas problemáticas e tente novamente")
                return

        else:
            st.info("São necessárias pelo menos 3 colunas numéricas para realizar a análise fatorial.")

    
            

# =============================================
# Funções de processamento de dados
# =============================================
def load_data(uploaded):
    df_original = None
    
    if uploaded.name.endswith('.csv'):
        st.markdown("#### Pré-visualização e Configurações do CSV")
        
        # Configurações de importação
        col1, col2 = st.columns(2)
        with col1:
            # Detecção automática do delimitador (usando latin1 como fallback)
            try:
                raw_text = uploaded.getvalue().decode('latin1')
                first_lines = raw_text.split('\n')[:5]
                
                probable_delimiters = {',': 0, ';': 0, '\t': 0, '|': 0}
                for line in first_lines[:5]:
                    for delim in probable_delimiters:
                        probable_delimiters[delim] += line.count(delim)
                
                sorted_delimiters = sorted(probable_delimiters.items(), key=lambda x: x[1], reverse=True)
                default_delim = sorted_delimiters[0][0] if sorted_delimiters[0][1] > 0 else ';'
                
                delimiter = st.selectbox(
                    "Delimitador de colunas:",
                    options=[',', ';', '\t', '|'],
                    format_func=lambda x: {
                        ',': 'Vírgula (,)',
                        ';': 'Ponto e vírgula (;)', 
                        '\t': 'Tabulação (\\t)',
                        '|': 'Pipe (|)'
                    }[x],
                    index=[',', ';', '\t', '|'].index(default_delim)
                )
            except:
                delimiter = ';'  # Fallback seguro
        
        with col2:
            encoding = st.selectbox(
                "Codificação do arquivo:",
                options=['latin1', 'iso-8859-1', 'cp1252', 'utf-8'],  # Latin1 primeiro
                index=0
            )
        
        # Pré-visualização do conteúdo bruto com a codificação selecionada
        try:
            uploaded.seek(0)
            raw_text = uploaded.getvalue().decode(encoding)
            first_lines = raw_text.split('\n')[:5]
            
            with st.expander("🔍 Visualizar conteúdo bruto do arquivo (primeiras 5 linhas)"):
                st.code("\n".join(first_lines))
        except Exception as e:
            st.error(f"❌ Falha ao ler conteúdo bruto: {str(e)}")
            return None
        
        # Opção para tratar cabeçalho
        has_header = st.checkbox("O arquivo possui linha de cabeçalho", value=True)
        
        # Pré-visualização com configurações atuais
        st.markdown("#### Prévia com configurações atuais")
        try:
            uploaded.seek(0)
            df_preview = pd.read_csv(
                uploaded,
                delimiter=delimiter,
                decimal=',',
                thousands='.',
                encoding=encoding,  # Usando a codificação selecionada
                header=0 if has_header else None,
                nrows=5,
                engine='python'  # Engine mais tolerante
            )
            
            # Ajuste especial para cabeçalhos deslocados
            if has_header and any(col.startswith('Unnamed:') for col in df_preview.columns):
                st.warning("⚠️ Possível desalinhamento de colunas detectado!")
                df_preview = pd.read_csv(
                    uploaded,
                    delimiter=delimiter,
                    decimal=',',
                    thousands='.',
                    encoding=encoding,
                    header=None,
                    nrows=5,
                    engine='python'
                )
                st.dataframe(df_preview)
                st.warning("Sugestão: Desmarque 'possui linha de cabeçalho' se a primeira linha não for cabeçalho real")
            else:
                st.dataframe(df_preview)
            
            if df_preview.shape[1] == 1:
                st.warning("⚠️ Apenas 1 coluna detectada. Verifique o delimitador!")
            if any(str(col).startswith('Unnamed') for col in df_preview.columns):
                st.warning("⚠️ Colunas sem nome detectadas. Verifique cabeçalhos.")
                
        except Exception as e:
            st.error(f"❌ Falha na pré-visualização: {str(e)}")
            return None
        
        # Botão para carregar definitivamente
        if st.button("📤 Carregar arquivo CSV com estas configurações"):
            try:
                uploaded.seek(0)
                df_original = pd.read_csv(
                    uploaded,
                    delimiter=delimiter,
                    decimal=',',
                    thousands='.',
                    encoding=encoding,
                    header=0 if has_header else None,
                    engine='python'
                )
                
                # Correção pós-carregamento para colunas Unnamed
                df_original.columns = [f"col_{i}" if str(col).startswith('Unnamed') else col 
                                     for i, col in enumerate(df_original.columns)]
                
                # Armazena no estado da sessão
                st.session_state['df_original'] = df_original
                st.session_state['df_loaded'] = True
                st.session_state['file_config'] = {
                    'delimiter': delimiter,
                    'encoding': encoding,
                    'has_header': has_header
                }
                
                st.success("✅ Arquivo carregado com sucesso!")
                return df_original
                
            except Exception as e:
                st.error(f"❌ Erro ao carregar: {str(e)}")
                return None
            
    elif uploaded.name.endswith('.xlsx'):
        try:
            xls = pd.ExcelFile(uploaded)
            st.markdown("#### Abas encontradas no arquivo:")
            for name in xls.sheet_names:
                st.write(f"- {name}")
            
            selected_sheet_name = st.selectbox(
                "Selecione a aba do Excel para carregar", 
                xls.sheet_names, 
                key="excel_sheet_selector"
            )
            
            # PRÉ-VISUALIZAÇÃO DA ABA SELECIONADA
            st.markdown("#### Pré-visualização da aba selecionada")
            
            # Lê apenas as primeiras 5 linhas para preview
            df_preview = pd.read_excel(xls, sheet_name=selected_sheet_name, nrows=5)
            st.dataframe(df_preview)
            
            # Verifica problemas comuns
            if df_preview.empty:
                st.warning("⚠️ A aba selecionada está vazia!")
            if df_preview.columns.duplicated().any():
                st.warning("⚠️ Existem colunas duplicadas no cabeçalho!")
            
            # Botão para carregar definitivamente
            if st.button("📤 Carregar aba selecionada"):
                df_original = pd.read_excel(xls, sheet_name=selected_sheet_name)
                
                # Armazena no estado da sessão
                st.session_state['df_original'] = df_original
                st.session_state['df_loaded'] = True
                st.session_state['file_config'] = {
                    'sheet_name': selected_sheet_name
                }
                
                st.success(f"✅ Aba '{selected_sheet_name}' carregada com sucesso!")
                return df_original
        
        except Exception as e:
            st.error(f"❌ Erro ao ler arquivo Excel: {str(e)}")
            return None

def prepare_data(df_original, x_cols, y_col):
    cols_to_check = x_cols + [y_col]
    initial_nans_present = df_original[cols_to_check].isnull().sum().sum() > 0
    
    if initial_nans_present:
        linhas_antes = df_original.shape[0]
        df_limpo = df_original.dropna(subset=cols_to_check)
        linhas_depois = df_limpo.shape[0]
        linhas_removidas = linhas_antes - linhas_depois
        if linhas_removidas > 0:
            st.info(f"✨ **{linhas_removidas} linhas com valores ausentes foram automaticamente removidas** das colunas selecionadas. Nova dimensão: **{linhas_depois} linhas**.")
        return df_limpo, False
    else:
        return df_original.copy(), False

# =============================================
# Funções de modelagem
# =============================================
def train_regression_models(modelos_disponiveis, modelos_selecionados, parametros_grid, X, y, optimization_method, n_iter_random, cv_folds):
    status_placeholder = st.empty()
    progress_bar2 = st.progress(0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    _, p_value = shapiro(y_train)
    if p_value < 0.05:
        st.warning(f"⚠️ Os dados da variável alvo não parecem ter distribuição normal (p-value = {p_value:.4f}). Considere aplicar transformações como logarítmica.")

    resultados = []
    
    total_models2 = len(modelos_selecionados)
    for idx, nome_modelo in enumerate(modelos_selecionados):
        progress2 = (idx + 1) / total_models2
        progress_bar2.progress(progress2)
    
        status_placeholder.info(f"⚙️ Treinando e avaliando: **{nome_modelo}** ({idx+1}/{total_models2}) com **{optimization_method}**...")
        
        modelo_base = modelos_disponiveis['regression'][nome_modelo]
        num_cols = X.select_dtypes(include=np.number).columns.tolist()
        cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
        
        pipeline = build_pipeline(modelo_base, num_cols, cat_cols)
        
        if optimization_method == "GridSearchCV":
            grid_search = GridSearchCV(
                pipeline,
                param_grid=parametros_grid.get(nome_modelo, {}),
                scoring='neg_mean_squared_error',
                cv=cv_folds,
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            modelo_final = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
        elif optimization_method == "RandomizedSearchCV":
            random_search = RandomizedSearchCV(
                pipeline,
                param_distributions=parametros_grid.get(nome_modelo, {}),
                n_iter=n_iter_random,
                scoring='neg_mean_squared_error',
                cv=cv_folds,
                random_state=42,
                n_jobs=-1
            )
            random_search.fit(X_train, y_train)
            modelo_final = random_search.best_estimator_
            best_params = random_search.best_params_
        
        y_pred = modelo_final.predict(X_test)
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('Valores Reais')
        ax.set_ylabel('Valores Preditos')
        ax.set_title(f'{nome_modelo} - Valores Reais vs Preditos')
        plt.close(fig)
        
        resultados.append({
            'Modelo': nome_modelo,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'Melhores Params': best_params,
            'Gráfico': fig
        })
        
        status_placeholder.success(f"✅ {nome_modelo} treinado com sucesso! R² = {r2:.4f}")
    
    progress_bar2.empty()
    status_placeholder.empty()
    st.success("🎉 Todos os modelos foram treinados e avaliados!")
    
    return resultados, X_test, y_test
    
def train_classification_models(modelos_disponiveis, modelos_selecionados, parametros_grid, X, y, optimization_method, 
                                n_iter_random, cv_folds, interest_classes_selected, threshold_metric):

    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    # Inicializar ou reutilizar o label encoder
    if st.session_state['label_encoder'] is None:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        st.session_state['label_encoder'] = le
        st.session_state['y_numeric_mapping'] = dict(zip(le.classes_, le.transform(le.classes_)))
        st.session_state['y_original_labels'] = le.classes_.tolist()
    else:
        le = st.session_state['label_encoder']
        try:
            y_encoded = le.transform(y)
        except ValueError:
            st.warning("Novas classes encontradas na variável alvo. Recriando o label encoder.")
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            st.session_state['label_encoder'] = le
            st.session_state['y_numeric_mapping'] = dict(zip(le.classes_, le.transform(le.classes_)))
            st.session_state['y_original_labels'] = le.classes_.tolist()
    
    y_names = le.classes_
    num_classes = len(np.unique(y_encoded))
    is_binary_problem = num_classes == 2
    
    # Divisão treino-teste
    class_counts = pd.Series(y_encoded).value_counts()
    stratify_param = y_encoded if all(class_counts >= 2) else None
    if stratify_param is None:
        st.warning("Uma ou mais classes têm menos de 2 amostras. A estratificação na divisão treino/teste será desativada.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=stratify_param, random_state=42
    )
    
    # Verificação de balanceamento
    balance_info = {}
    if num_classes > 1:
        total_samples_train = len(y_train)
        y_train_counts = pd.Series(y_train).value_counts().sort_index()
        class_proportions = y_train_counts / total_samples_train
        minority_class_prop = class_proportions.min()
        
        is_imbalanced = minority_class_prop < 0.30
        
        if is_imbalanced:
            st.warning(f"⚠️ **Desbalanceamento detectado!** A proporção da classe minoritária no conjunto de treino é de {minority_class_prop:.2%}.")
            balance_info['is_imbalanced'] = True
            balance_info['minority_class_prop'] = minority_class_prop
        else:
            st.info("Dataset balanceado (proporção da classe minoritária >= 40%). Nenhum tratamento de balanceamento automático será aplicado.")
            balance_info['is_imbalanced'] = False
    else:
        balance_info['is_imbalanced'] = False
        st.info("A variável alvo possui apenas uma classe. O balanceamento de classes não é aplicável.")

    class_distribution_str = "Distribuição de classes no CONJUNTO DE TREINO:\n\n"
    for class_label, count in pd.Series(y_train).value_counts().items():
        class_name = le.inverse_transform([class_label])[0]
        class_distribution_str += f"Classe '{class_name}' (código {class_label}): {count} itens\n\n"

    st.warning(class_distribution_str)
    
    # Treinamento dos modelos
    resultados = []
    roc_curves_data = []
    
    # Processar classes de interesse
    numeric_interest_classes = [st.session_state['y_numeric_mapping'][label] 
                               for label in interest_classes_selected 
                               if label in st.session_state['y_numeric_mapping']]
    
    total_models = len(modelos_selecionados)
    for idx, nome_modelo in enumerate(modelos_selecionados):
        progress = (idx + 1) / total_models
        progress_bar.progress(progress)
        
        status_placeholder.info(f"⚙️ Treinando e avaliando: **{nome_modelo}** ({idx+1}/{total_models}) com **{optimization_method}**...")
        resultado_modelo = {'Modelo': nome_modelo, 'AUC': np.nan, 'PR-AUC': np.nan}  # Inicializa com NaN
        
        modelo_base = modelos_disponiveis['classification'][nome_modelo]
        num_cols_model = X.select_dtypes(include=np.number).columns.tolist()
        cat_cols_model = X.select_dtypes(exclude=np.number).columns.tolist()

        current_params_grid = parametros_grid.get(nome_modelo, {})
        sampler_for_pipeline = None

        # Tratamento de desbalanceamento
        if balance_info['is_imbalanced']:
            if nome_modelo in ["SGDClassifier", "LogisticRegression", "DecisionTreeClassifier", "RandomForestClassifier", "SVC"]:
                if 'clf__class_weight' in current_params_grid:
                    if 'balanced' not in current_params_grid['clf__class_weight']:
                        current_params_grid['clf__class_weight'].append('balanced')
                else:
                    current_params_grid['clf__class_weight'] = ['balanced']
            
            elif nome_modelo == "XGBClassifier":
                if is_binary_problem:
                    pos_class_count = sum(y_train == 1)
                    neg_class_count = sum(y_train == 0)
                    if pos_class_count > 0 and neg_class_count > 0:
                        scale_pos_weight_value = neg_class_count / pos_class_count
                        current_params_grid['clf__scale_pos_weight'] = [scale_pos_weight_value]
                        st.info(f"Balanceamento de classes: **scale_pos_weight={scale_pos_weight_value:.2f}** adicionado para {nome_modelo} (binário).")
                    else:
                        st.warning(f"Não foi possível aplicar scale_pos_weight para {nome_modelo}: Classes positivas/negativas inválidas.")
                else:
                    st.warning(f"XGBClassifier: scale_pos_weight só é aplicável para problemas de classificação binária. Considere técnicas de reamostragem.")
            
            elif nome_modelo == "LGBMClassifier":
                if is_binary_problem:
                    pos_class_count = sum(y_train == 1)
                    neg_class_count = sum(y_train == 0)
                    if pos_class_count > 0 and neg_class_count > 0:
                        # REMOVA is_unbalance OU scale_pos_weight - use apenas um deles
                        # Opção 1: Usar apenas scale_pos_weight
                        current_params_grid['clf__scale_pos_weight'] = [neg_class_count / pos_class_count]
                        st.info(f"Balanceamento de classes: scale_pos_weight={neg_class_count / pos_class_count:.2f} para {nome_modelo}")
                        
                        # Ou Opção 2: Usar apenas is_unbalance
                        # current_params_grid['clf__is_unbalance'] = [True]
                        # st.info(f"Balanceamento de classes: is_unbalance=True para {nome_modelo}")
                    else:
                        st.warning(f"Não foi possível aplicar balanceamento para {nome_modelo}: Classes positivas/negativas inválidas.")
                else:
                    # Para problemas multiclasse
                    current_params_grid['clf__class_weight'] = ['balanced']
                    st.info(f"Balanceamento de classes: class_weight='balanced' para {nome_modelo} (multiclasse).")

            elif nome_modelo in ["GradientBoostingClassifier", "KNeighborsClassifier", "GaussianNB"]:
                st.info(f"Balanceamento de classes: **RandomUnderSampler** será aplicado para {nome_modelo}.")
                sampler_for_pipeline = RandomUnderSampler(random_state=42)

        # Construir pipeline
        pipeline = build_pipeline(modelo_base, num_cols_model, cat_cols_model, sampler=sampler_for_pipeline)

        # Otimização de hiperparâmetros
        if optimization_method == "GridSearchCV":
            with st.spinner(f"Executando GridSearchCV para {nome_modelo}..."):
                grid_search = GridSearchCV(pipeline, param_grid=current_params_grid, scoring='accuracy', 
                                          cv=StratifiedKFold(n_splits=cv_folds), n_jobs=-1)
                grid_search.fit(X_train, y_train)
                modelo_final = grid_search.best_estimator_
                best_params = grid_search.best_params_
        
        elif optimization_method == "RandomizedSearchCV":
            with st.spinner(f"Executando RandomizedSearchCV para {nome_modelo} com {n_iter_random} iterações..."):
                random_search = RandomizedSearchCV(pipeline, param_distributions=current_params_grid, n_iter=n_iter_random, 
                                                  scoring='accuracy', cv=StratifiedKFold(n_splits=cv_folds), 
                                                  random_state=42, n_jobs=-1)
                random_search.fit(X_train, y_train)
                modelo_final = random_search.best_estimator_
                best_params = random_search.best_params_
        
        # Determinar limiar ótimo
        optimal_threshold = 0.5
        if hasattr(modelo_final, 'predict_proba'):
            probs = modelo_final.predict_proba(X_test)
            
            if is_binary_problem or numeric_interest_classes:
                if is_binary_problem:
                    positive_class_idx = 1 
                    y_true_binary = y_test
                    probabilities_positive_class = probs[:, positive_class_idx]
                else:
                    y_true_binary = np.isin(y_test, numeric_interest_classes).astype(int)
                    probabilities_positive_class = np.sum(probs[:, numeric_interest_classes], axis=1)

                precisions, recalls, thresholds = precision_recall_curve(y_true_binary, probabilities_positive_class)
                f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10) 
                
                accuracies = []
                for t in thresholds:
                    y_pred_thresholded = (probabilities_positive_class >= t).astype(int)
                    accuracies.append(accuracy_score(y_true_binary, y_pred_thresholded))
                accuracies = np.array(accuracies)

                if threshold_metric == "F1-Score":
                    idx_optimal_threshold = np.argmax(f1_scores)
                elif threshold_metric == "Acurácia":
                    idx_optimal_threshold = np.argmax(accuracies)
                elif threshold_metric == "Precisão":
                    idx_optimal_threshold = np.argmax(precisions)
                elif threshold_metric == "Recall":
                    idx_optimal_threshold = np.argmax(recalls)
                
                optimal_threshold = thresholds[idx_optimal_threshold]
                st.info(f"✨ Limiar ótimo calculado para {nome_modelo} baseado em **{threshold_metric}**: **{optimal_threshold:.4f}**")
        
        # Fazer previsões
        if hasattr(modelo_final, 'predict_proba'):
            if is_binary_problem: 
                final_preds = (modelo_final.predict_proba(X_test)[:, 1] >= optimal_threshold).astype(int)
            elif numeric_interest_classes: 
                positive_prob_sum_for_pred = np.sum(modelo_final.predict_proba(X_test)[:, numeric_interest_classes], axis=1)
                final_preds = (positive_prob_sum_for_pred >= optimal_threshold).astype(int)
            else: 
                final_preds = modelo_final.predict(X_test)
        else: 
            final_preds = modelo_final.predict(X_test)

        # Avaliar modelo
        acc = accuracy_score(y_test, final_preds)
        if numeric_interest_classes and not is_binary_problem:
            y_test_eval = np.isin(y_test, numeric_interest_classes).astype(int)
        else:
            y_test_eval = y_test
        
        prec = precision_score(y_test_eval, final_preds, average='weighted', zero_division=0)
        rec = recall_score(y_test_eval, final_preds, average='weighted')
        f1 = f1_score(y_test_eval, final_preds, average='weighted')
        
        labels_presentes_numericas = unique_labels(y_test_eval, final_preds)
        
        if numeric_interest_classes and not is_binary_problem:
            complementary_label_for_report = "Outras Classes"
            interest_label_for_report = ", ".join(interest_classes_selected)
            target_names_str_for_report = [complementary_label_for_report, interest_label_for_report]
            labels_presentes_numericas_for_report = [0, 1] 
        else:
            target_names_str_for_report = [le.inverse_transform([i])[0] for i in labels_presentes_numericas]
            labels_presentes_numericas_for_report = labels_presentes_numericas

        # Matriz de confusão
        cm = confusion_matrix(y_test_eval, final_preds, labels=labels_presentes_numericas_for_report)
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm,
                    xticklabels=target_names_str_for_report, yticklabels=target_names_str_for_report)
        ax_cm.set_xlabel('Predito')
        ax_cm.set_ylabel('Verdadeiro')
        ax_cm.set_title(f'Matriz de Confusão - {nome_modelo}')
        plt.close(fig_cm)
        
        buf_cm = BytesIO()
        fig_cm.savefig(buf_cm, format="png")
        buf_cm.seek(0)
        resultado_modelo['Confusion Matrix'] = base64.b64encode(buf_cm.read()).decode()
        
        # Inicializa roc_auc como NaN
        roc_auc = np.nan
        
        # Curva ROC
        if hasattr(modelo_final, 'predict_proba'):
            if is_binary_problem and not interest_classes_selected: 
                probs_roc = modelo_final.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, probs_roc)
                roc_auc = auc(fpr, tpr)
                
                fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
                ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
                ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax_roc.set_xlim([0.0, 1.0])
                ax_roc.set_ylim([0.0, 1.05])
                ax_roc.set_xlabel('Taxa de Falsos Positivos')
                ax_roc.set_ylabel('Taxa de Verdadeiros Positivos')
                ax_roc.set_title(f'Curva ROC - {nome_modelo}')
                ax_roc.legend(loc='lower right')
                plt.close(fig_roc)
                
                buf_roc = BytesIO()
                fig_roc.savefig(buf_roc, format="png")
                buf_roc.seek(0)
                resultado_modelo['ROC Curve'] = base64.b64encode(buf_roc.read()).decode()
                roc_curves_data.append({'fpr': fpr, 'tpr': tpr, 'auc': roc_auc, 'model': nome_modelo, 'plot_type': 'binary'})
            
            elif interest_classes_selected: 
                y_test_binarized_roc = np.isin(y_test, numeric_interest_classes).astype(int) 
                positive_prob_sum_roc = np.sum(modelo_final.predict_proba(X_test)[:, numeric_interest_classes], axis=1) 

                fpr, tpr, _ = roc_curve(y_test_binarized_roc, positive_prob_sum_roc)
                roc_auc = auc(fpr, tpr)
                
                fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
                ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
                ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax_roc.set_xlim([0.0, 1.0])
                ax_roc.set_ylim([0.0, 1.05])
                ax_roc.set_xlabel('Taxa de Falsos Positivos')
                ax_roc.set_ylabel('Taxa de Verdadeiros Positivos')
                ax_roc.set_title(f'Curva ROC - {nome_modelo}')
                ax_roc.legend(loc='lower right')
                plt.close(fig_roc)
                
                buf_roc = BytesIO()
                fig_roc.savefig(buf_roc, format="png")
                buf_roc.seek(0)
                resultado_modelo['ROC Curve'] = base64.b64encode(buf_roc.read()).decode()
                roc_curves_data.append({'fpr': fpr, 'tpr': tpr, 'auc': roc_auc, 'model': nome_modelo, 'plot_type': 'custom_binary',
                                        'interest_classes': interest_classes_selected, 
                                        'complementary_classes': [label for label in st.session_state['y_original_labels'] 
                                                                 if label not in interest_classes_selected]})

        # Atualiza o AUC no resultado
        resultado_modelo['AUC'] = roc_auc
        
        # Curva Precision-Recall
        pr_auc = np.nan
        if hasattr(modelo_final, 'predict_proba') and (is_binary_problem or interest_classes_selected):
            if is_binary_problem and not interest_classes_selected:
                probs_pr = modelo_final.predict_proba(X_test)[:, 1]
                precision, recall, _ = precision_recall_curve(y_test, probs_pr) 
                pr_auc = auc(recall, precision)
                positive_class_prop = np.mean(y_test)
            elif interest_classes_selected:
                y_test_binarized_pr = np.isin(y_test, numeric_interest_classes).astype(int)
                positive_prob_sum_pr = np.sum(modelo_final.predict_proba(X_test)[:, numeric_interest_classes], axis=1)
                precision, recall, _ = precision_recall_curve(y_test_binarized_pr, positive_prob_sum_pr)
                pr_auc = auc(recall, precision)
                positive_class_prop = np.mean(y_test_binarized_pr)
            
            fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
            ax_pr.plot(recall, precision, color='blue', lw=2, label=f'AUC = {pr_auc:.2f}')
            ax_pr.axhline(y=positive_class_prop, color='r', linestyle='--', 
                         label=f'Baseline ({positive_class_prop:.2f})')
            ax_pr.set_xlim([0.0, 1.0])
            ax_pr.set_ylim([0.0, 1.05])
            ax_pr.set_xlabel('Recall')
            ax_pr.set_ylabel('Precision')
            ax_pr.set_title(f'Curva Precision-Recall - {nome_modelo}')
            ax_pr.legend(loc='lower left')
            plt.close(fig_pr)
            
            buf_pr = BytesIO()
            fig_pr.savefig(buf_pr, format="png")
            buf_pr.seek(0)
            resultado_modelo['PR Curve'] = base64.b64encode(buf_pr.read()).decode()
            resultado_modelo['PR-AUC'] = pr_auc

        # Atualiza as métricas no resultado
        resultado_modelo.update({
            'Acurácia': acc,
            'Precisão': prec,
            'Recall': rec,
            'F1-Score': f1,
            'Optimal Threshold': optimal_threshold if (is_binary_problem or numeric_interest_classes) else np.nan,
            'Melhores Params': best_params,
        })

        resultados.append(resultado_modelo)
        status_placeholder.success(f"✅ Modelo **{nome_modelo}** avaliado com sucesso!")
    
    progress_bar.empty()           
    status_placeholder.empty()
    st.success("🎉 Todos os modelos foram treinados e avaliados!")
    
    # Gerar relatório HTML
    html_relatorio = generate_classification_html_report(resultados, roc_curves_data, le, interest_classes_selected, X_test, y_test)
    
    return resultados, roc_curves_data, html_relatorio, X_test, y_test

# =============================================
# Configuração de estado da sessão
# =============================================
def initialize_session_state():
    if 'df_original_loaded' not in st.session_state:
        st.session_state['df_original_loaded'] = False
    if 'data_preprocessed' not in st.session_state:
        st.session_state['data_preprocessed'] = False
    if 'nans_present_in_selection' not in st.session_state: 
        st.session_state['nans_present_in_selection'] = False
    if 'y_original_labels' not in st.session_state:
        st.session_state['y_original_labels'] = []
    if 'y_numeric_mapping' not in st.session_state:
        st.session_state['y_numeric_mapping'] = {}
    if 'interest_classes_selected' not in st.session_state: 
        st.session_state['interest_classes_selected'] = []
    if 'x_cols' not in st.session_state:
        st.session_state['x_cols'] = []
    if 'y_col' not in st.session_state:
        st.session_state['y_col'] = None
    if 'df_trabalho' not in st.session_state: 
        st.session_state['df_trabalho'] = pd.DataFrame()
    if 'uploaded_file_name' not in st.session_state: 
        st.session_state['uploaded_file_name'] = None
    if 'excel_sheet_name' not in st.session_state: 
        st.session_state['excel_sheet_name'] = None
    if 'excel_header_row' not in st.session_state: 
        st.session_state['excel_header_row'] = 0
    if 'n_iter_random' not in st.session_state: 
        st.session_state['n_iter_random'] = 20
    if 'optimization_method' not in st.session_state:
        st.session_state['optimization_method'] = "GridSearchCV"
    if 'threshold_metric' not in st.session_state: 
        st.session_state['threshold_metric'] = "Acurácia"
    if 'cv_folds' not in st.session_state:
        st.session_state['cv_folds'] = 5
    if 'resultados_regressao' not in st.session_state:
        st.session_state['resultados_regressao'] = []
    if 'resultados_classificacao' not in st.session_state:
        st.session_state['resultados_classificacao'] = None
    if 'roc_curves_data' not in st.session_state:
        st.session_state['roc_curves_data'] = None
    if 'html_relatorio_classificacao' not in st.session_state:
        st.session_state['html_relatorio_classificacao'] = None
    if 'html_relatorio_regressao' not in st.session_state:
        st.session_state['html_relatorio_regressao'] = None
    if 'dados_treinamento' not in st.session_state:
        st.session_state['dados_treinamento'] = None
    if 'X_test_reg' not in st.session_state:
        st.session_state['X_test_reg'] = None
    if 'y_test_reg' not in st.session_state:
        st.session_state['y_test_reg'] = None
    if 'X_test_cla' not in st.session_state:
        st.session_state['X_test_cla'] = None
    if 'y_test_cla' not in st.session_state:
        st.session_state['y_test_cla'] = None
    if 'preprocessor' not in st.session_state:
        st.session_state['preprocessor'] = None    
    if 'label_encoder' not in st.session_state:
        st.session_state['label_encoder'] = None
    if 'modelos_configurados' not in st.session_state:
        st.session_state['modelos_configurados'] = False

# =============================================
# Função principal
# =============================================
def main():
    setup_page()
    initialize_session_state()
    
    # ----- 1. Carregamento de Dados -----
    st.markdown("## 1. Carregamento de Dados")
    uploaded = st.file_uploader("Carregue seu dataset (CSV ou Excel)", type=['csv', 'xlsx'])
    
    # [Seu código existente de carregamento de dados...]
    if uploaded:
        if uploaded.name != st.session_state['uploaded_file_name']:
            st.session_state['uploaded_file_name'] = uploaded.name
            st.session_state['df_original_loaded'] = False
            st.session_state['data_preprocessed'] = False
            st.session_state['nans_present_in_selection'] = False
            st.session_state['y_original_labels'] = []
            st.session_state['y_numeric_mapping'] = {}
            st.session_state['interest_classes_selected'] = [] 
            st.session_state['x_cols'] = []
            st.session_state['y_col'] = None
            st.session_state['df_trabalho'] = pd.DataFrame() 
            st.session_state['excel_sheet_name'] = None 
            st.session_state['excel_header_row'] = 0
            st.session_state['label_encoder'] = None
            st.rerun()

        df_original = load_data(uploaded)
        if df_original is not None:
            st.session_state['df_original'] = df_original.copy()
            st.session_state['df_trabalho'] = df_original.copy()
            st.session_state['df_original_loaded'] = True
            st.dataframe(df_original.head())
    
    if not st.session_state['df_original_loaded']:
        st.warning("⬆️ **Carregue seu dataset acima para começar a análise!**")
        st.stop()
    
    # ----- 2. Seleção da Variável Alvo -----
    st.markdown("---")
    st.markdown("## 2. Seleção da Variável Alvo")
    
    all_cols = st.session_state['df_original'].columns.tolist()
    
    # Modificação crucial: Removemos a pré-seleção automática
    y_col_current = st.selectbox(
        "Selecione a variável alvo (y)", 
        [""] + all_cols,  # Adicionamos opção vazia no início
        index=0,  # Sempre começa vazio
        key="y_col_selector"
    )
    
    # Só continua se uma variável for selecionada (não vazia)
    if y_col_current:
        if y_col_current != st.session_state.get('y_col'):
            st.session_state.update({
                'y_col': y_col_current,
                'data_preprocessed': False,
                'label_encoder': None,
                'interest_classes_selected': []
            })
            st.rerun()
        
        # Detecção do tipo de problema
        y_data = st.session_state['df_original'][st.session_state['y_col']]
        problem_type = detect_problem_type(y_data)
        st.session_state['problem_type'] = problem_type
        st.info(f"🔍 Tipo de problema detectado: {'Classificação' if problem_type == 'classification' else 'Regressão'}")
    
        # ----- 3. Análise Exploratória (SÓ APÓS CONFIRMAÇÃO) -----
        if st.button("Confirmar variável alvo e gerar análise exploratória automática"):
            st.session_state['y_col_confirmed'] = True
            st.success("Variável alvo confirmada! Gerando análises...")
            st.rerun()
        
        if st.session_state.get('y_col_confirmed'):
            st.markdown("---")
            st.markdown("## 3. Análise Exploratória Automática")
            
            cols_for_analysis = [col for col in st.session_state['df_original'].columns 
                               if col != st.session_state['y_col']]
            
            if cols_for_analysis:
                with st.spinner("Gerando análises exploratórias..."):
                    perform_eda(st.session_state['df_original'][cols_for_analysis])
            else:
                st.info("Não há colunas disponíveis para análise além da variável alvo.")
    
    # ----- 4. Seleção de Variáveis Preditoras -----
    if st.session_state.get('y_col_confirmed'):
        st.markdown("---")
        st.markdown("## 4. Seleção de Variáveis Preditoras")

        # Obter colunas disponíveis (excluindo a variável alvo)
        available_cols = [col for col in st.session_state['df_original'].columns 
                         if col != st.session_state['y_col']]

        # Obter colunas numéricas (excluindo a variável alvo) - CORREÇÃO AQUI
        num_cols = [col for col in st.session_state['df_original'].select_dtypes(include=np.number).columns 
                   if col != st.session_state['y_col']]

        # Inicializar vif_info como dicionário vazio
        vif_info = {}

        if len(num_cols) >= 2:
            # Calcular VIF para cada variável numérica
            vif_data = pd.DataFrame()
            vif_data["Variável"] = num_cols
            vif_data["VIF"] = [variance_inflation_factor(st.session_state['df_original'][num_cols].dropna().values, i) 
                             for i in range(len(num_cols))]
            
            # Criar dicionário de VIF para uso posterior
            vif_info = dict(zip(vif_data["Variável"], vif_data["VIF"]))

        # Função para adicionar emojis com base no VIF (se aplicável)
        def label_with_vif(col):
            if len(num_cols) >= 2 and col in vif_info:
                vif = vif_info[col]
                if vif > 10:
                    return f"🔴 {col}"  # Alta multicolinearidade
                elif vif > 5:
                    return f"🟠 {col}"  # Moderada
            return f"⚫ {col}"  # Baixa ou não numérica

        # Criar mapeamento entre labels com emoji e nomes reais
        col_labels = {label_with_vif(col): col for col in available_cols}
        reverse_labels = {v: k for k, v in col_labels.items()}

        # Valores default convertidos para labels com emojis
        default_x_cols = st.session_state.get('x_cols', [])
        default_x_cols = [col for col in default_x_cols if col in available_cols]
        default_x_labels = [reverse_labels.get(col, col) for col in default_x_cols]

        # Seleção com emojis
        x_cols_labeled = st.multiselect(
            "Selecione as colunas preditoras (X)", 
            options=list(col_labels.keys()),
            default=default_x_labels,
            key="x_cols_selector"
        )

        # Traduzir de volta para nomes reais
        x_cols_current = [col_labels[label] for label in x_cols_labeled]
        
        # Identificar e armazenar colunas categóricas selecionadas
        selected_cat_cols = [col for col in x_cols_current 
                            if col in st.session_state['df_original'].select_dtypes(include=['object', 'category']).columns]
        st.session_state['selected_cat_cols'] = selected_cat_cols
        

        
        # Mostrar legenda dos emojis
        if len(num_cols) >= 2:
            st.markdown("""
            **Legenda dos Emojis de Multicolinearidade:**
            - 🔴 VIF > 10 → Alta multicolinearidade
            - 🟠 5 < VIF ≤ 10 → Moderada
            - ⚫ VIF ≤ 5 ou não numérica → Baixa ou irrelevante
            """)
        
        if selected_cat_cols:
            st.success(f"✅ Colunas categóricas selecionadas: {', '.join(selected_cat_cols)}")
        else:
            st.info("ℹ️ Nenhuma coluna categórica foi selecionada como preditora")

        # Se fatores tiverem sido calculados
        if st.session_state.get('factors_df') is not None:
            st.markdown("### Opções de Análise Fatorial")
            use_factors = st.checkbox(
                "Usar componentes principais (fatores) no lugar das variáveis numéricas originais",
                value=st.session_state.get('use_factors', False),
                key="use_factors_checkbox"
            )
            st.session_state['use_factors'] = use_factors
            
            if use_factors:
                st.info("ℹ️ Os fatores calculados na análise fatorial serão usados no lugar das variáveis numéricas originais.")
                st.write("Variáveis de fator disponíveis:", list(st.session_state['factors_df'].columns))
            else:
                st.info("ℹ️ As variáveis originais serão usadas para modelagem.")

        # Atualizar sessão se houve mudança nas variáveis preditoras
        if x_cols_current != st.session_state['x_cols']:
            st.session_state['x_cols'] = x_cols_current
            st.session_state['data_preprocessed'] = False
            st.rerun()
        
        
        # Processar dados
        if st.session_state['x_cols'] and st.session_state['y_col']:
            df_temp = st.session_state['df_original'].copy() 
            df_limpo, nans_present = prepare_data(df_temp, st.session_state['x_cols'], st.session_state['y_col'])
            st.session_state['df_trabalho'] = df_limpo
            st.session_state['nans_present_in_selection'] = nans_present
            st.session_state['data_preprocessed'] = True

        st.info(f"🔍 Dimensão atual do dataframe de trabalho: **{st.session_state['df_trabalho'].shape[0]} linhas**, **{st.session_state['df_trabalho'].shape[1]} colunas**")

    # ----- 5. Configuração de Classes (apenas classificação) -----
    if st.session_state.get('data_preprocessed'):
        if st.session_state.get('problem_type') == 'classification':
            st.markdown("---")
            st.markdown("## 5. Configuração de Classes para Curva ROC/AUC (Opcional)")
            st.info("Para problemas de classificação binária ou multiclasse com binarização, selecione as classes que você considera a **'classe de interesse'** para o cálculo da Curva ROC e AUC.")

            if st.session_state['y_col'] and st.session_state['y_col'] in st.session_state['df_trabalho'].columns:
                y_value_counts = st.session_state['df_trabalho'][st.session_state['y_col']].value_counts().sort_index()
                y_unique_values_with_counts = [f"{label} - {count}" for label, count in y_value_counts.items()]

                st.session_state['y_original_labels'] = y_value_counts.index.tolist()

                if len(y_value_counts) > 0:
                    le = LabelEncoder()
                    le.fit(st.session_state['df_trabalho'][st.session_state['y_col']])
                    st.session_state['y_numeric_mapping'] = dict(zip(le.classes_, le.transform(le.classes_)))

                if len(st.session_state['y_original_labels']) > 1:
                    display_to_original_map = {f"{label} - {count}": label for label, count in y_value_counts.items()}
                    current_interest_classes_original = st.session_state.get('interest_classes_selected', []) 
                    valid_current_interest_classes_display = [ 
                        f"{label} - {y_value_counts.get(label, 0)}" 
                        for label in current_interest_classes_original 
                        if label in y_value_counts.index
                    ]

                    selected_display_names = st.multiselect(
                        "Selecione as classes que representam a **'classe de interesse'** para a AUC:", 
                        options=y_unique_values_with_counts,
                        default=valid_current_interest_classes_display,
                        key="interest_classes_selection" 
                    )
                    
                    st.session_state['interest_classes_selected'] = [ 
                        display_to_original_map[display_name] for display_name in selected_display_names
                    ]
                    
                    if not st.session_state['interest_classes_selected']: 
                        st.warning("⚠️ **Nenhuma 'classe de interesse' selecionada.** A AUC não será calculada/plotada para problemas multiclasse.")
                    elif len(st.session_state['interest_classes_selected']) == len(st.session_state['y_original_labels']): 
                        st.warning("⚠️ **Todas as classes foram selecionadas como 'classe de interesse'.** Isso fará com que a binarização para AUC seja trivial.")
                    
                    if len(st.session_state['y_original_labels']) == 2 or len(st.session_state['interest_classes_selected']) > 0:
                        st.session_state['threshold_metric'] = st.selectbox(
                            "Escolha a métrica para determinar o limiar de corte ótimo:",
                            options=["Acurácia", "F1-Score", "Precisão", "Recall"],
                            index=["Acurácia", "F1-Score", "Precisão", "Recall"].index(st.session_state['threshold_metric']),
                            key="threshold_metric_selector"
                        )
                else:
                    st.info("A variável alvo tem apenas uma classe única. A curva ROC não é aplicável.")
                    st.session_state['interest_classes_selected'] = [] 
            else:
                st.info("Selecione a variável alvo (y) na etapa anterior para configurar as classes.")

    # ----- 6. Escolha de Modelos e Hiperparâmetros -----
    
    if st.session_state.get('data_preprocessed'):
        st.markdown("---")
        st.markdown("## 6. Escolha de Modelos e Hiperparâmetros")
        
        modelos_disponiveis = {
            "classification": {
                "SGDClassifier": SGDClassifier(random_state=42),
                "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
                "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
                "RandomForestClassifier": RandomForestClassifier(random_state=42), 
                "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
                "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
                "SVC": SVC(probability=True, random_state=42),
                "KNeighborsClassifier": KNeighborsClassifier(),
                "GaussianNB": GaussianNB(),
                "LGBMClassifier": lgb.LGBMClassifier(random_state=42)},
            "regression": {
                "LinearRegression": LinearRegression(),
                "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
                "RandomForestRegressor": RandomForestRegressor(random_state=42),
                "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
                "XGBRegressor": XGBRegressor(random_state=42),
                "SVR": SVR(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "LGBMRegressor": lgb.LGBMRegressor(random_state=42)
            }
        }

        problem_type = st.session_state.get('problem_type', 'classification')
        modelos_selecionados = st.multiselect(
            f"Selecione os modelos que deseja testar ({'Regressão' if problem_type == 'regression' else 'Classificação'}):",
            list(modelos_disponiveis[problem_type].keys()),
            default=list(modelos_disponiveis[problem_type].keys())
        )

        st.markdown("#### Tipo de Otimização de Hiperparâmetros")
        st.session_state['optimization_method'] = st.radio(
            "Selecione o método de otimização de hiperparâmetros:",
            ("GridSearchCV", "RandomizedSearchCV"),
            key="optimization_method_selector"
        )

        if st.session_state['optimization_method'] == "RandomizedSearchCV":
            n_iter = st.number_input(
                "Número de iterações para RandomizedSearchCV:", 
                min_value=5, 
                value=st.session_state.get('n_iter_random', 20),
                step=5, 
                key="n_iter_random_input"
            )
            st.session_state['n_iter_random'] = n_iter

        parametros_grid = {}

        if st.session_state.get('problem_type') == 'classification':
            with st.expander("🔧 Configurar Hiperparâmetros para Modelos de Classificação"):
                st.session_state['cv_folds'] = st.number_input(
                    "Número de folds para validação cruzada:",
                    min_value=2,
                    max_value=10,
                    value=st.session_state['cv_folds'],
                    step=1,
                    key="cv_folds_classification"
                )
                
                # Configurações de hiperparâmetros para cada modelo de classificação
                if "SGDClassifier" in modelos_selecionados:
                    st.subheader("SGDClassifier")
                    parametros_grid["SGDClassifier"] = {
                        'clf__loss': st.multiselect("Função de perda (loss)", ["hinge", "log_loss", "modified_huber", "squared_error"], default=["log_loss"], key="sgd_loss_grid"),
                        'clf__alpha': st.multiselect("Parâmetro de regularização (alpha)", [0.00001, 0.0001, 0.001, 0.01, 0.1], default=[0.0001], key="sgd_alpha_grid"),
                        'clf__penalty': st.multiselect("Penalidade (penalty)", ['l2', 'l1', 'elasticnet', None], default=['l2'], key="sgd_penalty_grid")
                    }

                if "LogisticRegression" in modelos_selecionados:
                    st.subheader("LogisticRegression")
                    C_options = [0.01, 0.1, 1.0, 10.0, 100.0]
                    solver_options = ["lbfgs", "liblinear", "saga"]
                    penalty_options = ['l1', 'l2', 'elasticnet', None]
                    
                    selected_solvers = st.multiselect("Solver", solver_options, default=["lbfgs"], key="lr_solver_grid")
                    
                    valid_penalties = []
                    if 'liblinear' in selected_solvers or 'saga' in selected_solvers:
                        valid_penalties.extend(['l1', 'l2', 'elasticnet'])
                    if 'lbfgs' in selected_solvers:
                        valid_penalties.append('l2')
                    if not selected_solvers:
                        valid_penalties = ['l1', 'l2', 'elasticnet', None]
                    valid_penalties = list(set(valid_penalties))
                    if 'lbfgs' in selected_solvers and None not in valid_penalties:
                        valid_penalties.append(None)
                    
                    parametros_grid["LogisticRegression"] = {
                        'clf__C': st.multiselect("Inverso da regularização (C)", C_options, default=[1.0], key="lr_c_grid"),
                        'clf__solver': selected_solvers,
                        'clf__penalty': st.multiselect("Penalidade (penalty)", penalty_options, default=['l2'], key="lr_penalty_grid")
                    }

                if "DecisionTreeClassifier" in modelos_selecionados:
                    st.subheader("DecisionTreeClassifier")
                    parametros_grid["DecisionTreeClassifier"] = {
                        'clf__max_depth': st.multiselect("Profundidade máxima (max_depth)", [3, 5, 7, 10, None], default=[5], key="dt_depth_grid"),
                        'clf__min_samples_split': st.multiselect("Mínimo de amostras para split", [2, 5, 10, 20], default=[2], key="dt_split_grid"),
                        'clf__criterion': st.multiselect("Critério de divisão (criterion)", ["gini", "entropy"], default=["gini"], key="dt_criterion_grid")
                    }

                if "RandomForestClassifier" in modelos_selecionados:
                    st.subheader("RandomForestClassifier")
                    parametros_grid["RandomForestClassifier"] = {
                        'clf__n_estimators': st.multiselect("Número de árvores (n_estimators)", [100, 200, 300, 500], default=[300], key="rf_est_grid"),
                        'clf__max_features': st.multiselect("Número de features para split (max_features)", ['sqrt', 'log2', None], default=['sqrt'], key="rf_max_features_grid"),
                        'clf__min_samples_leaf': st.multiselect("Mínimo de amostras por folha (min_samples_leaf)", [1, 2, 4], default=[1], key="rf_min_leaf_grid")
                    }

                if "GradientBoostingClassifier" in modelos_selecionados:
                    st.subheader("GradientBoostingClassifier")
                    parametros_grid["GradientBoostingClassifier"] = {
                        'clf__learning_rate': st.multiselect("Taxa de aprendizado (learning_rate)", [0.01, 0.05, 0.1, 0.2], default=[0.1], key="gb_lr_grid"),
                        'clf__n_estimators': st.multiselect("Número de estágios (n_estimators)", [100, 200, 300], default=[100], key="gb_est_grid"),
                        'clf__max_depth': st.multiselect("Profundidade máxima por estimador (max_depth)", [3, 5, 7], default=[3], key="gb_depth_grid")
                    }
                
                if "XGBClassifier" in modelos_selecionados:
                    st.subheader("XGBClassifier")
                    parametros_grid["XGBClassifier"] = {
                        'clf__learning_rate': st.multiselect("Taxa de aprendizado (learning_rate)", [0.01, 0.05, 0.1, 0.3], default=[0.1], key="xgb_lr_grid"),
                        'clf__n_estimators': st.multiselect("Número de estágios (n_estimators)", [100, 200, 300, 500], default=[100], key="xgb_est_grid"),
                        'clf__max_depth': st.multiselect("Profundidade máxima por estimador (max_depth)", [3, 5, 7, 9], default=[3], key="xgb_depth_grid"),
                        'clf__colsample_bytree': st.multiselect("Subamostragem de colunas por árvore (colsample_bytree)", [0.6, 0.8, 1.0], default=[1.0], key="xgb_colsample_grid")
                    }

                if "SVC" in modelos_selecionados:
                    st.subheader("SVC")
                    parametros_grid["SVC"] = {
                        'clf__C': st.multiselect("Parâmetro de regularização (C)", [0.1, 1, 10, 100], default=[1], key="svc_c_grid"),
                        'clf__kernel': st.multiselect("Tipo de kernel", ["linear", "rbf", "poly", "sigmoid"], default=["rbf"], key="svc_kernel_grid"),
                        'clf__gamma': st.multiselect("Coeficiente do kernel (gamma)", ['scale', 'auto', 0.01, 0.1, 1], default=['scale'], key="svc_gamma_grid")
                    }
                
                if "KNeighborsClassifier" in modelos_selecionados:
                    st.subheader("KNeighborsClassifier")
                    parametros_grid["KNeighborsClassifier"] = {
                        'clf__n_neighbors': st.multiselect("Número de vizinhos (n_neighbors)", [3, 5, 7, 9, 11], default=[5], key="knn_neighbors_grid"),
                        'clf__weights': st.multiselect("Peso dos vizinhos (weights)", ["uniform", "distance"], default=["uniform"], key="knn_weights_grid"),
                        'clf__metric': st.multiselect("Métrica de distância (metric)", ["euclidean", "manhattan"], default=["euclidean"], key="knn_metric_grid")
                    }
                
                if "GaussianNB" in modelos_selecionados:
                    st.subheader("GaussianNB")
                    parametros_grid["GaussianNB"] = {
                        'clf__var_smoothing': st.multiselect("Suavização de variância (var_smoothing)", [1e-9, 1e-8, 1e-7, 1e-6], default=[1e-9], key="gnb_smoothing_grid")
                    }
                
                if "LGBMClassifier" in modelos_selecionados:
                    st.subheader("LGBMClassifier")
                    parametros_grid["LGBMClassifier"] = {
                        'clf__n_estimators': st.multiselect("Número de estimadores", [100, 200, 300, 500], default=[100], key="lgbm_est_grid"),
                        'clf__learning_rate': st.multiselect("Taxa de aprendizado", [0.01, 0.05, 0.1, 0.2], default=[0.1], key="lgbm_lr_grid"),
                        'clf__num_leaves': st.multiselect("Número máximo de folhas", [20, 31, 40, 50], default=[31], key="lgbm_leaves_grid"),
                        'clf__max_depth': st.multiselect("Profundidade máxima da árvore", [5, 7, 10, -1], default=[-1], key="lgbm_depth_grid"),
                    }
                
        else:
            with st.expander("🔧 Configurar Hiperparâmetros para Modelos de Regressão"):
                st.session_state['cv_folds'] = st.number_input(
                    "Número de folds para validação cruzada:",
                    min_value=2,
                    max_value=10,
                    value=st.session_state['cv_folds'],
                    step=1,
                    key="cv_folds_regression"
                )
                
                # Configurações de hiperparâmetros para modelos de regressão
                if "LinearRegression" in modelos_selecionados:
                    st.subheader("LinearRegression")
                    parametros_grid["LinearRegression"] = {
                        'clf__fit_intercept': st.multiselect("fit_intercept (LinearRegression)",[True, False],default=[True],key="lr_fit_intercept"),
                        'clf__positive': st.multiselect("Restringir a coeficientes positivos (positive)",[True, False],default=[False],key="lr_positive"),
                        'clf__copy_X': st.multiselect("Copiar dados X (copy_X)",[True, False],default=[True],key="lr_copy_X")
                    }

                if "DecisionTreeRegressor" in modelos_selecionados:
                    st.subheader("DecisionTreeRegressor")
                    parametros_grid["DecisionTreeRegressor"] = {
                        'clf__max_depth': st.multiselect("Profundidade máxima (DecisionTree)", [None, 3, 5, 7, 10, 15], default=[None], key="dt_max_depth"),
                        'clf__min_samples_split': st.multiselect("min_samples_split (DecisionTree)", [2, 5, 10, 20], default=[2], key="dt_min_samples_split"),
                        'clf__criterion': st.multiselect("Critério (DecisionTree)", ["squared_error", "friedman_mse", "absolute_error"], default=["squared_error"], key="dt_criterion")
                    }

                if "RandomForestRegressor" in modelos_selecionados:
                    st.subheader("RandomForestRegressor")
                    parametros_grid["RandomForestRegressor"] = {
                        'clf__n_estimators': st.multiselect("Número de árvores (RandomForest)",[50, 100, 200, 300],default=[100],key="rf_n_estimators"),
                        'clf__max_features': st.multiselect("max_features (RandomForest)",["sqrt", "log2", None],  # Remova 'auto' e adicione None#
                            default=["sqrt"],key="rf_max_features"),
                        'clf__min_samples_leaf': st.multiselect("min_samples_leaf (RandomForest)",[1, 2, 4, 8],default=[1],key="rf_min_samples_leaf")
                    }

                if "XGBRegressor" in modelos_selecionados:
                    st.subheader("XGBRegressor")
                    parametros_grid["XGBRegressor"] = {
                        'clf__learning_rate': st.multiselect("Taxa de aprendizado (XGBoost)", [0.001, 0.01, 0.05, 0.1, 0.2], default=[0.1], key="xgb_learning_rate"),
                        'clf__n_estimators': st.multiselect("n_estimators (XGBoost)", [50, 100, 200, 300], default=[100], key="xgb_n_estimators"),
                        'clf__max_depth': st.multiselect("max_depth (XGBoost)", [3, 5, 7, 9, 12], default=[3], key="xgb_max_depth"),
                        'clf__subsample': st.multiselect("subsample (XGBoost)", [0.6, 0.8, 1.0], default=[1.0], key="xgb_subsample")
                    }

                if "SVR" in modelos_selecionados:
                    st.subheader("SVR")
                    parametros_grid["SVR"] = {
                        'clf__C': st.multiselect("C (SVR)", [0.1, 1, 10, 100], default=[1], key="svr_c"),
                        'clf__kernel': st.multiselect("kernel (SVR)", ["linear", "rbf", "poly"], default=["rbf"], key="svr_kernel"),
                        'clf__epsilon': st.multiselect("epsilon (SVR)", [0.01, 0.1, 0.5, 1.0], default=[0.1], key="svr_epsilon")
                    }

                if "KNeighborsRegressor" in modelos_selecionados:
                    st.subheader("KNeighborsRegressor")
                    parametros_grid["KNeighborsRegressor"] = {
                        'clf__n_neighbors': st.multiselect("n_neighbors (KNN)", [3, 5, 7, 9, 11], default=[5], key="knn_n_neighbors"),
                        'clf__weights': st.multiselect("weights (KNN)", ["uniform", "distance"], default=["uniform"], key="knn_weights"),
                        'clf__p': st.multiselect("p (KNN - 1=Manhattan, 2=Euclidean)", [1, 2], default=[2], key="knn_p")
                    }

                if "LGBMRegressor" in modelos_selecionados:
                    st.subheader("LGBMRegressor")
                    parametros_grid["LGBMRegressor"] = {
                        'clf__num_leaves': st.multiselect("num_leaves (LightGBM)", [20, 31, 40, 50], default=[31], key="lgbm_num_leaves"),
                        'clf__learning_rate': st.multiselect("learning_rate (LightGBM)", [0.01, 0.05, 0.1, 0.2], default=[0.1], key="lgbm_learning_rate"),
                        'clf__n_estimators': st.multiselect("n_estimators (LightGBM)", [50, 100, 200, 300], default=[100], key="lgbm_n_estimators"),
                        'clf__min_child_samples': st.multiselect("min_child_samples (LightGBM)", [5, 10, 20, 30], default=[20], key="lgbm_min_child_samples")
                    }

        st.session_state['modelos_selecionados'] = modelos_selecionados
        st.session_state['parametros_grid'] = parametros_grid
        st.session_state['modelos_configurados'] = True


    if st.session_state.get('modelos_configurados'):
        # ----- 7. Treinamento e Avaliação dos Modelos -----
        st.markdown("---")
        st.markdown("## 7. Treinamento e Avaliação dos Modelos")

        # Verifica se já existem resultados
        resultados_existentes = st.session_state.get('resultados_classificacao') or st.session_state.get('resultados_regressao')

        if not resultados_existentes:
            if st.button("🚀 Iniciar Treinamento e Comparação", type="primary"):
                df_trabalho = st.session_state.get('df_trabalho')
                x_cols = st.session_state.get('x_cols')
                y_col = st.session_state.get('y_col')
                
                if not (x_cols and y_col and df_trabalho is not None and not df_trabalho.empty):
                    st.error("❌ Por favor, selecione as colunas X e y antes de treinar.")
                    st.stop()
                
                problem_type = st.session_state.get('problem_type', 'classification')
                
                # Verificar se devemos usar os fatores
                if st.session_state.get('use_factors', False) and 'factors_df' in st.session_state:
                    # Usar os fatores no lugar das variáveis numéricas
                    factors_df = st.session_state['factors_df']
                    
                    # Obter colunas categóricas selecionadas
                    selected_cat_cols = st.session_state.get('selected_cat_cols', [])
                    
                    # Combinar fatores com colunas categóricas selecionadas
                    X = pd.concat([
                        factors_df,
                        df_trabalho[selected_cat_cols]
                    ], axis=1)
                    
                    # Atualizar as colunas X para incluir apenas os fatores e colunas categóricas selecionadas
                    x_cols = list(factors_df.columns) + selected_cat_cols
                    
                    st.info(f"🔧 Utilizando {len(factors_df.columns)} fatores principais + {len(selected_cat_cols)} variáveis categóricas selecionadas para modelagem.")
                else:
                    # Usar as variáveis originais
                    X = df_trabalho[x_cols]
                
                y = df_trabalho[y_col].values
                
                
                if problem_type == 'regression':
                    st.info(f"🔧 Iniciando processo de regressão com {len(modelos_selecionados)} modelo(s)...")
                    
                    try: 
                        resultados, X_test, y_test = train_regression_models(
                            modelos_disponiveis,
                            modelos_selecionados,
                            parametros_grid,
                            X, y,
                            st.session_state['optimization_method'],
                            st.session_state['n_iter_random'],
                            st.session_state['cv_folds']
                        )
                        
                        st.session_state['resultados_regressao'] = resultados
                        st.session_state['X_test_reg'] = X_test
                        st.session_state['y_test_reg'] = y_test
                        st.rerun()
                    
                    except Exception as e:
                        st.error(f"Erro no treinamento: {str(e)}")
                        st.stop()
                
                else:  # Classification
                    st.info(f"🔧 Iniciando processo de classificação com {len(modelos_selecionados)} modelo(s)...")
                    
                    try:
                        resultados, roc_curves, html_relatorio, X_test, y_test = train_classification_models(
                            modelos_disponiveis,
                            modelos_selecionados,
                            parametros_grid,
                            st.session_state['df_trabalho'][st.session_state['x_cols']],
                            st.session_state['df_trabalho'][st.session_state['y_col']],
                            st.session_state['optimization_method'],
                            st.session_state['n_iter_random'],
                            st.session_state['cv_folds'],
                            st.session_state.get('interest_classes_selected', []),
                            st.session_state['threshold_metric']
                        )
                        
                        st.session_state['resultados_classificacao'] = resultados
                        st.session_state['roc_curves_data'] = roc_curves
                        st.session_state['html_relatorio_classificacao'] = html_relatorio
                        st.session_state['X_test_cla'] = X_test
                        st.session_state['y_test_cla'] = y_test
                        st.rerun()
                    
                    except Exception as e:
                        st.error(f"Erro no treinamento: {str(e)}")
                        st.stop()

        else:
            st.info("✅ Modelos já foram treinados. Os resultados estão disponíveis abaixo.")
            
            if st.button("🔄 Recalcular Modelos", type="secondary"):
                st.session_state['resultados_classificacao'] = None
                st.session_state['resultados_regressao'] = None
                st.rerun()

            # Exibição dos resultados (para ambos os casos - novos ou existentes)
            if st.session_state.get('resultados_regressao'):
                resultados = st.session_state['resultados_regressao']
                X_test = st.session_state['X_test_reg']
                y_test = st.session_state['y_test_reg']
                
                st.markdown("## 📊 Comparação entre Modelos de Regressão")
                df_resultados = pd.DataFrame(resultados).sort_values('R²', ascending=False)
                
                # Criar colunas para os gráficos
                col1, col2 = st.columns(2)

                # Gráfico na primeira coluna
                with col1:
                    st.markdown("#### Comparação de R²")
                    fig_r2 = plot_metric_comparison(resultados, 'R²')
                    st.pyplot(fig_r2, use_container_width=True)
                    plt.close(fig_r2)

                # Gráfico na segunda coluna
                with col2:
                    st.markdown("#### Comparação de Erros")
                    fig_errors = plot_error_comparison(resultados, 'RMSE')
                    st.pyplot(fig_errors, use_container_width=True)
                    plt.close(fig_errors)

                # Linha divisória
                st.markdown("---")

                # Título para a tabela
                st.markdown("### Tabela Comparativa Detalhada")

                # Tabela com os resultados
                st.dataframe(
                    df_resultados[['Modelo', 'R²', 'RMSE', 'MAE', 'MSE']].style
                        .format({'R²': '{:.4f}', 'RMSE': '{:.4f}', 'MAE': '{:.4f}', 'MSE': '{:.4f}'})
                        .background_gradient(cmap='Blues', subset=['R²'])
                        .highlight_max(subset=['R²'], color='lightgreen')
                        .highlight_min(subset=['RMSE', 'MAE', 'MSE'], color='lightcoral')
    )
                
                st.markdown("## 📈 Visualizações Individuais por Modelo")
                for i, resultado in enumerate(resultados):
                    if i % 2 == 0:
                        cols = st.columns(2)
                    with cols[i % 2]:
                        with st.expander(f"{resultado['Modelo']} - R² = {resultado['R²']:.4f}", expanded=False):
                            st.pyplot(resultado['Gráfico'])
                            st.write("Melhores parâmetros:", resultado['Melhores Params'])
                
                html_report = generate_regression_html_report(resultados, X_test, y_test)
                st.session_state['html_relatorio_regressao'] = html_report
                
                st.markdown("### 📊 Relatório Completo")
                st.download_button(
                    label="⬇️ Baixar Relatório HTML",
                    data=html_report,
                    file_name="relatorio_regressao.html",
                    mime="text/html"
                )
                
                with st.expander("🔍 Visualizar Relatório HTML"):
                    components_html(html_report, height=1000, scrolling=True)
                
                melhor_modelo = df_resultados.loc[df_resultados['R²'].idxmax()]
                st.success(f"🏆 Melhor modelo: {melhor_modelo['Modelo']} com R² = {melhor_modelo['R²']:.4f}")
                
                with st.spinner("Preparando o melhor modelo para download..."):
                    modelo_base = modelos_disponiveis['regression'][melhor_modelo['Modelo']]
                    href = download_link(
                        modelo_base,
                        'melhor_modelo_regressao.pkl',
                        '⬇️ Baixar Melhor Modelo de Regressão'
                    )
                    st.markdown(href, unsafe_allow_html=True)

            elif st.session_state.get('resultados_classificacao'):
                resultados = st.session_state['resultados_classificacao']
                roc_curves = st.session_state['roc_curves_data']
                html_relatorio = st.session_state['html_relatorio_classificacao']
                X_test = st.session_state['X_test_cla']
                y_test = st.session_state['y_test_cla']
                df_resultados = pd.DataFrame(resultados)
                
                st.markdown("## 📊 Resultados da Classificação")
                
                cols_para_exibir = ['Modelo', 'Acurácia', 'AUC', 'PR-AUC', 'F1-Score', 'Precisão', 'Recall', 'Optimal Threshold']
                cols_presentes = [col for col in cols_para_exibir if col in df_resultados.columns]
                
                st.dataframe(
                    df_resultados[cols_presentes]
                    .sort_values('Acurácia', ascending=False)
                    .style
                    .format({col: '{:.4f}' for col in cols_presentes if col != 'Modelo'})
                    .background_gradient(cmap='Blues', subset=['Acurácia'])
                    .highlight_max(subset=['Acurácia', 'AUC', 'PR-AUC', 'F1-Score', 'Precisão', 'Recall'], color='lightgreen')
                    .highlight_min(subset=['Acurácia', 'AUC', 'PR-AUC', 'F1-Score', 'Precisão', 'Recall'], color='lightcoral')
                )
                
                if roc_curves:
                    st.markdown("### Comparação de Curvas ROC")
                    fig_combined, ax_combined = plt.subplots(figsize=(10, 8))
                    for curve in roc_curves:
                        roc_plot_label = f"{curve['model']} (AUC = {curve['auc']:.3f})"
                        if curve['plot_type'] == 'custom_binary':
                            int_lbls = ", ".join(map(str, curve['interest_classes'])) 
                            comp_lbls = ", ".join(map(str, curve['complementary_classes'])) 
                            roc_plot_label = f"{curve['model']} (AUC = {curve['auc']:.3f}) - Interesse: [{int_lbls}], Compl.: [{comp_lbls}]" 
                        ax_combined.plot(curve['fpr'], curve['tpr'], lw=2, label=roc_plot_label)
                    
                    ax_combined.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance')
                    ax_combined.set_xlim([0.0, 1.0])
                    ax_combined.set_ylim([0.0, 1.05])
                    ax_combined.set_xlabel('Taxa de Falsos Positivos')
                    ax_combined.set_ylabel('Taxa de Verdadeiros Positivos')
                    ax_combined.set_title('Comparação de Curvas ROC dos Modelos')
                    ax_combined.legend(loc='lower right', fontsize='small') 
                    plt.tight_layout() 
                    ax_combined.grid(alpha=0.3)
                    # Dividir em duas colunas no Streamlit
                    col1, col2 = st.columns(2)
                    # Exibir o gráfico em uma das colunas
                    with col1:
                        st.pyplot(fig_combined)
                    plt.close(fig_combined)
                
                st.markdown("### 📊 Relatório Completo")
                with st.expander("🔍 Visualizar Relatório HTML"):
                    components_html(html_relatorio, height=1000, scrolling=True)
                    
                st.download_button(
                    label="⬇️ Baixar Relatório HTML",
                    data=html_relatorio,
                    file_name="relatorio_classificacao.html",
                    mime="text/html"
                )
                
                if not df_resultados.empty:
                    melhor_modelo = df_resultados.loc[df_resultados['Acurácia'].idxmax()]
                    st.success(f"🏆 Melhor modelo: {melhor_modelo['Modelo']} com acurácia de {melhor_modelo['Acurácia']:.4f}")
                    
                    with st.spinner("Preparando o melhor modelo para download..."):
                        modelo_base = modelos_disponiveis['classification'][melhor_modelo['Modelo']]
                        href = download_link(
                            modelo_base,
                            'melhor_modelo_classificacao.pkl',
                            '⬇️ Baixar Melhor Modelo de Classificação'
                        )
                        st.markdown(href, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
