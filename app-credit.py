import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px

st.set_page_config(page_title="Análise de Crédito", layout="wide")
st.title("Dashboard de Análise de Crédito")
st.markdown("""
Esse dashboard foi criado para prática de ciência de dados pelo grupo formado por Anderlan Siqueira, André Poroca, Bruno Souza, Carlos Nascimento, José Silvio, Hélio Ricardo e Laís Dagnésia para demonstrar a análise de crédito utilizando dados fictícios.""") 

# Carregando dados
# DATA_PATH = "credit_data.csv"
# data = pd.read_csv(DATA_PATH)

@st.cache_data
def load_data():
    data_url = "https://raw.githubusercontent.com/laisdagnesia/CESAR_IA/refs/heads/main/credit_data.csv"
    data = pd.read_csv(data_url, encoding="ISO-8859-1")
    return data

data = load_data()

#Ajustes nos dados
data = data[data['age'] > 0] 
data = data.rename(columns={
    'default': 'risco',
    'balance': 'saldo',
    'income': 'renda',
    'age': 'idade',
    'loan': 'emprestimo'
})
data = data.drop(columns=['clientid'])


st.subheader("Visualização dos Dados")
st.markdown("A coluna `risco` representa se há risco de crédito:\n- `0` = não\n- `1` = sim")
st.dataframe(data.head())

st.subheader("Estatísticas Descritivas")
st.markdown("A coluna `risco` representa se há risco de crédito:\n- `0` = não\n- `1` = sim")
st.write(data.describe())

st.subheader("Distribuição das Variáveis Numéricas")
selected_column: str = st.selectbox("Escolha a coluna para histograma", data.select_dtypes(include=['float64', 'int64']).columns)
fig, ax = plt.subplots()
sns.histplot(data[selected_column], kde=True, ax=ax)
st.pyplot(fig)

st.subheader("Relação entre Variáveis")
num_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
col_x: str = st.selectbox("Variável X", num_cols, key="x")
col_y: str = st.selectbox("Variável Y", num_cols, key="y")
fig2, ax2 = plt.subplots()
sns.scatterplot(x=col_x, y=col_y, data=data, ax=ax2)
st.pyplot(fig2)

# Gráfico de Dispersão Interativo

st.subheader("Gráfico de Dispersão Interativo")

x_axis = st.selectbox("Escolha a variável para o eixo X", data.select_dtypes(include=['float64', 'int64', 'object']).columns)
y_axis = st.selectbox("Escolha a variável para o eixo Y", data.select_dtypes(include=['float64', 'int64', 'object']).columns)
color_by = st.selectbox("Escolha a variável para cor", data.columns)

if color_by == "risco":
    st.markdown("A coluna `risco` representa se há risco de crédito:\n- `0` = não\n- `1` = sim")
    data['risco'] = data['risco'].astype(str)
    fig = px.scatter(
        data,
        x=x_axis,
        y=y_axis,
        color='risco',
        color_discrete_map={"0": "green", "1": "red"},
        title="Gráfico de Dispersão por Risco"
    )
else:
    fig = px.scatter(data, x=x_axis, y=y_axis, color=color_by)

st.plotly_chart(fig)

# Machine Learning
st.subheader("Aplicação de Machine Learning para Análise de Crédito")
target: str = st.selectbox("Escolha a variável alvo - target", data.columns)
features = st.multiselect("Escolha as variáveis preditoras - features", [col for col in data.columns if col != target])

if features:
    X = data[features]
    y = data[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    is_classification = y.nunique() <= 10 and y.dtype in ['int64', 'object']

    if is_classification:
        st.markdown("**Detecção automática:** variável alvo parece ser categórica (classificação).")
        model_type = st.selectbox("Escolha o modelo de ML", ["Regressão Logística", "Árvore de Decisão"])

        if model_type == "Regressão Logística":
            model = LogisticRegression()
        else:
            model = DecisionTreeClassifier()

        if st.button("Treinar Modelo"):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=False)

            st.success(f"Acurácia: {acc:.2f}")
            st.text("Matriz de Confusão:")
            st.write(cm)
            st.text("Relatório de Classificação:")
            st.text(report)

    else:
        st.markdown("**Detecção automática:** variável alvo parece ser contínua (regressão).")
        model_type = st.selectbox("Escolha o modelo de Regressão", ["Regressão Linear", "Random Forest Regressor"])

        if model_type == "Regressão Linear":
            model = LinearRegression()
        else:
            model = RandomForestRegressor()

        if st.button("Treinar Modelo"):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.success("Modelo treinado com sucesso!")
            st.markdown(f"**Erro Absoluto MAE:** {mae:.2f}")
            st.markdown(f"**Erro Quadrático MES:** {mse:.2f}")
            st.markdown(f"**R²:** {r2:.2f}")
            
# Tabela com filtros
st.subheader("Tabela de Dados com Filtros")
salario_min = st.slider('Renda Mínima', int(data['renda'].min()), int(data['renda'].max()))
df_filtrado = data[data['renda'] >= salario_min]
st.dataframe(df_filtrado)
            
# st.download_button("Baixar CSV filtrado", data=df_filtrado.to_csv(index=False), file_name='dados_filtrados.csv')
