import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
# from sklearn.preprocessing import RobustScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
import joblib
from rdkit.Chem import Descriptors

# Função para processar e prever SMILES e gerar o DataFrame com os descritores

from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import streamlit as st

def process_smiles(smiles_list, model, feature_list):
    molecules = []
    descriptors_list = []

    # Processar todos os SMILES para calcular os descritores
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            molecules.append(mol)
            descriptors = [
                getattr(Descriptors, desc)(mol) if getattr(Descriptors, desc, None) else None
                for desc in feature_list
            ]
            descriptors_list.append(descriptors)
        else:
            molecules.append(None)
            descriptors_list.append([None] * len(feature_list))  # Molécula inválida

    # Criar um DataFrame com os descritores calculados
    features_df = pd.DataFrame(descriptors_list, columns=feature_list)
    # st.write("Descritores calculados:")
    # st.write(features_df)

    # Remover a parte de normalização
    # Não há normalização ou pré-processamento agora.

    # Fazer previsões para os SMILES válidos
    predictions = model.predict(features_df)

    # Criar um DataFrame com os SMILES, descritores e predições
    results = pd.DataFrame({
        'SMILES': smiles_list,
        **{desc: features_df[desc] for desc in feature_list},
        'Prediction': predictions
    })

    st.write("resultados:")
    st.write(results)
    
    return results, molecules, predictions


# Carregar o modelo treinado
def load_model():
    return joblib.load("model_with_features.pkl")  # Substitua pelo caminho correto do modelo

# Streamlit UI
st.title("Previsão de Propriedades de Moléculas")
st.markdown("""
    Insira um ou mais SMILES (separados por vírgula) para gerar os descritores e fazer a previsão.
""")

# Entrada do usuário para os SMILES
smiles_input = st.text_area("Digite SMILES")
# Carregar modelo e features
modelo_salvo = load_model()
model = modelo_salvo["model"]
feature_list = modelo_salvo["features"]

# Lista de SMILES do usuário
smiles_list = [smiles.strip() for smiles in smiles_input.split(",") if smiles.strip()]

# Botão para carregar um arquivo Excel com SMILES
uploaded_file = st.file_uploader("Carregar arquivo Excel com SMILES", type="xlsx")

# Processar o arquivo carregado
if st.button("Processar"):
    if uploaded_file:
        # Carregar e processar o arquivo Excel
        df = pd.read_excel(uploaded_file)
        smiles_list.extend(df["smiles"].dropna().tolist())

    if smiles_list:
        results, molecules, predictions = process_smiles(smiles_list, model, feature_list)

        # Exibir resultados
        for mol, pred in zip(molecules, predictions):
            if mol:
                img = Draw.MolToImage(mol, size=(300, 300))  # Diminuir o tamanho da imagem (300x300)
                st.image(img, caption=f"Predição: {pred}", width=300)  # Exibe a imagem com o novo tamanho
                
                # Aumentar o tamanho da fonte da predição
                if pred == "ativa":
                    st.success(f"<h2 style='font-size:24px; color: green;'>Predição: {pred}</h2>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h3 style='font-size:24px;'>Predição: {pred}</h3>", unsafe_allow_html=True)
            else:
                st.warning("Molécula inválida.")
    else:
        st.warning("Nenhum SMILES válido foi encontrado.")
