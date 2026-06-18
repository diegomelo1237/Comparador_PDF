import streamlit as st
from analise_visual_avancada import IntegracaoStreamlit

# Configura a página para o modo amplo (ocupa melhor a tela do navegador)
st.set_page_config(
    page_title="Revisão Automatizada de PDFs",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Comparador e Revisor Automatizado de PDFs")
st.markdown("""
    Esta ferramenta compara de forma **100% automática** dois arquivos PDF. 
    Qualquer texto alterado, elemento removido ou erro de layout será alinhado e destacado em azul.
""")

st.markdown("---")

# Cria duas colunas organizadas lado a lado para o upload dos arquivos
col_up1, col_up2 = st.columns(2)

with col_up1:
    st.subheader("📄 Arquivo Original (Referência)")
    arquivo_esq = st.file_uploader(
        "Envie o PDF 1 (Esquerda)", 
        type=["pdf"], 
        key="upload_pdf_esq_unico",
        label_visibility="collapsed"
    )

with col_up2:
    st.subheader("📄 Arquivo Modificado (Revisão)")
    arquivo_dir = st.file_uploader(
        "Envie o PDF 2 (Direita)", 
        type=["pdf"], 
        key="upload_pdf_dir_unico",
        label_visibility="collapsed"
    )

# Dispara o processamento apenas quando os dois arquivos forem carregados
if arquivo_esq is not None and arquivo_dir is not None:
    st.markdown("---")
    
    # Exibe uma mensagem de carregamento enquanto o Python processa e alinha as imagens
    with st.spinner("Alinhando páginas e analisando textos/elementos automaticamente..."):
        # Instancia a classe que configuramos no outro arquivo
        app_integracao = IntegracaoStreamlit(arquivo_esq, arquivo_dir)
        
        # Executa a renderização e exibe o resultado final com as caixas azuis na tela
        app_integracao.exibir_analise_visual()
        
else:
    st.markdown("---")
    st.info("💡 **Aguardando arquivos:** Por favor, faça o upload de ambos os PDFs acima para iniciar a revisão visual automatizada.")