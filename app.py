import streamlit as st
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
import re
import os
import tempfile
from datetime import datetime
from analise_visual_avancada import IntegracaoStreamlit

st.set_page_config(page_title="Comparador de PDFs", layout="wide")
st.title("Comparador de Artes PDF")
st.markdown("Identifique omissões de texto, ícones e imagens em PDFs manipulados pela gráfica.")

# --- SIDEBAR ---
st.sidebar.header("Configurações")

termos_input = st.sidebar.text_area(
    "Termos críticos a verificar (um por linha):",
    "CAFÉ DA LINI\nARARA\nTORRA MÉDIA\n100% ARÁBICA\nBRASIL"
)
termos_para_checar = [t.strip() for t in termos_input.split('\n') if t.strip()]

rotacionar_180 = st.sidebar.checkbox(
    "Corrigir orientação do PDF Manipulado (rotacionar 180°)",
    value=False,
    help="Marque esta opção caso o PDF manipulado apareça de ponta-cabeça."
)

dpi_opcao = st.sidebar.select_slider(
    "Resolução de análise (DPI):",
    options=[72, 100, 150, 200, 300],
    value=150,
    help="DPI maior = mais detalhes detectados, porém mais lento."
)

# --- UPLOAD DOS PDFs ---
st.subheader("Carregar PDFs")
col1, col2 = st.columns(2)

with col1:
    pdf_esquerda = st.file_uploader("PDF Original (Esquerda)", type="pdf", key="esq")

with col2:
    pdf_direita = st.file_uploader("PDF Manipulado (Direita)", type="pdf", key="dir")


# --- FUNÇÃO DE NORMALIZAÇÃO DE TEXTO ---
def normalizar(texto):
    if not texto:
        return ""
    texto = texto.upper()
    texto = re.sub(r'[^A-Z0-9\s]', '', texto)
    return texto


# --- LÓGICA PRINCIPAL ---
if pdf_esquerda and pdf_direita:
    tmp_dir = tempfile.gettempdir()
    caminho_esq = os.path.join(tmp_dir, "pdf_esq.pdf")
    caminho_dir = os.path.join(tmp_dir, "pdf_dir.pdf")

    with open(caminho_esq, "wb") as f:
        f.write(pdf_esquerda.getbuffer())
    with open(caminho_dir, "wb") as f:
        f.write(pdf_direita.getbuffer())

    if st.button("Verificar PDFs", type="primary"):
        st.divider()

        with st.spinner("Processando PDFs e executando análises..."):
            try:
                # 1. PROCESSAMENTO DE TEXTO (OCR)
                imgs_esq = convert_from_path(caminho_esq, dpi=dpi_opcao)
                imgs_dir = convert_from_path(caminho_dir, dpi=dpi_opcao)

                texto_esq = pytesseract.image_to_string(imgs_esq[0], lang='por') if imgs_esq else ""
                texto_dir = pytesseract.image_to_string(imgs_dir[0], lang='por') if imgs_dir else ""

                texto_esq_norm = normalizar(texto_esq)
                texto_dir_norm = normalizar(texto_dir)

                resultados_termos = []
                termos_faltando = []

                for termo in termos_para_checar:
                    termo_norm = normalizar(termo)
                    esq_tem = termo_norm in texto_esq_norm
                    dir_tem = termo_norm in texto_dir_norm

                    if esq_tem and not dir_tem:
                        status = "FALTANDO"
                        termos_faltando.append(termo)
                    else:
                        status = "OK"

                    resultados_termos.append({"Termo": termo, "Status": status})

                # 2. ANÁLISE VISUAL
                integracao = IntegracaoStreamlit(caminho_esq, caminho_dir, rotacionar_180)

            except Exception as e:
                st.error(f"Erro crítico durante o processamento: {str(e)}")
                resultados_termos = []
                termos_faltando = []
                integracao = None

        # --- ABAS DE RESULTADO ---
        tab1, tab2, tab3 = st.tabs(["Análise Textual", "Análise Visual", "Relatório"])

        with tab1:
            st.subheader("Análise de Texto com OCR")
            if resultados_termos:
                for res in resultados_termos:
                    if res["Status"] == "FALTANDO":
                        st.error(f"❌ FALTANDO: {res['Termo']}")
                    else:
                        st.success(f"✅ OK: {res['Termo']}")

                if termos_faltando:
                    st.warning(f"Total de omissões textuais encontradas: {len(termos_faltando)}")
                else:
                    st.info("Nenhuma omissão textual detectada.")

                with st.expander("Ver texto extraído dos PDFs (OCR bruto)"):
                    col_txt1, col_txt2 = st.columns(2)
                    with col_txt1:
                        st.caption("Texto extraído — PDF Original")
                        st.text(texto_esq)
                    with col_txt2:
                        st.caption("Texto extraído — PDF Manipulado")
                        st.text(texto_dir)
            else:
                st.info("Nenhum dado disponível.")

        with tab2:
            st.subheader("Análise Visual Avançada")
            if integracao:
                try:
                    dados_visual = integracao.exibir_analise_visual()
                except Exception as e:
                    st.error(f"Erro na exibição da análise visual: {str(e)}")
            else:
                st.warning("Módulo visual indisponível.")

        with tab3:
            st.subheader("Relatório Consolidado")
            if resultados_termos:
                st.dataframe(resultados_termos, use_container_width=True, hide_index=True)

                timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
                relatorio_texto = (
                    f"RELATÓRIO DE VALIDAÇÃO GRÁFICA\n"
                    f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n"
                    f"{'=' * 40}\n\n"
                )
                for res in resultados_termos:
                    relatorio_texto += f"[{res['Status']}] {res['Termo']}\n"

                if termos_faltando:
                    relatorio_texto += f"\n{'=' * 40}\n"
                    relatorio_texto += f"TOTAL DE OMISSÕES: {len(termos_faltando)}\n"
                    for t in termos_faltando:
                        relatorio_texto += f"  - {t}\n"

                st.download_button(
                    label="📥 Baixar Laudo Técnico (.txt)",
                    data=relatorio_texto,
                    file_name=f"laudo_auditoria_{timestamp}.txt",
                    mime="text/plain"
                )
            else:
                st.info("Nenhum dado gerado para o relatório.")
