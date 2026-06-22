import streamlit as st
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
import re
import os
import tempfile
import platform
from datetime import datetime
import hashlib

# Importar as duas classes de analisadores
from analise_visual_avancada import IntegracaoStreamlit as IntegracaoTradicional
from siamese_network import AnalisadorSiamese # Importa a classe AnalisadorSiamese

# --- CONFIGURAÇÃO DO TESSERACT ---
tesseract_cmd_path = None
if platform.system() == "Windows":
    tesseract_cmd_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
elif platform.system() == "Darwin": # macOS
    if os.path.exists("/opt/homebrew/bin/tesseract"):
        tesseract_cmd_path = "/opt/homebrew/bin/tesseract"
    elif os.path.exists("/usr/local/bin/tesseract"):
        tesseract_cmd_path = "/usr/local/bin/tesseract"
elif platform.system() == "Linux":
    if os.path.exists("/usr/bin/tesseract"):
        tesseract_cmd_path = "/usr/bin/tesseract"
    elif os.path.exists("/bin/tesseract"):
        tesseract_cmd_path = "/bin/tesseract"

if tesseract_cmd_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path
else:
    st.warning("Tesseract não encontrado. A funcionalidade de OCR e detecção de orientação pode ser limitada.")

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Comparador de Artes PDF", layout="wide")
st.title("Comparador de Artes PDF")
st.markdown("Identifique omissões de texto, ícones e imagens em PDFs manipulados pela gráfica.")

# --- INICIALIZAÇÃO DO SESSION STATE ---
# Garante que todas as chaves do session_state existam antes de serem acessadas
if 'processamento_concluido' not in st.session_state:
    st.session_state.processamento_concluido = False
if 'resultados_termos' not in st.session_state:
    st.session_state.resultados_termos = []
if 'termos_faltando' not in st.session_state:
    st.session_state.termos_faltando = []
if 'integracao' not in st.session_state: # Para o analisador tradicional
    st.session_state.integracao = None
if 'analisador_siamese' not in st.session_state: # Para o analisador Siamese
    st.session_state.analisador_siamese = None
if 'texto_esq' not in st.session_state:
    st.session_state.texto_esq = ""
if 'texto_dir' not in st.session_state:
    st.session_state.texto_dir = ""
if 'file_hash_esq' not in st.session_state:
    st.session_state.file_hash_esq = ""
if 'file_hash_dir' not in st.session_state:
    st.session_state.file_hash_dir = ""
if 'img_esq_siamese' not in st.session_state: # Imagem pré-processada para Siamese
    st.session_state.img_esq_siamese = None
if 'img_dir_siamese' not in st.session_state: # Imagem pré-processada para Siamese
    st.session_state.img_dir_siamese = None
if 'diff_mapa_remocao_siamese' not in st.session_state:
    st.session_state.diff_mapa_remocao_siamese = None
if 'diff_mapa_adicao_siamese' not in st.session_state:
    st.session_state.diff_mapa_adicao_siamese = None


# --- FUNÇÕES CACHEADAS ---

@st.cache_data(show_spinner="Convertendo PDFs para imagem...")
def converter_pdf_para_imagem(caminho_pdf, dpi, file_id):
    """Converte um PDF para uma lista de imagens PIL, usando cache."""
    try:
        # Adicionado timeout para evitar que a conversão trave indefinidamente
        images = convert_from_path(caminho_pdf, dpi=dpi, timeout=300)
        return images
    except Exception as e:
        st.error(f"Erro ao converter PDF {caminho_pdf} para imagem: {e}")
        return []

@st.cache_data(show_spinner="Extraindo texto com OCR...")
def extrair_texto_ocr(_images: list, dpi, file_id): # Renomeado para _images para evitar erro de hash
    """Extrai texto de uma lista de imagens usando OCR, com cache."""
    texto_completo = []
    for i, img in enumerate(_images):
        try:
            # Adicionado timeout para evitar que o OCR trave indefinidamente
            text = pytesseract.image_to_string(img, lang='por', config=f'--dpi {dpi}', timeout=30)
            texto_completo.append(text)
        except Exception as e:
            st.warning(f"Erro ao extrair texto da página {i+1} com OCR: {e}")
            texto_completo.append("")

    return "\n".join(texto_completo)

@st.cache_resource(show_spinner="Inicializando analisador visual tradicional...")
def inicializar_integracao_visual(caminho_pdf_esq, caminho_pdf_dir, dpi, tesseract_cmd, rotacionar_manual_180, file_id_esq, file_id_dir):
    """Inicializa a integração do analisador visual tradicional, com cache."""
    return IntegracaoTradicional(caminho_pdf_esq, caminho_pdf_dir, dpi, tesseract_cmd, rotacionar_manual_180)

@st.cache_resource(show_spinner="Inicializando analisador Siamese Network...")
def inicializar_analisador_siamese():
    """Inicializa o AnalisadorSiamese, com cache."""
    # O AnalisadorSiamese não precisa dos caminhos dos PDFs, apenas das imagens PIL
    # e não precisa do tesseract_cmd ou rotacionar_manual_180, pois o pré-processamento
    # já foi feito pela IntegracaoTradicional (temporariamente)
    return AnalisadorSiamese()

# --- FUNÇÕES AUXILIARES ---

def normalizar(texto):
    """Normaliza o texto para comparação (minúsculas, sem espaços extras, sem acentos)."""
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r'\s+', ' ', texto).strip() # Remove múltiplos espaços e quebras de linha
    # Remove acentos (exemplo simplificado, pode ser melhorado com unicodedata)
    texto = texto.replace('á', 'a').replace('à', 'a').replace('ã', 'a').replace('â', 'a')
    texto = texto.replace('é', 'e').replace('ê', 'e')
    texto = texto.replace('í', 'i')
    texto = texto.replace('ó', 'o').replace('õ', 'o').replace('ô', 'o')
    texto = texto.replace('ú', 'u')
    texto = texto.replace('ç', 'c')
    return texto

# --- BARRA LATERAL ---
with st.sidebar:
    termos_para_checar = [
        "Lote", "Validade", "Fabricação", "Ingredientes",
        "Informação Nutricional", "Código de Barras"
    ]

    dpi_opcao = st.slider("DPI para conversão e OCR:", 72, 600, 300, step=10)
    sensibilidade = st.slider("Sensibilidade da Análise Visual:", 0.0, 1.0, 0.3, step=0.05)

    st.subheader("Escolha do Modelo Visual")
    tipo_analise_visual = st.radio(
        "Qual modelo de análise visual usar?",
        ("Tradicional (SSIM/OpenCV)", "Siamese Network (IA)"),
        index=0 # Padrão para Tradicional
    )

# --- UPLOAD DE ARQUIVOS ---
st.subheader("Upload dos PDFs")
col_upload_esq, col_upload_dir = st.columns(2)

with col_upload_esq:
    pdf_esquerda = st.file_uploader("PDF Original (Referência)", type="pdf", key="pdf_esq")
with col_upload_dir:
    pdf_direita = st.file_uploader("PDF Manipulado (Gráfica)", type="pdf", key="pdf_dir")

# --- BOTÃO DE VERIFICAÇÃO ---
if st.button("Verificar PDFs", type="primary"):
    if pdf_esquerda and pdf_direita:
        with st.spinner("Processando PDFs... Isso pode levar alguns minutos."):
            try:
                # Salvar PDFs temporariamente para que convert_from_path possa acessá-los
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_esq:
                    temp_esq.write(pdf_esquerda.getvalue())
                    caminho_esq = temp_esq.name
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_dir:
                    temp_dir.write(pdf_direita.getvalue())
                    caminho_dir = temp_dir.name

                # Calcular hashes dos arquivos para o cache
                st.session_state.file_hash_esq = hashlib.md5(pdf_esquerda.getvalue()).hexdigest()
                st.session_state.file_hash_dir = hashlib.md5(pdf_direita.getvalue()).hexdigest()

                # 1. CONVERSÃO E OCR
                imgs_esq = converter_pdf_para_imagem(caminho_esq, dpi_opcao, st.session_state.file_hash_esq)
                imgs_dir = converter_pdf_para_imagem(caminho_dir, dpi_opcao, st.session_state.file_hash_dir)

                st.session_state.texto_esq = extrair_texto_ocr(imgs_esq, dpi_opcao, st.session_state.file_hash_esq)
                st.session_state.texto_dir = extrair_texto_ocr(imgs_dir, dpi_opcao, st.session_state.file_hash_dir)

                texto_esq_norm = normalizar(st.session_state.texto_esq)
                texto_dir_norm = normalizar(st.session_state.texto_dir)

                # 2. ANÁLISE TEXTUAL
                st.session_state.resultados_termos = []
                st.session_state.termos_faltando = []

                for termo in termos_para_checar:
                    termo_norm = normalizar(termo)
                    esq_tem = termo_norm in texto_esq_norm
                    dir_tem = termo_norm in texto_dir_norm

                    if esq_tem and not dir_tem:
                        status = "FALTANDO"
                        st.session_state.termos_faltando.append(termo)
                    else:
                        status = "OK"
                    st.session_state.resultados_termos.append({"Termo": termo, "Status": status})

                # 3. INICIALIZAÇÃO DA ANÁLISE VISUAL AVANÇADA (com cache e file_ids)
                if tipo_analise_visual == "Tradicional (SSIM/OpenCV)":
                    st.session_state.integracao = inicializar_integracao_visual(
                        caminho_esq,
                        caminho_dir,
                        dpi_opcao,
                        tesseract_cmd_path,
                        False,
                        st.session_state.file_hash_esq,
                        st.session_state.file_hash_dir
                    )
                    st.session_state.analisador_siamese = None # Garante que o outro analisador seja nulo
                else: # Siamese Network (IA)
                    # Para a Siamese, precisamos das imagens PIL já pré-processadas (orientadas, sem bordas, normalizadas)
                    # Usamos uma instância temporária do AnalisadorVisual para fazer esse pré-processamento
                    # sem armazenar a instância completa no session_state.
                    with st.spinner("Pré-processando imagens para Siamese Network..."):
                        temp_analisador_visual = IntegracaoTradicional(
                            caminho_esq,
                            caminho_dir,
                            dpi_opcao,
                            tesseract_cmd_path,
                            False
                        )
                        if temp_analisador_visual.analisador:
                            st.session_state.img_esq_siamese = temp_analisador_visual.analisador.img_esquerda
                            st.session_state.img_dir_siamese = temp_analisador_visual.analisador.img_direita
                            st.session_state.integracao = None # Garante que o outro analisador seja nulo

                            st.session_state.analisador_siamese = inicializar_analisador_siamese()
                        else:
                            st.error("Falha no pré-processamento das imagens para Siamese Network.")
                            st.session_state.analisador_siamese = None

                st.session_state.processamento_concluido = True

            except Exception as e:
                st.exception(f"Erro crítico durante o processamento: {e}")
                st.session_state.processamento_concluido = False
                st.session_state.resultados_termos = []
                st.session_state.termos_faltando = []
                st.session_state.integracao = None
                st.session_state.analisador_siamese = None # Limpa também o analisador siamese
            finally:
                # Limpar arquivos temporários
                if 'caminho_esq' in locals() and os.path.exists(caminho_esq):
                    os.remove(caminho_esq)
                if 'caminho_dir' in locals() and os.path.exists(caminho_dir):
                    os.remove(caminho_dir)
    else:
        st.warning("Por favor, carregue ambos os PDFs para iniciar a verificação.")

# --- EXIBIÇÃO DOS RESULTADOS (APÓS PROCESSAMENTO CONCLUÍDO) ---
if st.session_state.processamento_concluido:
    tab1, tab2, tab3 = st.tabs(["Análise Textual", "Análise Visual", "Relatório"])

    with tab1:
        st.subheader("Análise de Texto com OCR")
        if st.session_state.resultados_termos:
            st.write("Resultados da checagem em tempo real:")
            for res in st.session_state.resultados_termos:
                if res["Status"] == "FALTANDO":
                    st.error(f"❌ FALTANDO: {res['Termo']}")
                else:
                    st.success(f"✅ OK: {res['Termo']}")

            if st.session_state.termos_faltando:
                st.warning(f"Total de omissões textuais encontradas: {len(st.session_state.termos_faltando)}")
            else:
                st.info("Nenhuma omissão textual detectada.")

            with st.expander("Ver texto extraído (OCR bruto)"):
                col_txt1, col_txt2 = st.columns(2)
                with col_txt1:
                    st.caption("PDF Original")
                    st.text(st.session_state.texto_esq)
                with col_txt2:
                    st.caption("PDF Manipulado")
                    st.text(st.session_state.texto_dir)
        else:
            st.info("Nenhum dado disponível.")

    with tab2:
        st.subheader("Análise Visual Avançada")
        if tipo_analise_visual == "Tradicional (SSIM/OpenCV)":
            if st.session_state.integracao:
                try:
                    st.session_state.integracao.exibir_analise_visual(sensibilidade=sensibilidade)
                except Exception as e:
                    st.exception(f"Erro na análise visual tradicional: {e}")
            else:
                st.warning("Módulo visual tradicional indisponível.")
        else: # Siamese Network (IA)
            if st.session_state.analisador_siamese and st.session_state.img_esq_siamese and st.session_state.img_dir_siamese:
                try:
                    # Calcula a área mínima para contornos com base no tamanho da imagem
                    h_img, w_img = np.array(st.session_state.img_esq_siamese).shape[:2]
                    area_total = h_img * w_img
                    area_minima_siamese = int(area_total * 0.0005) # 0.05% da área total

                    score, diff_mapa_simetrico, diff_mapa_remocao, diff_mapa_adicao, contours, thresh_combinado = \
                        st.session_state.analisador_siamese.comparar(
                            st.session_state.img_esq_siamese,
                            st.session_state.img_dir_siamese,
                            sensibilidade=sensibilidade
                        )

                    # Armazena os mapas direcionais no session_state
                    st.session_state.diff_mapa_remocao_siamese = diff_mapa_remocao
                    st.session_state.diff_mapa_adicao_siamese = diff_mapa_adicao

                    img_marcada_siamese, contagem_siamese = st.session_state.analisador_siamese.marcar_diferencas(
                        st.session_state.img_dir_siamese,
                        contours,
                        thresh_combinado, # Passa o thresh_combinado para marcar
                        area_minima=area_minima_siamese
                    )

                    st.subheader("Métricas da Análise (Siamese)")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            label="Similaridade (Siamese)",
                            value=f"{score * 100:.2f}%",
                            delta="Alta" if score >= 0.95 else "Baixa",
                            delta_color="normal" if score >= 0.95 else "inverse"
                        )
                    with col2:
                        st.metric(
                            label="Diferenças Detectadas",
                            value=contagem_siamese
                        )

                    st.subheader("Comparação Visual (Siamese)")
                    col_esq_siamese, col_dir_siamese = st.columns(2)
                    with col_esq_siamese:
                        st.image(
                            st.session_state.img_esq_siamese,
                            caption="PDF Original (pré-processado)",
                            use_container_width=True
                        )
                    with col_dir_siamese:
                        st.image(
                            img_marcada_siamese,
                            caption=f"PDF Manipulado — {contagem_siamese} diferença(s) marcada(s)",
                            use_container_width=True
                        )

                    with st.expander("Ver Overlay de Diferenças (Siamese)"):
                        overlay_siamese = st.session_state.analisador_siamese.gerar_overlay(st.session_state.img_dir_siamese, thresh_combinado)
                        st.image(overlay_siamese, caption="Overlay — regiões diferentes em vermelho", use_container_width=True)

                except Exception as e:
                    st.exception(f"Erro na análise visual Siamese: {e}")
            else:
                st.warning("Módulo visual Siamese indisponível ou imagens não pré-processadas.")

    with tab3:
        st.subheader("Relatório Consolidado")
        if st.session_state.resultados_termos:
            st.dataframe(st.session_state.resultados_termos, use_container_width=True, hide_index=True)

            timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            relatorio_texto = (
                f"RELATÓRIO DE VALIDAÇÃO GRÁFICA\n"
                f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n"
                f"{'=' * 40}\n\n"
            )

            for res in st.session_state.resultados_termos:
                relatorio_texto += f"[{res['Status']}] {res['Termo']}\n"

            if st.session_state.termos_faltando:
                relatorio_texto += f"\n{'=' * 40}\n"
                relatorio_texto += f"TOTAL DE OMISSÕES: {len(st.session_state.termos_faltando)}\n"
                for t in st.session_state.termos_faltando:
                    relatorio_texto += f"  - {t}\n"

            st.download_button(
                label="📥 Baixar Laudo Técnico (.txt)",
                data=relatorio_texto,
                file_name=f"laudo_auditoria_{timestamp}.txt",
                mime="text/plain"
            )
        else:
            st.info("Nenhum dado gerado para o relatório.")
