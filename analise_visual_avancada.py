import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import streamlit as st
import pytesseract
import re

class AnalisadorVisual:
    def __init__(self, img_esquerda, img_direita, tesseract_cmd=None, rotacionar_manual_180=False, dpi=150):
        self.tesseract_cmd = tesseract_cmd
        if self.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        self.dpi = dpi

        # Converte para RGB e faz uma cópia para garantir que não estamos modificando a imagem original
        self.img_esquerda = img_esquerda.convert('RGB').copy()
        self.img_direita = img_direita.convert('RGB').copy()

        # --- ORDEM DE PROCESSAMENTO OTIMIZADA ---
        # 1. Corrigir orientação inicial (sempre retrato) para AMBAS as imagens
        self.img_esquerda = self._garantir_orientacao_retrato(self.img_esquerda)
        self.img_direita = self._garantir_orientacao_retrato(self.img_direita)

        # 2. Aplicar rotação manual de 180 graus se solicitada (apenas na direita)
        if rotacionar_manual_180:
            self.img_direita = Image.fromarray(cv2.rotate(np.array(self.img_direita), cv2.ROTATE_180))
            st.info("Rotação manual de 180° aplicada ao PDF manipulado.")
        else:
            # 3. Tentar correção de 180 graus via OSD (se não houver rotação manual)
            self.img_direita = self._corrigir_orientacao_osd(self.img_direita)

        # 4. Remover bordas brancas de AMBAS as imagens
        self.img_esquerda = self._remover_bordas_brancas(self.img_esquerda)
        self.img_direita = self._remover_bordas_brancas(self.img_direita)
        st.info("Bordas brancas removidas das imagens.")

        # 5. Normalizar tamanhos (redimensionar se necessário)
        self.normalizar_tamanhos()

        # 6. Alinhar imagens (corrigir pequenos deslocamentos)
        self._alinhar_imagens()

        # Inicializa variáveis para o relatório
        self.diff_mapa = None
        self.thresh_mask = None
        self.contornos = None
        self.similaridade = 0.0
        self.contornos_encontrados = 0
        self.diferenca_elementos = 0

    def _garantir_orientacao_retrato(self, img: Image.Image) -> Image.Image:
        """
        Garante que a imagem esteja em orientação retrato (altura >= largura).
        Rotaciona 90 graus no sentido anti-horário se a largura for maior que a altura.
        """
        img_array = np.array(img)
        altura, largura = img_array.shape[:2]
        if largura > altura:
            img_array = cv2.rotate(img_array, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return Image.fromarray(img_array)

    def _corrigir_orientacao_osd(self, img: Image.Image) -> Image.Image:
        """
        Tenta corrigir rotação de 180 graus (ponta-cabeça) usando OSD do Tesseract.
        """
        img_array = np.array(img)
        try:
            if self.tesseract_cmd:
                pil_img = Image.fromarray(img_array)
                # Adicionado timeout para evitar que o Tesseract trave indefinidamente
                osd_data = pytesseract.image_to_osd(pil_img, config=f'--dpi {self.dpi} --psm 0', timeout=5) # psm 0 para OSD
                match = re.search(r"Rotate: (\d+)", osd_data)
                if match:
                    rotation_angle = int(match.group(1))
                    if rotation_angle == 180:
                        img_array = cv2.rotate(img_array, cv2.ROTATE_180)
                        st.info("Orientação automática corrigida (180°) no PDF manipulado via OSD.")
            else:
                st.warning("Tesseract não configurado. A detecção de orientação via OSD não será realizada.")
        except Exception as e:
            st.warning(f"Erro na detecção de orientação OSD: {e}. A imagem não será rotacionada automaticamente.")
        return Image.fromarray(img_array)

    def _remover_bordas_brancas(self, img: Image.Image, margin: int = 5) -> Image.Image:
        """
        Remove bordas brancas de uma imagem PIL.
        Adiciona uma pequena margem para não cortar conteúdo.
        """
        img_array = np.array(img)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        # Inverte para que o conteúdo seja branco e o fundo preto
        # Isso ajuda a encontrar o contorno do conteúdo
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV) # Pixels quase brancos viram preto

        # Encontra os pixels não brancos (conteúdo)
        # Certifica-se de que thresh é um array numpy de tipo uint8 ou similar para findNonZero
        if thresh.dtype != np.uint8:
            thresh = thresh.astype(np.uint8)

        coords = cv2.findNonZero(thresh)
        if coords is None: # Imagem completamente branca ou preta
            return img

        x, y, w, h = cv2.boundingRect(coords)

        # Adiciona margem, garantindo que não saia dos limites da imagem
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img_array.shape[1], x + w + margin)
        y2 = min(img_array.shape[0], y + h + margin)

        cropped_img_array = img_array[y1:y2, x1:x2]
        return Image.fromarray(cropped_img_array)

    def normalizar_tamanhos(self):
        """Redimensiona a imagem da direita para o tamanho da esquerda."""
        w_esq, h_esq = self.img_esquerda.size
        w_dir, h_dir = self.img_direita.size

        if (w_esq, h_esq) != (w_dir, h_dir):
            # Escolhe a interpolação baseada se está aumentando ou diminuindo
            interpolation = cv2.INTER_AREA if (w_dir * h_dir) > (w_esq * h_esq) else cv2.INTER_LANCZOS4
            self.img_direita = Image.fromarray(
                cv2.resize(np.array(self.img_direita), (w_esq, h_esq), interpolation=interpolation)
            )
            st.info(f"PDF manipulado redimensionado de {w_dir}x{h_dir} para {w_esq}x{h_esq}.")

    def _alinhar_imagens(self):
        """
        Tenta alinhar a imagem da direita com a da esquerda usando ECC.
        Se falhar, continua sem alinhamento fino.
        """
        img_esq_gray = cv2.cvtColor(np.array(self.img_esquerda), cv2.COLOR_RGB2GRAY)
        img_dir_gray = cv2.cvtColor(np.array(self.img_direita), cv2.COLOR_RGB2GRAY)

        # Define o tipo de movimento (translação, rotação, escala)
        warp_mode = cv2.MOTION_AFFINE # MOTION_AFFINE é um bom balanço entre precisão e performance
        warp_matrix = np.eye(2, 3, dtype=np.float32) # Matriz de transformação inicial

        number_of_iterations = 1000 # Aumentado para tentar mais convergência
        termination_eps = 1e-6 # Reduzido para maior precisão
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

        try:
            (cc, warp_matrix) = cv2.findTransformECC(
                img_esq_gray,
                img_dir_gray,
                warp_matrix,
                warp_mode,
                criteria,
                inputMask=None,
                gaussFiltSize=5 # Aumentado para suavizar ruído
            )
            # Aplica a transformação à imagem colorida
            self.img_direita = Image.fromarray(
                cv2.warpAffine(
                    np.array(self.img_direita),
                    warp_matrix,
                    (img_esq_gray.shape[1], img_esq_gray.shape[0]),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REFLECT_101 # Preenche bordas com reflexo
                )
            )
            st.success(f"Imagens alinhadas com sucesso (Correlação: {cc:.4f}).")
        except cv2.error as e:
            if "Iterations do not converge" in str(e):
                st.warning("Não foi possível alinhar imagens automaticamente: Iterações não convergiram. A detecção pode ser menos precisa.")
            else:
                st.warning(f"Erro inesperado no alinhamento: {e}. A detecção pode ser menos precisa.")
        except Exception as e:
            st.warning(f"Erro geral no alinhamento: {e}. A detecção pode ser menos precisa.")

    def gerar_relatorio_visual(self, sensibilidade: float = 0.3):
        """
        Compara as imagens e gera um relatório visual com métricas e máscaras.
        Armazena os resultados internamente para evitar reprocessamento.
        """
        img_esq_cv = np.array(self.img_esquerda)
        img_dir_cv = np.array(self.img_direita)

        # 1. SSIM (Structural Similarity Index)
        # Converte para escala de cinza para SSIM
        img_esq_gray = cv2.cvtColor(img_esq_cv, cv2.COLOR_RGB2GRAY)
        img_dir_gray = cv2.cvtColor(img_dir_cv, cv2.COLOR_RGB2GRAY)

        # Calcula SSIM. data_range é importante para imagens uint8
        self.similaridade, _ = ssim(img_esq_gray, img_dir_gray, data_range=img_dir_gray.max(), full=False)

        # 2. Diferença Absoluta e Threshold
        # Calcula a diferença absoluta entre as imagens em escala de cinza
        diff = cv2.absdiff(img_esq_gray, img_dir_gray)

        # Aplica um threshold para binarizar a imagem de diferença
        # O valor de threshold é ajustado pela sensibilidade
        # Quanto menor a sensibilidade (mais próximo de 0), mais diferenças são detectadas (thresh_val alto)
        # Quanto maior a sensibilidade (mais próximo de 1), menos diferenças são detectadas (thresh_val baixo)
        # Invertemos a lógica para que sensibilidade alta = mais rigoroso (menos diferenças)
        valor_thresh = int(255 * (1 - sensibilidade)) # Sensibilidade 0.3 -> valor_thresh 178
                                                    # Sensibilidade 0.8 -> valor_thresh 51
        _, self.thresh_mask = cv2.threshold(diff, valor_thresh, 255, cv2.THRESH_BINARY)

        # 3. Encontrar Contornos
        # Encontra contornos na máscara de diferença
        # cv2.RETR_EXTERNAL recupera apenas os contornos externos
        # cv2.CHAIN_APPROX_SIMPLE compacta segmentos horizontais, verticais e diagonais
        contours, _ = cv2.findContours(self.thresh_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contornos = contours

        # 4. Contagem de Elementos Gráficos (exemplo simplificado)
        # Pode ser aprimorado com detecção de features mais robusta
        num_elementos_esq = len(cv2.findContours(img_esq_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0])
        num_elementos_dir = len(cv2.findContours(img_dir_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0])
        self.diferenca_elementos = num_elementos_esq - num_elementos_dir

        # Filtra contornos muito pequenos (ruído)
        h_img, w_img = img_esq_cv.shape[:2]
        area_total = h_img * w_img
        area_minima = int(area_total * 0.0001) # 0.01% da área total como mínimo

        self.contornos_encontrados = sum(1 for c in self.contornos if cv2.contourArea(c) > area_minima)

        return {
            "similaridade": self.similaridade,
            "diff_mapa": diff,
            "thresh_mask": self.thresh_mask,
            "contornos_encontrados": self.contornos_encontrados,
            "diferenca_elementos": self.diferenca_elementos
        }

    def marcar_diferencas(self, img_original: Image.Image, contours, diff_mask: np.ndarray, area_minima: int = 100):
        """
        Marca as diferenças encontradas na imagem manipulada com retângulos e números.
        """
        img_marcada = np.array(img_original.convert("RGB")).copy()
        contagem = 0
        for i, c in enumerate(contours):
            if cv2.contourArea(c) > area_minima:
                x, y, w, h = cv2.boundingRect(c)

                # Desenha um círculo em vez de um retângulo
                center_x = x + w // 2
                center_y = y + h // 2
                # Raio proporcional à maior dimensão do contorno
                radius = int(max(w, h) * 0.6)

                cv2.circle(img_marcada, (center_x, center_y), radius, (255, 0, 0), 3) # Círculo vermelho

                # Adiciona o número da diferença com fundo preto para contraste
                label = str(contagem + 1)
                (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

                # Garante que o texto e o fundo não saiam da imagem
                tx = max(0, x)
                ty = max(text_h + baseline + 5, y) # Garante que o texto não saia para cima

                # Ajusta a posição do retângulo de fundo
                rect_x1 = max(0, tx - 2)
                rect_y1 = max(0, ty - text_h - baseline - 5)
                rect_x2 = min(img_marcada.shape[1], tx + text_w + 2)
                rect_y2 = min(img_marcada.shape[0], ty)

                cv2.rectangle(img_marcada, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1) # Fundo preto
                cv2.putText(
                    img_marcada,
                    label,
                    (tx, ty - baseline - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255), # Texto branco
                    2
                )
                contagem += 1

        return Image.fromarray(img_marcada), contagem

    def gerar_heatmap_diferencas(self, diff_mapa):
        """Gera heatmap colorido — azul = similar, vermelho = diferente."""
        diff_norm = cv2.normalize(diff_mapa, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(diff_norm.astype("uint8"), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        return Image.fromarray(heatmap_rgb)

    def gerar_overlay_diferencas(self, thresh_mask):
        """Gera overlay vermelho semitransparente sobre as regiões diferentes."""
        base = np.array(self.img_direita).copy()
        overlay = base.copy()
        mascara = thresh_mask > 0
        overlay[mascara] = [255, 50, 50] # Vermelho claro
        resultado = cv2.addWeighted(base, 0.6, overlay, 0.4, 0)
        return Image.fromarray(resultado)

class IntegracaoStreamlit:
    def __init__(self, caminho_pdf_esq, caminho_pdf_dir, dpi, tesseract_cmd, rotacionar_manual_180):
        self.caminho_pdf_esq = caminho_pdf_esq
        self.caminho_pdf_dir = caminho_pdf_dir
        self.dpi = dpi
        self.tesseract_cmd = tesseract_cmd
        self.rotacionar_manual_180 = rotacionar_manual_180
        self.analisador = None
        self.relatorio_visual = None # Armazena o relatório visual para evitar reprocessamento
        self._inicializar_analisador()

    def _inicializar_analisador(self):
        from pdf2image import convert_from_path # Importa aqui para evitar circular import
        try:
            # Sempre usa convert_from_path pois app.py já salva em tempfile
            imgs_esq = convert_from_path(self.caminho_pdf_esq, dpi=self.dpi, timeout=300)
            imgs_dir = convert_from_path(self.caminho_pdf_dir, dpi=self.dpi, timeout=300)

            if imgs_esq and imgs_dir:
                self.analisador = AnalisadorVisual(
                    imgs_esq[0], imgs_dir[0],
                    tesseract_cmd=self.tesseract_cmd,
                    rotacionar_manual_180=self.rotacionar_manual_180,
                    dpi=self.dpi # Passa o DPI para AnalisadorVisual
                )
            else:
                st.error("Não foi possível converter um ou ambos os PDFs para imagem.")

        except Exception as e:
            st.error(f"Erro ao processar PDFs: {str(e)}")
            self.analisador = None # Garante que o analisador não seja usado se houver erro

    def exibir_analise_visual(self, sensibilidade=0.3):
        if not self.analisador:
            st.warning("Analisador não inicializado ou ocorreu um erro no processamento inicial.")
            return None

        # Gera o relatório visual apenas uma vez
        # A sensibilidade afeta a detecção de diferenças, então o relatório visual precisa ser gerado
        # sempre que a sensibilidade muda.
        self.relatorio_visual = self.analisador.gerar_relatorio_visual(sensibilidade)

        relatorio = self.relatorio_visual
        img_marcada, contagem_diferencas = self.analisador.marcar_diferencas(
            self.analisador.img_direita,
            relatorio['contornos'],
            relatorio['thresh_mask'],
            area_minima=int(self.analisador.img_direita.size[0] * self.analisador.img_direita.size[1] * 0.0001)
        )

        # --- MÉTRICAS ---
        st.subheader("Métricas da Análise")
        col1, col2, col3 = st.columns(3)

        with col1:
            similaridade_pct = relatorio['similaridade'] * 100
            st.metric(
                label="Similaridade (SSIM)",
                value=f"{similaridade_pct:.2f}%",
                delta="Alta" if similaridade_pct >= 95 else "Baixa",
                delta_color="normal" if similaridade_pct >= 95 else "inverse"
            )
        with col2:
            st.metric(
                label="Diferenças Detectadas",
                value=contagem_diferencas
            )
        with col3:
            st.metric(
                label="Δ Elementos Gráficos",
                value=relatorio['diferenca_elementos'],
                help="Positivo = original tem mais elementos que o manipulado"
            )

        # --- COMPARAÇÃO LADO A LADO ---
        st.subheader("Comparação Visual")
        col_esq, col_dir = st.columns(2)

        with col_esq:
            st.image(
                self.analisador.img_esquerda,
                caption="PDF Original",
                use_container_width=True
            )
        with col_dir:
            st.image(
                img_marcada,
                caption=f"PDF Manipulado — {contagem_diferencas} diferença(s) marcada(s)",
                use_container_width=True
            )

        # --- VISUALIZAÇÕES EXTRAS ---
        with st.expander("Ver Overlay de Diferenças"):
            overlay = self.analisador.gerar_overlay_diferencas(relatorio['thresh_mask'])
            st.image(overlay, caption="Overlay — regiões diferentes em vermelho", use_container_width=True)

        with st.expander("Ver Heatmap de Diferenças (Debug)"):
            heatmap = self.analisador.gerar_heatmap_diferencas(relatorio['diff_mapa'])
            st.image(heatmap, caption="Heatmap — Azul = similar | Vermelho = diferente", use_container_width=True)

        with st.expander("Ver Máscara de Threshold (Debug)"):
            st.image(relatorio['thresh_mask'], caption="Máscara binária — branco = diferença detectada", use_container_width=True)

        return relatorio
