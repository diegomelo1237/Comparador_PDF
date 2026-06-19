import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import streamlit as st


class AnalisadorVisual:
    def __init__(self, img_esquerda, img_direita, rotacionar_180=False):
        self.img_esquerda = img_esquerda.convert('RGB').copy()
        self.img_direita = self._orientar_verticalmente(img_direita.convert('RGB'), rotacionar_180)
        self.normalizar_tamanhos()

    def _orientar_verticalmente(self, img, rotacionar_180=False):
        img_array = np.array(img)
        altura, largura = img_array.shape[:2]

        # 1. Se a imagem estiver deitada (paisagem), rotaciona para ficar em pé (retrato)
        if largura > altura:
            img_array = cv2.rotate(img_array, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # 2. CORREÇÃO OPCIONAL de ponta-cabeça — só aplica se o usuário ativar
        if rotacionar_180:
            img_array = cv2.rotate(img_array, cv2.ROTATE_180)

        return Image.fromarray(img_array)

    def normalizar_tamanhos(self):
        esq = np.array(self.img_esquerda)
        dir_ = np.array(self.img_direita) # 'dir_' para evitar conflito com função built-in
        altura, largura = esq.shape[:2]
        if dir_.shape[:2] != (altura, largura):
            dir_ = cv2.resize(dir_, (largura, altura), interpolation=cv2.INTER_AREA)
            self.img_direita = Image.fromarray(dir_)

    def detectar_diferencas_visuais(self):
        img1 = cv2.cvtColor(np.array(self.img_esquerda), cv2.COLOR_RGB2GRAY)
        img2 = cv2.cvtColor(np.array(self.img_direita), cv2.COLOR_RGB2GRAY)

        # Garante que win_size seja ímpar, >= 3 e menor que as dimensões da imagem
        min_dim = min(img1.shape[:2])
        win_size = min(7, min_dim - 1)
        if win_size % 2 == 0:
            win_size -= 1
        win_size = max(win_size, 3)

        score, diff = ssim(img1, img2, full=True, win_size=win_size)
        diff = (diff * 255).astype("uint8")

        # Inverte o diff: áreas de diferença ficam brancas (255), fundo preto (0)
        diff_inv = cv2.bitwise_not(diff)

        # Threshold com Otsu para separar diferenças do fundo
        _, thresh = cv2.threshold(diff_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Dilatação leve para agrupar diferenças próximas
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return score, contours, diff

    def extrair_elementos_graficos(self, img): # Recebe img como parâmetro
        img_array = np.array(img)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        elementos = [c for c in contours if cv2.contourArea(c) > 100]
        return len(elementos), elementos

    def comparar_elementos_graficos(self):
        elementos_esq, _ = self.extrair_elementos_graficos(self.img_esquerda)
        elementos_dir, _ = self.extrair_elementos_graficos(self.img_direita)
        diferenca = elementos_esq - elementos_dir
        return elementos_esq, elementos_dir, diferenca

    def gerar_relatorio_visual(self):
        score, contours, diff = self.detectar_diferencas_visuais()
        elem_esq, elem_dir, diferenca_elem = self.comparar_elementos_graficos()

        # Filtra ruídos pequenos da contagem oficial de diferenças
        contornos_reais = [c for c in contours if cv2.contourArea(c) > 50]

        return {
            'similaridade': score,
            'contornos_encontrados': len(contornos_reais),
            'elementos_esquerda': elem_esq,
            'elementos_direita': elem_dir,
            'diferenca_elementos': diferenca_elem,
            'diff_img': diff  # Útil para debug visual
        }

    def marcar_diferencas(self):
        img_marcada = np.array(self.img_direita).copy()
        score, contours, diff = self.detectar_diferencas_visuais()
        for contour in contours:
            if cv2.contourArea(contour) > 50:
                x, y, w, h = cv2.boundingRect(contour)
                # Desenha o retângulo delimitando o erro em vermelho (RGB: 255, 0, 0)
                cv2.rectangle(img_marcada, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return Image.fromarray(img_marcada), diff # Retorna diff também para o heatmap

    def gerar_heatmap_diferencas(self, diff):
        """Gera um heatmap colorido das diferenças para melhor visualização."""
        diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        return Image.fromarray(heatmap_rgb)


class IntegracaoStreamlit:
    def __init__(self, pdf_esquerda, pdf_direita, rotacionar_180=False, dpi_opcao=150):
        self.pdf_esquerda = pdf_esquerda
        self.pdf_direita = pdf_direita
        self.rotacionar_180 = rotacionar_180
        self.dpi_opcao = dpi_opcao
        self.analisador = None
        self._inicializar_analisador()

    def _inicializar_analisador(self):
        from pdf2image import convert_from_path, convert_from_bytes
        try:
            # --- PROCESSANDO O PDF DA ESQUERDA ---
            if isinstance(self.pdf_esquerda, str):
                imgs_esq = convert_from_path(self.pdf_esquerda, dpi=self.dpi_opcao)
            else:
                self.pdf_esquerda.seek(0)
                imgs_esq = convert_from_bytes(self.pdf_esquerda.read(), dpi=self.dpi_opcao)

            # --- PROCESSANDO O PDF DA DIREITA ---
            if isinstance(self.pdf_direita, str):
                imgs_dir = convert_from_path(self.pdf_direita, dpi=self.dpi_opcao)
            else:
                self.pdf_direita.seek(0)
                imgs_dir = convert_from_bytes(self.pdf_direita.read(), dpi=self.dpi_opcao)

            if imgs_esq and imgs_dir:
                self.analisador = AnalisadorVisual(imgs_esq[0], imgs_dir[0], self.rotacionar_180)

        except Exception as e:
            st.error(f"Erro ao processar PDFs: {str(e)}")

    def exibir_analise_visual(self):
        if not self.analisador:
            st.warning("Analisador não inicializado.")
            return None

        relatorio = self.analisador.gerar_relatorio_visual()

        # --- MÉTRICAS ---
        st.subheader("Métricas da Análise")
        col1, col2, col3 = st.columns(3)
        with col1:
            similaridade_pct = relatorio['similaridade'] * 100
            cor_delta = "normal" if similaridade_pct >= 95 else "inverse"
            st.metric(
                label="Similaridade (SSIM)",
                value=f"{similaridade_pct:.2f}%",
                delta="Alta" if similaridade_pct >= 95 else "Baixa",
                delta_color=cor_delta
            )
        with col2:
            st.metric(
                label="Diferenças Visuais Detectadas",
                value=relatorio['contornos_encontrados']
            )
        with col3:
            st.metric(
                label="Δ Elementos Gráficos",
                value=relatorio['diferenca_elementos'],
                help="Diferença no número de elementos gráficos detectados (Original - Manipulado)"
            )

        # --- COMPARAÇÃO LADO A LADO ---
        st.subheader("Comparação Visual")
        col_esq, col_dir = st.columns(2)

        img_marcada, diff = self.analisador.marcar_diferencas()

        with col_esq:
            st.image(
                self.analisador.img_esquerda,
                caption="PDF Original",
                use_container_width=True
            )
        with col_dir:
            st.image(
                img_marcada,
                caption="PDF Manipulado (diferenças em vermelho)",
                use_container_width=True
            )

        # --- HEATMAP DE DIFERENÇAS ---
        with st.expander("Ver Heatmap de Diferenças (Debug Visual)"):
            heatmap = self.analisador.gerar_heatmap_diferencas(diff)
            st.image(heatmap, caption="Heatmap — Azul = similar | Vermelho = diferente", use_container_width=True)

        return relatorio
