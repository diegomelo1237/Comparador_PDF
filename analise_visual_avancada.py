# VERSAO_CROP_AUTOMATICO_V8
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import streamlit as st

class AnalisadorVisual:
    def __init__(self, img_esquerda, img_direita):
        # 1. Converte e aplica o recorte automático de bordas brancas/vazias em ambas
        self.img_esquerda = self._recortar_bordas_vazias(img_esquerda.convert('RGB'))
        self.img_direita = self._recortar_bordas_vazias(img_direita.convert('RGB'))
        
        # 2. Equaliza os tamanhos das áreas úteis recortadas
        self.normalizar_tamanhos()

    def _recortar_bordas_vazias(self, img):
        """ Detecta a área útil colorida da embalagem e remove o excesso de fundo branco/claro """
        img_array = np.array(img)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Inverte a imagem: o que é branco vira preto e o que é colorido vira branco
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Encontra os contornos da área colorida (a embalagem verde)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Pega o maior contorno (que obrigatoriamente é o retângulo da embalagem)
            maior_contorno = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(maior_contorno)
            
            # Recorta a imagem exatamente na bordinha da arte verde
            img_recortada = img_array[y:y+h, x:x+w]
            return Image.fromarray(img_recortada)
        
        return img

    def normalizar_tamanhos(self):
        esq = np.array(self.img_esquerda)
        dir = np.array(self.img_direita)
        
        altura_esq, largura_esq = esq.shape[:2]
        altura_dir, largura_dir = dir.shape[:2]
        
        esq_e_paisagem = largura_esq > altura_esq
        dir_e_paisagem = largura_dir > altura_dir
        
        if esq_e_paisagem != dir_e_paisagem:
            dir = cv2.rotate(dir, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
        # Redimensiona a arte útil para bater pixel com pixel perfeitamente
        dir = cv2.resize(dir, (largura_esq, altura_esq), interpolation=cv2.INTER_AREA)
        self.img_direita = Image.fromarray(dir)

    def detectar_diferencas_visuais(self):
        img1 = cv2.cvtColor(np.array(self.img_esquerda), cv2.COLOR_RGB2GRAY)
        img2 = cv2.cvtColor(np.array(self.img_direita), cv2.COLOR_RGB2GRAY)
        
        # Desfoque leve para remover serrilhados e focando nas formas
        img1_blur = cv2.GaussianBlur(img1, (5, 5), 0)
        img2_blur = cv2.GaussianBlur(img2, (5, 5), 0)
        
        # Diferença absoluta direta na arte limpa e sem bordas
        diff = cv2.absdiff(img1_blur, img2_blur)
        
        # Limiar adaptado para capturar alterações severas (como novas tabelas e textos mudados)
        thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)[1]
        
        # Remove poeiras de pixel isoladas
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_clean)
        
        # Dilatação para cercar o bloco completo modificado de uma vez só
        kernel_join = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        thresh = cv2.dilate(thresh, kernel_join, iterations=1)
        
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contornos_filtrados = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # Ignora imperfeições de borda de corte e foca em mudanças de conteúdo
            if w > 25 and h > 25 and cv2.contourArea(c) > 400:
                contornos_filtrados.append(c)
                
        score = ssim(img1, img2, win_size=3)
        return score, contornos_filtrados, thresh

    def extrair_elementos_graficos(self):
        img_array = np.array(self.img_esquerda)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        elementos = [c for c in contours if cv2.contourArea(c) > 100]
        return len(elementos), elementos

    def comparar_elementos_graficos(self):
        elementos_esq, _ = self.extrair_elementos_graficos()
        img_array = np.array(self.img_direita)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        elementos_dir = len([c for c in contours if cv2.contourArea(c) > 100])
        diferenca = elementos_esq - elementos_dir
        return elementos_esq, elementos_dir, diferenca

    def gerar_relatorio_visual(self):
        score, contours, diff = self.detectar_diferencas_visuais()
        elem_esq, elem_dir, diferenca_elem = self.comparar_elementos_graficos()
        
        relatorio = {
            'similaridade': score,
            'contornos_encontrados': len(contours),
            'elementos_esquerda': elem_esq,
            'elementos_direita': elem_dir,
            'diferenca_elementos': diferenca_elem
        }
        return relatorio

    def marcar_diferencas(self):
        img_marcada = np.array(self.img_esquerda).copy()
        score, contours, diff = self.detectar_diferencas_visuais()
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img_marcada, (x, y), (x + w, y + h), (0, 0, 255), 4)
        return Image.fromarray(img_marcada)


class IntegracaoStreamlit:
    def __init__(self, pdf_esquerda, pdf_direita):
        self.pdf_esquerda = pdf_esquerda
        self.pdf_direita = pdf_direita
        self.analisador = None
        self._inicializar_analisador()

    def _inicializar_analisador(self):
        from pdf2image import convert_from_path, convert_from_bytes
        try:
            if isinstance(self.pdf_esquerda, str):
                imgs_esq = convert_from_path(self.pdf_esquerda, dpi=150)
            else:
                self.pdf_esquerda.seek(0)
                imgs_esq = convert_from_bytes(self.pdf_esquerda.read(), dpi=150)

            if isinstance(self.pdf_direita, str):
                imgs_dir = convert_from_path(self.pdf_direita, dpi=150)
            else:
                self.pdf_direita.seek(0)
                imgs_dir = convert_from_bytes(self.pdf_direita.read(), dpi=150)

            if imgs_esq and imgs_dir:
                self.analisador = AnalisadorVisual(imgs_esq[0], imgs_dir[0])
                
        except Exception as e:
            st.error(f"Erro ao processar PDFs: {str(e)}")

    def exibir_analise_visual(self):
        if not self.analisador:
            st.warning("Analisador não inicializado.")
            return None
        
        relatorio = self.analisador.gerar_relatorio_visual()
        
        st.subheader("Métricas da Análise")
        col_metrica1, col_metrica2 = st.columns(2)
        with col_metrica1:
            st.metric(label="Índice de Similaridade", value=f"{relatorio['similaridade']:.2%}")
        with col_metrica2:
            st.metric(label="Alterações Estruturais Detetadas", value=relatorio['contornos_encontrados'])

        st.subheader("Comparação Visual (Área Útil Recortada)")
        col_esq, col_dir = st.columns(2)
        
        with col_esq:
            img_marcada = self.analisador.marcar_diferencas()
            st.image(img_marcada, caption="PDF 1 - Esquerda (Erros Circulados na Arte)", use_container_width=True)
            
        with col_dir:
            st.image(self.analisador.img_direita, caption="PDF 2 - Direita (Arte Alinhada)", use_container_width=True)
            
        return relatorio