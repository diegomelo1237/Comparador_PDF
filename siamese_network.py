import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import streamlit as st # Importado para usar st.info/warning se necessário

class SiameseNetwork(nn.Module):
    """
    Rede Siamese usando ResNet-18 pré-treinada como backbone.
    Extrai embeddings das duas imagens e compara as diferenças
    pixel a pixel entre os mapas de features.
    """

    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Carrega ResNet-18 pré-treinada no ImageNet
        # Usar 'DEFAULT' para garantir a versão mais recente dos pesos
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Remove as últimas camadas (avgpool + fc) — queremos os feature maps
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])

        # Congela os pesos — não vamos treinar, só usar para extrair features
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.feature_extractor.eval() # Coloca o modelo em modo de avaliação

    def extrair_features(self, img_tensor):
        with torch.no_grad(): # Desativa o cálculo de gradientes para inferência
            features = self.feature_extractor(img_tensor)
        return features

    def forward(self, img1_tensor, img2_tensor):
        feat1 = self.extrair_features(img1_tensor)
        feat2 = self.extrair_features(img2_tensor)
        return feat1, feat2

class AnalisadorSiamese:
    """
    Wrapper que integra a SiameseNetwork ao pipeline existente.
    Recebe duas PIL Images e retorna:
      - score de similaridade global (float 0-1)
      - mapa de diferenças (numpy array HxW, uint8)
      - contornos das diferenças
    """

    def __init__(self):
        # Tenta usar CUDA se disponível, senão usa CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modelo = SiameseNetwork().to(self.device)
        self.modelo.eval() # Garante que o modelo está em modo de avaliação

        # Transformações para as imagens: redimensionar, converter para tensor e normalizar
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), # ResNet-18 espera 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _preprocessar_imagem(self, img: Image.Image) -> torch.Tensor:
        """Aplica as transformações necessárias à imagem PIL."""
        return self.transform(img).unsqueeze(0).to(self.device) # Adiciona dimensão de batch

    def _features_para_mapa_diferenca(self, feat1: torch.Tensor, feat2: torch.Tensor, original_size) -> np.ndarray:
        """
        Calcula a diferença absoluta entre os mapas de features e redimensiona
        para o tamanho original da imagem.
        """
        # Calcula a diferença absoluta entre os mapas de features
        diff_tensor = torch.abs(feat1 - feat2)

        # Média sobre os canais para obter um mapa de diferença 2D
        diff_map = torch.mean(diff_tensor, dim=1, keepdim=True) # Mantém a dimensão do canal para interpolação

        # Normaliza para o intervalo [0, 1]
        diff_map = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-6)

        # Redimensiona para o tamanho original da imagem usando interpolação
        # Permuta as dimensões para (H, W, C) antes de converter para numpy
        diff_map_resized = torch.nn.functional.interpolate(
            diff_map,
            size=original_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0).cpu().numpy() # Remove dimensões de batch e canal

        # Converte para uint8 no intervalo [0, 255]
        diff_map_uint8 = (diff_map_resized * 255).astype(np.uint8)
        return diff_map_uint8

    def _calcular_mapas_direcionais(self, feat1: torch.Tensor, feat2: torch.Tensor, original_size) -> tuple[np.ndarray, np.ndarray]:
        """
        Calcula mapas de diferença direcionais: remoção (feat1 > feat2) e adição (feat2 > feat1).
        """
        # Calcula a diferença bruta
        diff_raw = feat1 - feat2

        # Mapa de remoção: onde feat1 tinha mais "atividade" que feat2 (algo sumiu)
        # Valores positivos indicam remoção
        diff_remocao = torch.relu(diff_raw)
        diff_remocao = torch.mean(diff_remocao, dim=1, keepdim=True)
        diff_remocao = (diff_remocao - diff_remocao.min()) / (diff_remocao.max() - diff_remocao.min() + 1e-6)
        diff_remocao_resized = torch.nn.functional.interpolate(
            diff_remocao, size=original_size, mode='bilinear', align_corners=False
        ).squeeze(0).squeeze(0).cpu().numpy()
        diff_remocao_uint8 = (diff_remocao_resized * 255).astype(np.uint8)

        # Mapa de adição: onde feat2 tinha mais "atividade" que feat1 (algo foi adicionado)
        # Valores negativos (abs) indicam adição
        diff_adicao = torch.relu(-diff_raw) # -diff_raw para pegar onde feat2 > feat1
        diff_adicao = torch.mean(diff_adicao, dim=1, keepdim=True)
        diff_adicao = (diff_adicao - diff_adicao.min()) / (diff_adicao.max() - diff_adicao.min() + 1e-6)
        diff_adicao_resized = torch.nn.functional.interpolate(
            diff_adicao, size=original_size, mode='bilinear', align_corners=False
        ).squeeze(0).squeeze(0).cpu().numpy()
        diff_adicao_uint8 = (diff_adicao_resized * 255).astype(np.uint8)

        return diff_remocao_uint8, diff_adicao_uint8


    def comparar(self, img1: Image.Image, img2: Image.Image, sensibilidade: float = 0.3):
        """
        Compara duas imagens usando a Siamese Network.
        Retorna score de similaridade, mapa de diferença, contornos e máscara binária.
        """
        original_size = img1.size[::-1] # (height, width)

        img1_tensor = self._preprocessar_imagem(img1)
        img2_tensor = self._preprocessar_imagem(img2)

        feat1, feat2 = self.modelo(img1_tensor, img2_tensor)

        # Calcula o mapa de diferença simétrico (abs(feat1 - feat2))
        diff_mapa_simetrico = self._features_para_mapa_diferenca(feat1, feat2, original_size)

        # Calcula os mapas de diferença direcionais
        diff_mapa_remocao, diff_mapa_adicao = self._calcular_mapas_direcionais(feat1, feat2, original_size)

        # Calcula a similaridade (pode ser a média inversa da diferença)
        # Quanto menor a diferença, maior a similaridade
        score = 1.0 - (np.mean(diff_mapa_simetrico) / 255.0)

        # Aplica threshold para binarizar a imagem de diferença combinada
        # Usamos o mapa simétrico para a detecção de contornos gerais
        # O valor de threshold é ajustado pela sensibilidade
        # Invertemos a lógica para que sensibilidade alta = mais rigoroso (menos diferenças)
        valor_thresh = int(255 * (1 - sensibilidade)) # Sensibilidade 0.3 -> valor_thresh 178
                                                    # Sensibilidade 0.8 -> valor_thresh 51
        _, thresh_combinado = cv2.threshold(diff_mapa_simetrico, valor_thresh, 255, cv2.THRESH_BINARY)

        # Encontra contornos na máscara de diferença combinada
        contours, _ = cv2.findContours(thresh_combinado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return score, diff_mapa_simetrico, diff_mapa_remocao, diff_mapa_adicao, contours, thresh_combinado

    def marcar_diferencas(self, img_original: Image.Image, contours, diff_mask: np.ndarray, area_minima: int = 100):
        """
        Marca as diferenças encontradas na imagem manipulada com círculos e números.
        """
        img_marcada = np.array(img_original.convert("RGB")).copy()
        h_img, w_img = img_marcada.shape[:2]
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

    def gerar_heatmap(self, diff_mapa: np.ndarray):
        """Gera heatmap colorido — azul = similar, vermelho = diferente."""
        diff_norm = cv2.normalize(diff_mapa, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(diff_norm.astype("uint8"), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        return Image.fromarray(heatmap_rgb)

    def gerar_overlay(self, img_original: Image.Image, thresh_mask: np.ndarray):
        """Gera overlay vermelho semitransparente sobre as regiões diferentes."""
        base = np.array(img_original.convert("RGB")).copy()
        overlay = base.copy()
        mascara = thresh_mask > 0
        overlay[mascara] = [255, 50, 50] # Vermelho claro
        resultado = cv2.addWeighted(base, 0.6, overlay, 0.4, 0)
        return Image.fromarray(resultado)
