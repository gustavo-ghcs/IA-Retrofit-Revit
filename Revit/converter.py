import json
import os
from PIL import Image, ImageDraw

# --- Configurações ---

# Pega o caminho do Desktop do usuário automaticamente
try:
    desktop_path = os.path.join(os.environ['USERPROFILE'], 'Desktop')
except KeyError:
    print("Erro: Não foi possível encontrar o Desktop. Definindo o caminho manualmente.")
    # Se der erro, descomente a linha abaixo e ajuste manualmente:
    # desktop_path = "C:\\Users\\SeuUsuario\\Desktop"
    
# 1. ARQUIVO DE ENTRADA (O JSON que o pyRevit acabou de criar)
ARQUIVO_JSON = os.path.join(desktop_path, "revit.json")

# 2. ARQUIVO DE SAÍDA (A imagem PNG que o Graph2plan usará)
ARQUIVO_SAIDA = os.path.join(desktop_path, "revit_boundary.png")

ESCALA = 50         # Pixels por metro. Aumente se a imagem ficar pequena.
PADDING = 40        # Margem branca em pixels ao redor do desenho.
COR_FUNDO = 'white' # Fundo branco
COR_BORDA = 'black' # Borda preta
ESPESSURA_BORDA = 5 # Espessura da borda em pixels.

# --- 1. Carregar o JSON ---
print(f"Lendo o arquivo JSON de: {ARQUIVO_JSON}")
try:
    with open(ARQUIVO_JSON, 'r', encoding='utf-8') as f:
        dados_ambientes = json.load(f)
except FileNotFoundError:
    print(f"ERRO: Arquivo 'revit.json' não encontrado no seu Desktop.")
    print("Por favor, execute o script do pyRevit no Revit primeiro.")
    exit()
except Exception as e:
    print(f"Erro ao ler o JSON: {e}")
    exit()

if not dados_ambientes:
    print("ERRO: O JSON está vazio. O script do Revit não encontrou ambientes.")
    exit()
print(f"Encontrados {len(dados_ambientes)} ambientes no JSON.")

# --- 2. Achar limites para o tamanho da imagem ---
min_x, max_x = float('inf'), float('-inf')
min_y, max_y = float('inf'), float('-inf')

for amb in dados_ambientes:
    for ponto in amb.get('pontos_limite_m', []):
        min_x = min(min_x, ponto['x'])
        max_x = max(max_x, ponto['x'])
        min_y = min(min_y, ponto['y'])
        max_y = max(max_y, ponto['y'])

largura_img = int((max_x - min_x) * ESCALA) + (PADDING * 2)
altura_img = int((max_y - min_y) * ESCALA) + (PADDING * 2)

print(f"Calculando tamanho da imagem... Dimensões: {largura_img}x{altura_img} pixels.")

# --- 3. Criar a Imagem e Desenhar ---
img = Image.new('RGB', (largura_img, altura_img), COR_FUNDO)
draw = ImageDraw.Draw(img)

# Itera sobre cada ambiente no JSON
for amb in dados_ambientes:
    pontos_poligono = amb.get('pontos_limite_m', [])
    if not pontos_poligono:
        continue # Pula se o ambiente não tiver geometria

    # Converte os pontos de metros (Revit) para pixels (Imagem)
    pontos_traduzidos = []
    for ponto in pontos_poligono:
        # (ponto['x'] - min_x) -> Alinha o desenho no canto esquerdo
        # * ESCALA            -> Aplica o zoom
        # + PADDING           -> Adiciona a margem
        tx = int((ponto['x'] - min_x) * ESCALA) + PADDING
        
        # Inverte o eixo Y (Pillow tem 0 no topo, Revit/CAD tem 0 na base)
        ty = altura_img - (int((ponto['y'] - min_y) * ESCALA) + PADDING)
        
        pontos_traduzidos.append((tx, ty))

    # Desenha a borda do polígono
    if len(pontos_traduzidos) > 1:
        print(f"Desenhando o ambiente: {amb.get('nome', 'Ambiente sem nome')}")
        # Adiciona o primeiro ponto ao final para fechar o polígono
        draw.line(pontos_traduzidos + [pontos_traduzidos[0]], 
                  fill=COR_BORDA, 
                  width=ESPESSURA_BORDA,
                  joint='curve') # 'joint' deixa os cantos mais suaves

# --- 4. Salvar a Imagem ---
img.save(ARQUIVO_SAIDA)

print(f"\nSucesso! Imagem de borda salva em: {ARQUIVO_SAIDA}")