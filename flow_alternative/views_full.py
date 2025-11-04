import os
import math
import time
import json
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt


# ============================================================
# PARTE 1 - BOX UTILS
# ============================================================

def centers_to_extents(centers):
    """
    Converte caixas do formato (cx, cy, w, h) para (x1, y1, x2, y2)
    """
    cxy, wh = torch.split(centers, 2, dim=-1)
    x1y1 = cxy - wh / 2
    x2y2 = cxy + wh / 2
    return torch.cat([x1y1, x2y2], dim=-1)


def extents_to_centers(extents):
    """
    Converte caixas do formato (x1, y1, x2, y2) para (cx, cy, w, h)
    """
    x1y1, x2y2 = torch.split(extents, 2, dim=-1)
    cxy = (x1y1 + x2y2) / 2
    wh = x2y2 - x1y1
    return torch.cat([cxy, wh], dim=-1)


def normalize_boxes(boxes, boundary):
    """
    Normaliza coordenadas de caixas com base no tamanho do boundary.
    """
    width, height = boundary[1][0] - boundary[0][0], boundary[1][1] - boundary[0][1]
    boxes[:, [0, 2]] /= width
    boxes[:, [1, 3]] /= height
    return boxes


def denormalize_boxes(boxes, boundary):
    """
    Reverte a normalização das coordenadas.
    """
    width, height = boundary[1][0] - boundary[0][0], boundary[1][1] - boundary[0][1]
    boxes[:, [0, 2]] *= width
    boxes[:, [1, 3]] *= height
    return boxes


# ============================================================
# PARTE 2 - FLOORPLAN
# ============================================================

class FloorPlan:
    """
    Representação de uma planta.
    Contém informações geométricas, tipos de cômodos e relações espaciais.
    """

    def __init__(self, data, train=False):
        self.data = data
        self.boundary = np.array(data.boundary)
        self.train = train

        self.box = np.array(data.box) if hasattr(data, 'box') else None
        self.rType = np.array(data.rType) if hasattr(data, 'rType') else None
        self.rEdge = np.array(data.rEdge) if hasattr(data, 'rEdge') else None

    # ----------------------------------------------------------
    def get_test_data(self):
        """
        Prepara tensores para inferência do modelo.
        Retorna:
            boundary, inside_box, rooms, attrs, triples
        """
        boundary = torch.tensor(self.boundary).float()
        boundary = boundary.unsqueeze(0) if boundary.ndim == 2 else boundary

        if self.box is not None:
            inside_box = torch.tensor(self.box[:, :4]).float()
            rooms = torch.tensor(self.rType).long()
        else:
            inside_box = torch.zeros((1, 4))
            rooms = torch.zeros((1,), dtype=torch.long)

        attrs = torch.zeros_like(rooms)
        triples = torch.zeros((1, 3), dtype=torch.long)

        return boundary, inside_box, rooms, attrs, triples

    # ----------------------------------------------------------
    def adapt_graph(self, fp_graph):
        """
        Adapta o grafo de relacionamento da planta base (boundary)
        para o grafo da planta de referência.
        """
        self.data.rType = fp_graph.data.rType
        self.data.rEdge = fp_graph.data.rEdge
        return self

    def adjust_graph(self):
        """
        Ajuste fino da estrutura do grafo — placeholder simplificado.
        """
        if hasattr(self.data, 'rEdge'):
            self.data.rEdge = np.array(self.data.rEdge)
        return self

    def get_rooms(self, tensor=True):
        """
        Retorna tipos de cômodos.
        """
        rooms = self.rType
        if tensor:
            return torch.tensor(rooms).long()
        return rooms

    def get_triples(self, tensor=True):
        """
        Retorna as relações entre os cômodos (grafo).
        """
        triples = self.rEdge
        if tensor:
            return torch.tensor(triples).long()
        return triples

# ============================================================
# PARTE 3 - LAYERS (PyTorch)
# ============================================================

import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Bloco básico de convolução + batchnorm + ReLU
    """
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MLPBlock(nn.Module):
    """
    Multi-Layer Perceptron com ReLU
    """
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# ============================================================
# PARTE 4 - LAYOUT MODULE
# ============================================================

class Layout(nn.Module):
    """
    Módulo que gera mapas de layout (predição da planta).
    """
    def __init__(self, in_ch=3, out_ch=14):
        super().__init__()
        self.conv1 = ConvBlock(in_ch, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 64)
        self.conv5 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.conv5(x)


# ============================================================
# PARTE 5 - MODEL PRINCIPAL
# ============================================================

class Model(nn.Module):
    """
    Modelo principal que gera a planta e prediz bounding boxes.
    """
    def __init__(self):
        super().__init__()
        # Encoder do grafo
        self.room_embed = nn.Embedding(20, 64)
        self.edge_embed = nn.Embedding(10, 64)
        self.attr_embed = nn.Embedding(20, 32)

        # MLPs para processar atributos e relações
        self.mlp_room = MLPBlock(64 + 32, 128, 64)
        self.mlp_edge = MLPBlock(64 * 2 + 64, 128, 64)

        # Decodificador para caixas (x, y, w, h)
        self.fc_box = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4)
        )

        # Módulo de layout
        self.layout_head = Layout()

    def forward(
        self,
        rooms,
        triples,
        boundary,
        obj_to_img=None,
        attributes=None,
        boxes_gt=None,
        generate=True,
        refine=True,
        relative=True,
        inside_box=None
    ):
        # Embed rooms e atributos
        room_feat = self.room_embed(rooms)
        attr_feat = self.attr_embed(attributes)
        x = torch.cat([room_feat, attr_feat], dim=-1)
        x = self.mlp_room(x)

        # Processa arestas (relacionamentos)
        if triples.shape[0] > 0:
            src, rel, dst = triples[:, 0], triples[:, 1], triples[:, 2]
            src_feat = x[src]
            dst_feat = x[dst]
            rel_feat = self.edge_embed(rel)
            edge_input = torch.cat([src_feat, rel_feat, dst_feat], dim=-1)
            edge_feat = self.mlp_edge(edge_input)
        else:
            edge_feat = torch.zeros_like(x)

        # Predição de caixas
        boxes_pred = self.fc_box(x)

        # Layout
        batch_size = boundary.shape[0]
        fake_img = torch.randn(batch_size, 3, 64, 64, device=boundary.device)
        layout_map = self.layout_head(fake_img)

        # Refinamento simples (mock)
        boxes_refine = boxes_pred + torch.randn_like(boxes_pred) * 0.01

        return boxes_pred, layout_map, boxes_refine

# ============================================================
# PARTE 6 - RETRIEVAL
# ============================================================

class DataRetriever:
    """
    Busca plantas semelhantes usando vetores de características (tf/centroid/cluster).
    """
    def __init__(self, tf_train, centroids, clusters):
        self.tf_train = tf_train
        self.centroids = centroids
        self.clusters = clusters

    def retrieve_cluster(self, query, k=10, multi_clusters=False):
        """
        Recupera os índices mais próximos do vetor de consulta.
        """
        if not hasattr(query, "boundary"):
            raise ValueError("Query inválida: objeto sem atributo 'boundary'.")

        # Vetoriza boundary como feature (simplificado)
        query_vec = np.array(query.boundary).flatten()
        if len(query_vec) > self.tf_train.shape[1]:
            query_vec = query_vec[:self.tf_train.shape[1]]
        elif len(query_vec) < self.tf_train.shape[1]:
            query_vec = np.pad(query_vec, (0, self.tf_train.shape[1] - len(query_vec)))

        # Distância euclidiana
        dist = np.linalg.norm(self.tf_train - query_vec, axis=1)
        topk_idx = np.argsort(dist)[:k]
        return topk_idx


# ============================================================
# PARTE 7 - ALIGN
# ============================================================

def align_fp_refine(boundary, refine_boxes, room_types, edges, gene):
    """
    Ajusta a posição e o tamanho das caixas geradas (refine_boxes)
    para que fiquem dentro do contorno (boundary).
    """
    boxes = np.copy(refine_boxes)
    order = np.arange(len(boxes))

    bx_min, by_min = np.min(boundary, axis=0)
    bx_max, by_max = np.max(boundary, axis=0)

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        # Corrige se passar do limite
        x1 = np.clip(x1, bx_min, bx_max)
        x2 = np.clip(x2, bx_min, bx_max)
        y1 = np.clip(y1, by_min, by_max)
        y2 = np.clip(y2, by_min, by_max)
        boxes[i] = [x1, y1, x2, y2]

    room_boundaries = []
    for box in boxes:
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        room_boundaries.append([cx, cy])

    return boxes, order, room_boundaries


# ============================================================
# PARTE 8 - DECORATE
# ============================================================

def add_door_window(data):
    """
    Gera portas e janelas artificiais com base nas posições das caixas.
    """
    doors = []
    windows = []

    for i, box in enumerate(data.newBox):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1

        # Porta: posição média da parede inferior
        door_x1 = (x1 + x2) / 2 - w * 0.1
        door_x2 = (x1 + x2) / 2 + w * 0.1
        door_y = y2
        doors.append([door_x1, door_y, door_x2, door_y])

        # Janela: posição média da parede superior
        window_x1 = (x1 + x2) / 2 - w * 0.15
        window_x2 = (x1 + x2) / 2 + w * 0.15
        window_y = y1
        windows.append([window_x1, window_y, window_x2, window_y])

    return np.array(doors), np.array(windows)

# ============================================================
# PARTE 9 - VIEWS FINAL
# ============================================================

class Views:
    """
    Classe principal que unifica todo o pipeline.
    """

    def __init__(self, model_path, data_train_path, data_test_path, tf_path, centroids_path, clusters_path):
        start = time.process_time()
        print("Inicializando Views...")

        # 1️⃣ Carrega modelo
        self.model = Model()
        self.model.load_state_dict(torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
        print("Modelo carregado.")

        # 2️⃣ Carrega dados de treino e teste
        self.train_data = pickle.load(open(data_train_path, 'rb'))['data']
        self.test_data = pickle.load(open(data_test_path, 'rb'))['data']
        print("Dados carregados.")

        # 3️⃣ Carrega estruturas de recuperação
        self.tf_train = np.load(tf_path)
        self.centroids = np.load(centroids_path)
        self.clusters = np.load(clusters_path)
        self.retriever = DataRetriever(self.tf_train, self.centroids, self.clusters)

        # 4️⃣ Inicializa listas
        self.testNameList = list(self.test_data['testNameList'])
        self.trainNameList = list(self.train_data['nameList'])
        self.train_data_rNum = np.load('./static/Data/rNum_train.npy') if os.path.exists('./static/Data/rNum_train.npy') else None

        end = time.process_time()
        print(f"Inicialização concluída em {end - start:.2f}s")

    # ----------------------------------------------------------
    def LoadTestBoundary(self, testName) -> dict:
        """
        Carrega as coordenadas de boundary de um exemplo de teste.
        """
        start = time.process_time()

        test_index = self.testNameList.index(testName)
        data = self.test_data[test_index]

        data_dict = {}

        data_dict["door"] = f"{data.boundary[0][0]},{data.boundary[0][1]},{data.boundary[1][0]},{data.boundary[1][1]}"

        exterior = " ".join([f"{p[0]},{p[1]}" for p in data.boundary])
        data_dict["exterior"] = exterior

        end = time.process_time()
        print(f"LoadTestBoundary time: {end - start:.3f}s")

        return data_dict

    # ----------------------------------------------------------
    def GenerateLayout(self, testName):
        """
        Executa o pipeline completo de geração da planta.
        """
        start = time.process_time()
        print(f"Gerando layout para {testName}...")

        test_index = self.testNameList.index(testName)
        data_boundary = self.test_data[test_index]

        # 🔍 Retrieval
        top_idx = self.retriever.retrieve_cluster(data_boundary, k=1)
        data_graph = self.train_data[top_idx[0]]

        # 🔄 Transfer
        fp_boundary = FloorPlan(data_boundary)
        fp_graph = FloorPlan(data_graph)
        fp_transfer = fp_boundary.adapt_graph(fp_graph)
        fp_transfer.adjust_graph()

        # 🧠 Forward
        with torch.no_grad():
            boundary, inside_box, rooms, attrs, triples = fp_transfer.get_test_data()
            boundary = boundary.to(self.model.device)
            inside_box = inside_box.to(self.model.device)
            rooms = rooms.to(self.model.device)
            attrs = attrs.to(self.model.device)
            triples = triples.to(self.model.device)

            boxes_pred, gene_layout, boxes_refine = self.model(
                rooms, triples, boundary, attributes=attrs, inside_box=inside_box
            )

        boxes_pred = centers_to_extents(boxes_pred.cpu()) * 255
        boxes_refine = centers_to_extents(boxes_refine.cpu()) * 255
        gene_layout = gene_layout.cpu()

        # 📐 Align
        boxes_aligned, order, room_boundaries = align_fp_refine(
            data_boundary.boundary, boxes_refine.numpy().astype(int),
            getattr(data_boundary, 'rType', np.zeros((len(boxes_refine),))),
            getattr(data_boundary, 'rEdge', np.zeros((len(boxes_refine), 3))),
            None
        )

        data_boundary.newBox = boxes_aligned
        data_boundary.order = order
        data_boundary.rBoundary = room_boundaries

        # 🚪 Decorate
        doors, windows = add_door_window(data_boundary)
        data_boundary.doors = doors
        data_boundary.windows = windows

        end = time.process_time()
        print(f"GenerateLayout concluído em {end - start:.2f}s")

        return data_boundary

    # ----------------------------------------------------------
    def DrawSVG(self, data, output_dir="./output", filename="generated_layout"):
        """
        Renderiza o SVG da planta com cômodos, portas e janelas.
        """
        import svgwrite

        os.makedirs(output_dir, exist_ok=True)
        svg_path = os.path.join(output_dir, f"{filename}.svg")
        dwg = svgwrite.Drawing(svg_path, profile='tiny')

        # 🧱 Exterior boundary
        ext_points = [tuple(map(float, p)) for p in data.boundary]
        dwg.add(dwg.polygon(points=ext_points, fill="none", stroke="#000", stroke_width=3))

        # 📦 Cômodos
        for i, box in enumerate(data.newBox):
            color = f"hsl({(i * 40) % 360}, 70%, 70%)"
            dwg.add(
                dwg.rect(
                    insert=(box[0], box[1]),
                    size=(box[2]-box[0], box[3]-box[1]),
                    fill=color,
                    stroke="#333",
                    stroke_width=1
                )
            )

        # 🚪 Portas
        for door in data.doors:
            dwg.add(
                dwg.line(start=(door[0], door[1]), end=(door[2], door[3]), stroke="#ff6600", stroke_width=3)
            )

        # 🪟 Janelas
        for window in data.windows:
            dwg.add(
                dwg.line(start=(window[0], window[1]), end=(window[2], window[3]), stroke="#00bfff", stroke_width=2)
            )

        dwg.add(dwg.text("Generated Floorplan", insert=(10, 20), fill="gray"))
        dwg.save()
        print(f"SVG salvo em: {svg_path}")
        return svg_path

    # ----------------------------------------------------------
    def NumSearch(self, data_new):
        """
        Busca layouts similares baseando-se em múltiplos critérios.
        """
        start = time.process_time()

        testName = data_new[0].split(".")[0]
        test_index = self.testNameList.index(testName)
        data = self.test_data[test_index]

        test_data_topk = self.retriever.retrieve_cluster(data, k=100)

        topkList = []
        for idx in test_data_topk[:20]:
            topkList.append(f"{self.trainNameList[idx]}.png")

        end = time.process_time()
        print(f"NumSearch time: {end - start:.3f}s")
        return topkList


if __name__ == "__main__":
    views = Views(
        model_path="./model/model.pth",
        data_train_path="./static/Data/data_train_converted.pkl",
        data_test_path="./static/Data/data_test_converted.pkl",
        tf_path="./retrieval/tf_train.npy",
        centroids_path="./retrieval/centroids_train.npy",
        clusters_path="./retrieval/clusters_train.npy"
    )

    test_name = "75119"  # exemplo existente no dataset
    data_boundary = views.GenerateLayout(test_name)
    svg_path = views.DrawSVG(data_boundary, filename=test_name)
    print("SVG gerado:", svg_path)
