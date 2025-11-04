import time
import pickle
import matlab.engine
import numpy as np
import torch
import svgwrite
import os

from model import Model
from floorplan import FloorPlan
from box_utils import centers_to_extents

class Views:
    boxes_pred = None
    centroids = None
    clusters = None
    engview = None
    indxlist = None
    model = None
    relbox = None
    reledge = None
    test_data = None
    test_data_topk = None
    testNameList = None
    tf_train = None
    train_data = None
    train_data_eNum = None
    train_data_rNum = None
    trainNameList = None
    trainTF = None

    # def home(self, request):
    #     return render(request, "home.html", )

    def __init__(self, request_start):
        start = time.process_time()

        self.init_get_test_data()
        self.init_get_train_data()
        self.init_load_matlab_engine()
        self.init_load_model()
        self.init_load_retrieval()

        end = time.process_time()

        print('Init(model+test+train+engine+retrieval) time: %s Seconds' % (end - start))
    
    def init_get_test_data(self):
        start = time.process_time()
    
        self.test_data = pickle.load(open('./static/Data/data_test_converted.pkl', 'rb'))
        self.test_data = self.test_data['data']
        self.testNameList = list(self.test_data['testNameList'])
        self.trainNameList = list(self.test_data['trainNameList'])

        end = time.process_time()
        print('get_test_data time: %s Seconds' % (end - start))

    def init_get_train_data(self):
        start = time.process_time()
        
        self.train_data = pickle.load(open('./static/Data/data_train_converted.pkl', 'rb'))
        self.train_data = self.train_data['data']
        self.trainNameList = list(self.train_data['nameList'])
        self.trainTF = list(self.train_data['trainTF'])
        
        train_data_eNum = pickle.load(open('./static/Data/data_train_eNum.pkl', 'rb'))
        train_data_eNum = train_data_eNum['eNum']
        self.train_data_rNum = np.load('./static/Data/rNum_train.npy')

        end = time.process_time()
        print('getTrainData time: %s Seconds' % (end - start))


    def init_load_matlab_engine(self):
        startengview = time.process_time()
        
        self.engview = matlab.engine.start_matlab()
        self.engview.addpath(r'./align_fp/', nargout=0)
        
        endengview = time.process_time()
        
        print(' matlab.engineview time: %s Seconds' % (endengview - startengview))

    def init_load_model(self):
        start = time.process_time()

        self.model = Model()
        self.model.cuda(0)
        self.model.load_state_dict(torch.load('./model/model.pth', map_location={'cuda:0': 'cuda:0'}))
        self.model.eval()
        
        end = time.process_time()        
        print('loadModel time: %s Seconds' % (end - start))

        start = time.process_time()
        
        test = self.train_data[self.trainNameList.index("75119")]        
        self._test(self.model, FloorPlan(test, train=True))
        
        end = time.process_time()        
        print('test Model time: %s Seconds' % (end - start))

    def init_load_retrieval(self):
        t1 = time.process_time()
        
        self.tf_train = np.load('./retrieval/tf_train.npy')
        self.centroids = np.load('./retrieval/centroids_train.npy')
        self.clusters = np.load('./retrieval/clusters_train.npy')
        
        t2 = time.process_time()

        print('load tf/centroids/clusters', t2 - t1)
    
    def init_test(self, model,fp):
        with torch.no_grad():
            batch = self._get_data(fp)
            boundary,inside_box,rooms,attrs,triples = batch
            model_out = model(
                rooms, 
                triples, 
                boundary,
                obj_to_img = None,
                attributes = attrs,
                boxes_gt= None, 
                generate = True,
                refine = True,
                relative = True,
                inside_box=inside_box
            )
            boxes_pred,  gene_layout, boxes_refine= model_out
            boxes_pred = boxes_pred.detach()
            boxes_pred = centers_to_extents(boxes_pred)
            boxes_refine = boxes_refine.detach()
            boxes_refine = centers_to_extents(boxes_refine)
            gene_layout = gene_layout*boundary[:,:1]
            gene_preds = torch.argmax(gene_layout.softmax(1).detach(),dim=1)

            return boxes_pred.squeeze().cpu().numpy(),gene_preds.squeeze().cpu().double().numpy(),boxes_refine.squeeze().cpu().numpy()

    def init_get_data(self, fp):
        batch = list(fp.get_test_data())
        batch[0] = batch[0].unsqueeze(0).cuda()
        batch[1] = batch[1].cuda()
        batch[2] = batch[2].cuda()
        batch[3] = batch[3].cuda()
        batch[4] = batch[4].cuda()

        return batch
    
    def LoadTestBoundary(self, testName) -> dict:
        start = time.process_time()
        
        test_index = self.testNameList.index(testName)
        data = self.test_data[test_index]

        data_dict = {}

        data_dict["door"] = str(data.boundary[0][0]) + "," + str(data.boundary[0][1]) + "," + str(data.boundary[1][0]) + "," + str(data.boundary[1][1])
        
        exterior = ""
        
        for i in range(len(data.boundary)):
            exterior = exterior + str(data.boundary[i][0]) + "," + str(data.boundary[i][1]) + " "
        
        data_dict['exterior'] = exterior
        
        end = time.process_time()

        print('LoadTestBoundary time: %s Seconds' % (end - start))
        
        return data_dict

    
    def DrawSVG(self, boundary_data: dict, output_dir="./output", filename="boundary"):
        """
        Gera um SVG contendo o polígono exterior e a porta principal,
        baseado nos dados retornados por LoadTestBoundary().
        """

        filePath = f"{filename}.svg"

        # Cria diretório de saída se não existir
        os.makedirs(output_dir, exist_ok=True)

        # Define caminho completo
        svg_path = os.path.join(output_dir, filePath)

        # Cria o objeto SVG
        dwg = svgwrite.Drawing(svg_path, profile='tiny')

        # Estilos equivalentes aos usados no D3.js
        border_width = 4
        exterior_color = "#000000"   # roomcolor("Exterior wall")
        door_color = "#ff6600"       # roomcolor("Front door")

        # 1️⃣ Desenha o polígono exterior
        exterior_points = [
            tuple(map(float, p.split(',')))
            for p in boundary_data["exterior"].strip().split()
        ]
        dwg.add(
            dwg.polygon(
                points=exterior_points,
                fill="none",
                stroke=exterior_color,
                stroke_width=border_width
            )
        )

        # 2️⃣ Desenha a porta
        door_coords = list(map(float, boundary_data["door"].split(",")))
        dwg.add(
            dwg.line(
                start=(door_coords[0], door_coords[1]),
                end=(door_coords[2], door_coords[3]),
                stroke=door_color,
                stroke_width=border_width
            )
        )

        # 3️⃣ Adiciona opcionalmente um texto identificando
        dwg.add(dwg.text("Boundary Layout", insert=(10, 20), fill="gray"))

        # 4️⃣ Salva o arquivo
        dwg.save()

        print(f"SVG salvo em: {svg_path}")

        # Retorna o caminho do arquivo SVG
        return svg_path
        

    def NumSearch(self, data_new):
        start = time.process_time()
        
        test_index = self.testNameList.index(fileName)
        topkList = []
        topkList.clear()
        data = self.test_data[test_index]
    
        multi_clusters=False
        test_data_topk = retrieval(data, 1000,multi_clusters)
        
        if len(data_new) > 1:
            roomactarr = data_new[1]   # Ativos (ex: LivingRoom, Bath)
            roomexaarr = data_new[2]   # Exatos (ex: Balcony, Storage)
            roomnumarr = [int(x) for x in data_new[3]]  # Quantidades
            
            test_num = train_data_rNum[test_data_topk]
            filter_func = get_filter_func(roomactarr, roomexaarr, roomnumarr)
            indices = np.where(list(map(filter_func, test_num)))
            indices = list(indices)
            if len(indices[0]) < 20:
                topk = len(indices[0])
            else:
                topk = 20
            topkList.clear()
            for i in range(topk):
                topkList.append(str(trainNameList[int(test_data_topk[indices[0][i]])]) + ".png")

        end = time.process_time()

        print('NumberSearch time: %s Seconds' % (end - start))

        return topkList

    def retrieval(test_data,k,multi_clusters):

        retriever = DataRetriever(vw.tf_train,vw.centroids,vw.clusters)
        datum = test_data
        # vis_boundary(datum.boundary)

        t1 = time.process_time()
        index = retriever.retrieve_cluster(datum,k,multi_clusters)
        t2 = time.process_time()
        print('cluster',t2-t1)
        data_retrieval = vw.train_data[index]
        # data_retrieval= trainNameList[index]
        # vis_boundary(data_retrieval[0].boundary)

        # t1 = time()
        # index = retriever.retrieve_bf(datum,k=10)
        # t2 = time()
        # print('bf',t2-t1)
        # data_retrieval = train_data[index]
        return index
    


if __name__ == "main.py":
    views = Views(1)

    filePath = "543.png"
    fileName = int(filePath.split('.')[0])

    boundary_data: dict = views.LoadTestBoundary(filePath)
        
    views.DrawSVG(boundary_data, fileName)
    
    new_data = [filePath]
    topk_list = views.NumSearch(new_data)

    print(topk_list)