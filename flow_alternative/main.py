from views import Views

views = Views(1)

response_load_test_boundary: dict = views.LoadTestBoundary("543.png")


# response_load_test_boundary
# {
#     "door": "188,112,188,124",
#     "exterior": "188,112 188,124 188,184 69,184 69,74 188,74 "
# }





# /house/
# │
# ├── urls.py
# │     ├── /index/Init → views.Init
# │     ├── /index/NumSearch → views.NumSearch
# │     ├── /index/LoadTrainHouse → views.LoadTrainHouse
# │     ├── /index/TransGraph → views.TransGraph
# │     ├── /index/AdjustGraph → views.AdjustGraph
# │     ├── /index/GraphSearch → views.GraphSearch
# │     ├── /index/RelBox → views.RelBox
# │     ├── /index/Save_Editbox → views.Save_Editbox
# │     └── /home → views.home
# │
# └── views.py
#       ├── Init()                # Carrega dados, modelo e engine MATLAB
#       ├── getTestData()         # Lê dados de teste (.pkl)
#       ├── getTrainData()        # Lê dados de treino (.pkl, .npy)
#       ├── loadModel()           # Carrega o modelo Python (mltest)
#       ├── loadMatlabEng()       # Inicia engine do MATLAB
#       ├── loadRetrieval()       # Carrega embeddings tf/centroids/clusters
#       │
#       ├── LoadTestBoundary()    # Retorna bordas do layout de teste
#       ├── LoadTrainHouse()      # Retorna planta de treino (usada como referência)
#       │
#       ├── NumSearch()           # Busca plantas semelhantes por número de cômodos
#       ├── GraphSearch()         # Busca plantas semelhantes pela estrutura do grafo
#       │
#       ├── TransGraph()          # Transfere grafo da planta de treino para a de teste
#       ├── TransGraph_net()      # Mesma ideia, mas usa rede neural (mltest.get_userinfo_net)
#       ├── AdjustGraph()         # Ajusta grafo manualmente conforme interação do usuário
#       │
#       ├── RelBox()              # Retorna relações de um cômodo com outros
#       └── Save_Editbox()        # Salva alterações manuais da planta no MATLAB
