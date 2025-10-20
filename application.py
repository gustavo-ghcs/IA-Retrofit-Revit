from flask import Flask, jsonify, render_template, request, Response
# Garanta que a função run_model está no local correto
from python._infer import run_model 
import time
import json
from flask_cors import CORS
from waitress import serve

application = app = Flask(__name__)
# O CORS é importante para permitir a comunicação entre o navegador e o servidor local
CORS(app)

@app.route('/')
def home():
    print("✅ [INFO] Rota '/' acessada. Servindo a página principal (index.html).")
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    print("\n🚀 [ROTA] Rota '/generate' foi chamada via POST!")
    
    try:
        # Recebe e decodifica os dados enviados pelo navegador
        graph_str = request.data.decode('utf-8')
        print(f"   [DADO] Recebido do navegador (primeiros 100 caracteres): {graph_str[:100]}...")
        
        graph_data = json.loads(graph_str)
        print("   [INFO] Dados JSON decodificados com sucesso.")
        
        print("\n⏳ [PROCESSANDO] Chamando a função 'run_model'. Isso pode demorar...")
        start_time = time.time()
        
        # --- PONTO CRÍTICO ---
        result = run_model(graph_data)
        # ---------------------
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"🎉 [SUCESSO] 'run_model' concluída em {duration:.2f} segundos!")
        
        return Response(result, mimetype='text/plain')

    except Exception as e:
        print(f"❌ [ERRO] Ocorreu um erro na rota '/generate': {e}")
        return Response(f"Erro no servidor: {e}", status=500)

# Bloco para iniciar o servidor
if __name__ == '__main__':
    print("🔥 [SERVIDOR] Iniciando o servidor Waitress...")
    print("   Acesse a interface em http://127.0.0.1:5000")
    # Usando o servidor Waitress, como o código original pedia
    serve(app, host='127.0.0.1', port=5000)