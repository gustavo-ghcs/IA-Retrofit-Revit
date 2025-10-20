# IA-Retrofit-Revit: Geração Inteligente de Layouts para Retrofit com Integração BIM

![Status](https://img.shields.io/badge/Status-Em_Desenvolvimento-yellow)
![TCC](https://img.shields.io/badge/TCC-Sistemas_de_Informa%C3%A7%C3%A3o_UFPE-blue)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange?logo=pytorch)
![Revit](https://img.shields.io/badge/Autodesk_Revit-2023%2B-red?logo=autodesk)

Repositório oficial do Trabalho de Conclusão de Curso (TCC) de **Gustavo de Hollanda Cavalcanti Soares**, para o curso de Sistemas de Informação do Centro de Informática (CIn) da Universidade Federal de Pernambuco (UFPE).

* **Orientadora:** Prof(a). Rachel Perez Palha

---

## 🎯 Objetivo Principal

Este projeto visa desenvolver um sistema que utiliza Inteligência Artificial para gerar sugestões de layouts internos (divisões de cômodos) para projetos de arquitetura, com foco em **retrofit**. A solução busca criar uma ponte entre modelos de IA generativa (como o House-GAN++) e o software BIM (Building Information Modeling) **Autodesk Revit**.

O objetivo é criar uma ferramenta que transforme o processo de design em uma colaboração dinâmica: o arquiteto fornece um espaço existente, a IA propõe soluções de layout, e o arquiteto seleciona e refina a opção desejada, que é automaticamente convertida de volta para o ambiente BIM.

## 🚀 Fluxo de Trabalho Proposto

1.  **Entrada (Revit):** O usuário modela o "casco" do ambiente a ser reformado (apenas as paredes externas) no Revit e exporta suas informações geométricas.
2.  **Processamento (IA):** Um script converte a geometria do Revit para o formato de entrada do modelo de IA (ex: um arquivo `.json` representando o grafo do layout). O modelo de IA ([House-GAN++](https://github.com/ennauata/houseganpp)) processa essa entrada e gera múltiplas opções de plantas baixas, com divisões de cômodos (quartos, banheiros, cozinha, etc.).
3.  **Interação (GUI):** Uma interface gráfica exibe os layouts gerados, permitindo que o usuário visualize e selecione a opção que mais lhe agrada.
4.  **Saída (Revit):** Após a seleção, o sistema converte o layout escolhido de volta para um formato compatível com o Revit (seja um script Dynamo, uma macro, ou um arquivo `.dxf` para sobreposição), automatizando a criação das paredes internas e outros elementos no modelo BIM.

## 🛠️ Estrutura do Repositório
