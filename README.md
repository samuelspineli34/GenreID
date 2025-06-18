# Sistema de Classificação de Gêneros Musicais e Geração de Acompanhamentos

Este projeto consiste em um sistema para classificar o gênero musical de melodias e gerar acompanhamentos automáticos em formato MIDI. Um modelo **Random Forest** foi empregado para a classificação de gênero, utilizando características extraídas de mel-espectrogramas.

A aplicação é desenvolvida com Flask e permite as seguintes interações:

-   Upload de uma melodia (formatos MIDI, MP3, WAV, etc.) ou gravação em tempo real para análise.
-   Predição do gênero musical da melodia e sugestões de instrumentos para acompanhamento.
-   Geração e download de um arquivo MIDI de acompanhamento, ajustado ao BPM e ao gênero/instrumento selecionados.

## Índice

- [Visão Geral](#visão-geral)
- [Recursos Utilizados](#recursos-utilizados)
- [Instalação](#instalação)
- [Uso](#uso)
- [Estrutura do Projeto](#estrutura-do-projeto)

## Visão Geral

O sistema possui duas funcionalidades principais:

-   **Classificação de Gênero Musical:**
    O endpoint `/predict` recebe uma melodia de entrada. O sistema processa esta melodia para gerar mel-espectrogramas e extrai características como RMS, Zero Crossing Rate (ZCR), centroides espectrais, MFCCs e BPM. O modelo **Random Forest** prediz o gênero musical com base nessas características. O resultado inclui o gênero predito e uma lista de instrumentos sugeridos para o acompanhamento.

-   **Geração de Acompanhamento Musical:**
    O endpoint `/accompaniment` utiliza o gênero musical detectado, o instrumento e o BPM fornecidos pelo usuário. Um arquivo MIDI de acompanhamento pré-definido (baseado em progressões harmônicas do gênero) é ajustado ao BPM informado e retornado para download.

## Recursos Utilizados

-   **Linguagem:** Python
-   **Framework Web:** Flask
-   **Bibliotecas:**
    -   `Pandas`, `NumPy`: Manipulação e processamento de dados.
    -   `Librosa`: Extração de características de áudio (MFCCs, espectrogramas).
    -   `Pydub`: Manipulação e conversão de formatos de áudio.
    -   `Mido`, `midi2audio`: Processamento de arquivos MIDI e conversão para áudio.
    -   `Scikit-learn`: Implementação e treinamento do modelo **Random Forest**.
    -   `Joblib`: Carregamento de modelos, scaler e encoder pré-treinados.
    -   `Mlxtend`: Utilizado para visualizações durante o desenvolvimento.
-   **Ambiente de Execução:** Python 3.x

## Instalação

Para configurar e executar o projeto localmente:

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/seu-usuario/classificador-generos-musicais.git](https://github.com/seu-usuario/classificador-generos-musicais.git)
    cd classificador-generos-musicais
    ```
2.  **Crie e ative um ambiente virtual (recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    venv\Scripts\activate     # Windows
    ```
3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Inicie o servidor Flask:**
    ```bash
    python interface.py
    ```
5.  **Acesse a aplicação:**
    ```
    [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
    ```

## Uso

Instruções para utilizar a aplicação:

1.  **Acesse a Interface Web:**
    Abra seu navegador e vá para `http://localhost:5000/`.

2.  **Envie sua Melodia:**
    * Clique em "Escolher arquivo" e selecione uma melodia (MIDI, MP3, WAV, FLAC, M4A, OGG).
    * Clique em "Enviar".

3.  **Visualize Resultados:**
    * A aplicação exibirá o gênero musical previsto e instrumentos sugeridos.

4.  **Gere Acompanhamento (Opcional):**
    * Selecione um instrumento sugerido.
    * Defina o BPM desejado.
    * Clique em "Gerar Acompanhamento" para baixar o arquivo MIDI personalizado.

## Estrutura do Projeto

A organização do projeto é a seguinte:

* `interface.py`: Implementa a aplicação Flask.
* `uploads/`: Armazena temporariamente os arquivos de melodia enviados.
* `predictions/`: Armazena os arquivos MIDI de acompanhamento gerados.
* `accompaniments/`: Contém arquivos MIDI pré-definidos para cada combinação de gênero e instrumento.
* `modelos/`: Contém o classificador de gênero (`genre_classifier.pkl`).
* `soundfont/`: Armazena a soundfont para conversão de MIDI para áudio.
* `templates/`: Contém o arquivo `index.html` da interface web.
