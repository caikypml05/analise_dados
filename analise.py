"""
Análise de Dados Mensais

Este script realiza a análise de dados mensais de desempenho, incluindo o cálculo de métricas estatísticas, treinamento de modelos de regressão linear e geração de gráficos para visualização dos resultados. Os dados são lidos a partir de arquivos CSV em um diretório específico e os resultados da análise são registrados em arquivos de log.

Autor: Caiky Palleta
Data: 2024

"""


import os  # Importação do módulo os para lidar com operações do sistema operacional
import pandas as pd  # Importação do módulo pandas para manipulação de dados tabulares
import matplotlib.pyplot as plt  # Importação do módulo matplotlib para visualização de dados
from sklearn.linear_model import LinearRegression  # Importação do modelo de regressão linear do scikit-learn
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error  # Importação de métricas de avaliação do modelo
import numpy as np  # Importação do módulo numpy para operações numéricas eficientes
import statsmodels.api as sm  # Importação do módulo statsmodels para análise estatística
import logging  # Importação do módulo logging para registro de eventos
from datetime import datetime  # Importação da classe datetime para manipulação de datas e horas

def read_csv_files(directory):
    """Função para ler os arquivos CSV no diretório."""
    data = {}  # Dicionário para armazenar os dados lidos dos arquivos CSV
    # Lista de nomes de arquivos no diretório com extensão .csv
    filenames = [filename for filename in os.listdir(directory) if filename.endswith(".csv")]
    # Ordenação dos nomes de arquivos com base no mês (primeira parte do nome do arquivo)
    filenames.sort(key=lambda x: pd.to_datetime(x.split('_')[0], format='%b'))
    for filename in filenames:  # Iteração sobre os nomes dos arquivos
        month = filename.split('_')[0]  # Extrai o mês do nome do arquivo
        df = pd.read_csv(os.path.join(directory, filename))  # Leitura do arquivo CSV
        data[month] = df  # Armazena o DataFrame correspondente ao mês no dicionário
    return data  # Retorna o dicionário de dados

def process_data(data):
    """Função para processar os dados."""
    average_data = {}  # Dicionário para armazenar médias de dados
    throughput_data = {}  # Dicionário para armazenar valores de throughput
    prev_avg = None  # Variável para armazenar a média anterior
    prev_throughput = None  # Variável para armazenar o throughput anterior
    for month, df in data.items():  # Iteração sobre os meses e os DataFrames correspondentes
        average = df['Average'].mean()  # Calcula a média dos valores da coluna 'Average'
        throughput = df['Throughput'].max()  # Encontra o valor máximo da coluna 'Throughput'
        average_data[month] = average  # Armazena a média no dicionário de médias
        throughput_data[month] = throughput  # Armazena o throughput no dicionário de throughput
        # Verifica a tendência de melhoria ou piora em relação ao mês anterior
        if prev_avg is not None and prev_throughput is not None:
            avg_trend = "melhorando" if average < prev_avg else "piorando"
            throughput_trend = "melhorando" if throughput > prev_throughput else "piorando"
        else:
            avg_trend = "indeterminada"
            throughput_trend = "indeterminada"
        prev_avg = average  # Atualiza a média anterior para o próximo mês
        prev_throughput = throughput  # Atualiza o throughput anterior para o próximo mês
    
    # Encontra o mês com menor valor médio de Average
    best_month_avg = min(average_data, key=average_data.get)
    # Encontra o mês com o maior valor de Throughput
    best_month_throughput = max(throughput_data, key=throughput_data.get)
    # Retorna as informações
    return average_data, throughput_data, avg_trend, throughput_trend, best_month_avg, best_month_throughput

def plot_graph(data, title, max_value, output_directory, metric):
    """Função para plotar gráficos."""
    os.makedirs(output_directory, exist_ok=True)  # Cria o diretório de saída se não existir
    plt.figure(figsize=(10, 5))  # Define o tamanho da figura
    plt.plot(list(data.keys()), list(data.values()), marker='o')  # Plota os dados
    plt.title(f"{title} - {metric}")  # Define o título do gráfico
    plt.xlabel('Month')  # Define o rótulo do eixo x
    plt.ylabel(title)  # Define o rótulo do eixo y
    plt.grid(True)  # Adiciona grade ao gráfico
    plt.xticks(rotation=45)  # Rotaciona os rótulos do eixo x para melhor legibilidade
    plt.ylim(0, max_value + 50 if title == "Average Time" else max_value + 2.5)  # Define o intervalo do eixo y
    if title == "Average Time":
        plt.yticks(np.arange(0, max_value + 50, 50))  # Define os rótulos do eixo y para Average Time
    else:
        plt.yticks(np.arange(0, max_value + 2.5, 2.5))  # Define os rótulos do eixo y para Throughput
    for month, value in data.items():  # Iteração sobre os dados para adicionar rótulos nos pontos
        plt.text(month, value, str(round(value, 2)), ha='center', va='bottom')  # Adiciona o valor como rótulo
    plt.tight_layout()  # Ajusta o layout do gráfico
    plt.savefig(os.path.join(output_directory, f"{title.lower().replace(' ', '_')}_{metric.lower()}_graph.png"))  # Salva o gráfico como imagem
    plt.show()  # Exibe o gráfico

def analyze_residuals(X, y, y_pred, output_directory, metric):
    """Função para analisar os resíduos."""
    os.makedirs(output_directory, exist_ok=True)  # Cria o diretório de saída se não existir
    residuals = y - y_pred  # Calcula os resíduos
    fig, ax = plt.subplots(figsize=(10, 5))  # Cria uma figura e eixos para o gráfico
    ax.scatter(X, residuals)  # Plota os resíduos
    ax.axhline(y=0, color='gray', linestyle='--')  # Adiciona uma linha horizontal em y=0
    ax.set_title(f'Análise de Resíduos - {metric}')  # Define o título do gráfico
    ax.set_xlabel('Índice')  # Define o rótulo do eixo x
    ax.set_ylabel('Resíduos')  # Define o rótulo do eixo y
    plt.tight_layout()  # Ajusta o layout do gráfico
    plt.savefig(os.path.join(output_directory, f"residuals_analysis_{metric.lower()}.png"))  # Salva o gráfico como imagem
    plt.show()  # Exibe o gráfico de dispersão dos resíduos

    # Plotagem do Q-Q plot dos resíduos
    sm.qqplot(residuals.flatten(), line ='45')  # Cria o Q-Q plot
    plt.title(f'Q-Q Plot dos Resíduos - {metric}')  # Define o título do gráfico
    plt.savefig(os.path.join(output_directory, f"qqplot_residuals_{metric.lower()}.png"))  # Salva o Q-Q plot como imagem
    plt.show()  # Exibe o Q-Q plot

def train_linear_regression(X, y):
    """Função para treinar um modelo de regressão linear."""
    model = LinearRegression()  # Inicializa um modelo de regressão linear
    model.fit(X, y)  # Treina o modelo com os dados de entrada X e saída y
    return model  # Retorna o modelo treinado

def evaluate_model(y_true, y_pred):
    """Função para avaliar o modelo de regressão linear."""
    r2 = r2_score(y_true, y_pred)  # Calcula o coeficiente de determinação (R²)
    mae = mean_absolute_error(y_true, y_pred)  # Calcula o erro médio absoluto (MAE)
    mse = mean_squared_error(y_true, y_pred)  # Calcula o erro quadrático médio (MSE)
    return r2, mae, mse  # Retorna as métricas de avaliação

def print_results(best_month_avg, best_month_throughput, model_avg_coef, model_throughput_coef, r2_avg, r2_throughput, mae_avg, mse_avg, avg_trend, throughput_trend, output_directory):
    """Função para imprimir os resultados."""
    os.makedirs(output_directory, exist_ok=True)  # Cria o diretório de saída se não existir
    now = datetime.now()  # Obtém a data e hora atuais
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")  # Formata a data e hora como string
    logging.basicConfig(filename=os.path.join(output_directory, f"analysis_log_{timestamp}.txt"), level=logging.INFO)  # Configura o logging para registrar eventos em um arquivo
    logging.info("Melhor mês em termos de Average Time: %s", best_month_avg)  # Registra o melhor mês em termos de Average Time
    logging.info("Melhor mês em termos de Throughput: %s", best_month_throughput)  # Registra o melhor mês em termos de Throughput
    logging.info("Coeficiente de inclinação do modelo de tendência de Average Time: %s", model_avg_coef)  # Registra o coeficiente de inclinação do modelo de tendência de Average Time
    logging.info("Coeficiente de inclinação do modelo de tendência de Throughput: %s", model_throughput_coef)  # Registra o coeficiente de inclinação do modelo de tendência de Throughput
    logging.info("Precisão do modelo de Average Time (R²): %s", r2_avg)  # Registra a precisão do modelo de Average Time (R²)
    logging.info("Precisão do modelo de Throughput (R²): %s", r2_throughput)  # Registra a precisão do modelo de Throughput (R²)
    logging.info("Erro Médio Absoluto (MAE) do modelo de Average Time: %s", mae_avg)  # Registra o Erro Médio Absoluto (MAE) do modelo de Average Time
    logging.info("Erro Quadrático Médio (MSE) do modelo de Average Time: %s", mse_avg)  # Registra o Erro Quadrático Médio (MSE) do modelo de Average Time
    logging.info("Tendência do Average Time: %s", avg_trend)  # Registra a tendência do Average Time
    logging.info("Tendência do Throughput: %s", throughput_trend)  # Registra a tendência do Throughput

def main(directory, output_directory):
    """Função principal."""
    # Leitura dos arquivos CSV
    data = read_csv_files(directory)

    # Processamento dos dados
    average_data, throughput_data, avg_trend, throughput_trend, best_month_avg, best_month_throughput = process_data(data)

    # Plotagem dos gráficos
    max_average = max(average_data.values())
    max_throughput = max(throughput_data.values())
    plot_graph(average_data, "Average Time", max_average, output_directory, "Resultado")
    plot_graph(throughput_data, "Throughput", max_throughput, output_directory, "Resultado")

    # Treinamento do modelo de regressão linear para Average Time
    X_avg = np.array(range(len(average_data))).reshape(-1, 1)
    y_avg = np.array(list(average_data.values())).reshape(-1, 1)
    model_avg = train_linear_regression(X_avg, y_avg)
    y_avg_pred = model_avg.predict(X_avg)

    # Avaliação do modelo de regressão linear para Average Time
    r2_avg, mae_avg, mse_avg = evaluate_model(y_avg, y_avg_pred)

    # Treinamento do modelo de regressão linear para Throughput
    X_throughput = np.array(range(len(throughput_data))).reshape(-1, 1)
    y_throughput = np.array(list(throughput_data.values())).reshape(-1, 1)
    model_throughput = train_linear_regression(X_throughput, y_throughput)
    y_throughput_pred = model_throughput.predict(X_throughput)

    # Avaliação do modelo de regressão linear para Throughput
    r2_throughput, mae_throughput, mse_throughput = evaluate_model(y_throughput, y_throughput_pred)

    # Impressão dos resultados
    print_results(best_month_avg, best_month_throughput, model_avg.coef_[0][0], model_throughput.coef_[0][0], r2_avg, r2_throughput, mae_avg, mse_avg, avg_trend, throughput_trend, output_directory)

    # Análise de Resíduos
    analyze_residuals(X_avg, y_avg, y_avg_pred, output_directory, "Average")
    analyze_residuals(X_throughput, y_throughput, y_throughput_pred, output_directory, "Throughput")

if __name__ == "__main__":
    csv_directory = "/Users/palleta/Desktop/Dados_Mensais"  # Diretório contendo os arquivos CSV
    output_directory = "/Users/palleta/Desktop/Dados_Mensais/Resultado_Analise"  # Diretório de saída para os resultados e gráficos
    main(csv_directory, output_directory)  # Chamada da função principal