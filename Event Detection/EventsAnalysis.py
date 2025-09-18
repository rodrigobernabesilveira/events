import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.cluster import HDBSCAN
from operator import itemgetter
import pickle
import spacy
import en_core_web_lg
from spacy import load
from collections import defaultdict
import nltk
from nltk.cluster.util import cosine_distance
import os
os.environ["OMP_NUM_THREADS"] = "1"

full_text_col = 'full_text'

import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from unidecode import unidecode


nltk.download('stopwords')
nltk.download('rslp')

stop_words = set(stopwords.words('english'))
stop_words_spanish = set(stopwords.words('spanish'))

stop_words.update(["En"])


def preprocess_text(text):
    # Substituir caracteres com acentos por seus equivalentes sem acentos
    text = unidecode(text)
    # Converter o texto para minúsculo
    text = text.lower()
    # Substituir \\n por um espaço em branco
    text = re.sub(r'\s*\\n\s*', ' ', text)
    # Remover tags HTML
    text = BeautifulSoup(text, 'html.parser').get_text()
    # Remover links .onion
    text = re.sub(r'\S*\.onion\S*', ' ', text)
    # Remover URLs
    text = re.sub(r'http\S+', ' ', text)
    # Remover os title que aparecem em answers
    text = re.sub(r"'title': [^,]*,", ' ', text)
    # Remover os nomes de usuário que aparece em answers
    text = re.sub(r"'name': [^,]*,", ' ', text)
    # Remover os type que aparecem em answers
    text = re.sub(r"'type': [^,]*,", ' ', text)
    # Remover os author que aparecem em answers
    text = re.sub(r"'author': [^,]*,", ' ', text)
    # Remover sequência de kkkk
    text = re.sub(r'k{2,}\S*', ' ', text)

    # Remover termos com mais de 4 consoantes seguidas
    consoantes_5m = "([bcdfghjklmnpqrstvwxyz]{5,})"
    text = re.sub(consoantes_5m, " ", text)
    # Remover termos com mais de 5 vogais seguidas
    vogais_6m = "([aeiou]{6,})"
    text = re.sub(vogais_6m, " ", text)

    # Remover o texto links encontrados
    text = re.sub(r"links encontrados", ' ', text)
    text = re.sub(r"To view the content, you need to Sign In or Register.", ' ', text)

    # Substituir caracteres que não são letras por espaços em branco
    text = re.sub('[^A-Za-z]+', ' ', text)
    # Remover espaços em branco extras e substituir sequências de espaços em branco por um único espaço em branco
    text = re.sub('\s+', ' ', text.strip())
    # Remover stop words
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [word for word in words if word not in stop_words_spanish]
    words = [word for word in words if len(word) > 2]
    text = ' '.join(words)
    return text

def carregar_dados(nome):
    with open(nome, "rb") as input_file:
        return pickle.load(input_file)
eventos_novos_geral = carregar_dados('eventos_novos_geraldw.pkl')

#Graficos
#numero de eventos novos por cluster
X = [x for x in range(1,len(eventos_novos_geral)+1)]
Y = [len(y) for y in eventos_novos_geral]
plt.bar(X,Y)
plt.title("Eventos Novos por Cluster")
plt.show()

def juntar_eventos(eventos_novos_geral):
    eventos = []
    for cluster in eventos_novos_geral:
        for evento in cluster:
            evento[0]['score'] = evento[2][0]
            eventos.append(evento[0])
    return eventos

#eventos por data
eventos = juntar_eventos(eventos_novos_geral)
df_eventos = pd.DataFrame(eventos)
df_eventos = df_eventos.fillna('')
df_eventos.sort_values(by='created_at', ascending = True, inplace = True)
g = df_eventos.groupby(pd.Grouper(key='created_at', freq='W'))
dfs = [group for _,group in g]

X = [df.iloc[-1]['created_at'].strftime('%d/%m/%Y') for df in dfs if len(df) > 0]
#X = ["S-" + str(idx+1) for idx, df in enumerate(dfs) if len(df) > 0]
Y = [len(df) for df in dfs if len(df) > 0]

plt.figure(figsize=(10, 6))
plt.bar(X,Y)
plt.xticks(rotation=80)
plt.ylabel("Quantidade de eventos")
plt.xlabel("Semanas")
SMALL_SIZE = 10
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#plt.savefig("qtdsemanadw.png", bbox_inches='tight')
plt.show()

categorias = df_eventos['category'].values
categorias[categorias == 'General - Hackonology - Knowledge Base - Free Courses and Resources'] = 'G-H-KB-FCR'
X = [categorias]
plt.figure(figsize=(10, 6))
plt.hist(X)
#plt.title("Categorias - Dark Web")
plt.ylabel("Categorias")
plt.xlabel("Quantidade")
plt.xticks(rotation=80)
#plt.savefig("categoriassw.png", bbox_inches='tight')
plt.show()

weeks = list(range(1, 53))
posts = [23, 45, 12, 56, 78, 34, 23, 45, 67, 89, 34, 23, 45, 67, 23, 12, 34, 45, 67, 78, 45, 23, 34, 45, 56, 67, 78, 89, 45, 34, 23, 45, 67, 78, 89, 45, 34, 23, 45, 67, 78, 89, 45, 34, 23, 45, 67, 78, 89, 45, 34, 23]

# Criar gráfico de linhas
plt.figure(figsize=(10, 6))
plt.plot(weeks, posts, marker='o', linestyle='-')
plt.xlabel('Semana')
plt.ylabel('Quantidade de Posts')
plt.title('Quantidade de Posts por Semana')
plt.grid(True)
plt.xticks(weeks, rotation=90)
plt.show()

import collections
Y = [ counter/len(categorias)  for categoria, counter in collections.Counter(categorias).items()]
X = [ categoria for categoria, counter in collections.Counter(categorias).items()]
plt.bar(X,Y)
plt.title("Porcentagem Categorias")
plt.xticks(rotation=80)
plt.show()


def Jaccard_Similarity(doc1, doc2):
    str_1 = ''
    str_2 = ''
    for st in doc1:
        str_1 += ' ' + st
    for st in doc2:
        str_2 += ' ' + st

    # List the unique words in a document
    words_doc1 = set(str_1.lower().split())
    words_doc2 = set(str_2.lower().split())

    # Find the intersection of words list of doc1 & doc2
    intersection = words_doc1.intersection(words_doc2)

    # Find the union of words list of doc1 & doc2
    union = words_doc1.union(words_doc2)

    # Calculate Jaccard similarity score
    # using length of intersection set divided by length of union set
    return float(len(intersection)) / len(union)


def calcula_similaridade(sentenca1, sentenca2):
    palavras1 = [palavra for palavra in nltk.word_tokenize(sentenca1)]
    palavras2 = [palavra for palavra in nltk.word_tokenize(sentenca2)]
    todas_palavras = list(set(palavras1 + palavras2))
    vetor1 = [0] * len(todas_palavras)
    vetor2 = [0] * len(todas_palavras)

    for palavra in palavras1:
        vetor1[todas_palavras.index(palavra)] += 1
    for palavra in palavras2:
        vetor2[todas_palavras.index(palavra)] += 1

    return 1 - cosine_distance(vetor1, vetor2)

def transform_array_into_text(list_post1):
    str_1 = ''
    for st in list_post1:
        str_1 += ' ' + st
    return str_1


posts_semana = []
resultado_semelhanca_semanas = []
for idx, df_ in enumerate(dfs):
    df_ = df_.reset_index(drop=True)

    palavras = []
    ptemp = [text for text in df_[full_text_col]]
    for t in ptemp:
        splited = t.split()
        palavras += splited

    if idx == 0:
        posts_semana = palavras
        resultado_semelhanca_semanas.append((0.1, 0.1, df_))
        continue

    # print('palavras ',len(palavras))
    # print('posts_semana ',len(posts_semana))
    if len(palavras) == 0:
        cs = 0
        js = 0
    else:
        cs = calcula_similaridade(transform_array_into_text(posts_semana), transform_array_into_text(palavras))
        js = Jaccard_Similarity(posts_semana, palavras)
        posts_semana = palavras

    if len(df_) > 0:
        resultado_semelhanca_semanas.append((cs, js, df_))

css = [cs[0] for cs in resultado_semelhanca_semanas]
jss = [js[1] for js in resultado_semelhanca_semanas]
X= [df.iloc[-1]['created_at'].strftime('%d/%m/%Y') for df in dfs if len(df) > 0]

plt.figure(figsize=(10, 6))
plt.plot(X[1:],css[1:], label="similaridade por cosseno")
plt.plot(X[1:],jss[1:], label="similaridade de Jaccard")
plt.legend(['similaridade por cosseno','similaridade de Jaccard'])
#plt.title("Similaridade por cosseno e similaridade de Jaccard - Surface Web")
plt.ylabel("Similaridades")
plt.xlabel("Semanas")
plt.xticks(rotation=80)
#plt.savefig("similaridadesw.png", bbox_inches='tight')
plt.legend(loc='best')
plt.show()

df_eventos_alta  = df_eventos.loc[((df_eventos['Relevância'] == 'Alta'))]
df_eventos_alta['content'] = df_eventos_alta['content'].apply(preprocess_text)
palavras_list = [''.join(x) for x in df_eventos_alta['content']]
palavras = []
for pl in palavras_list:
    pl_temp = pl.split()
    for palavra in pl_temp:
        palavras.append(palavra)

from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus']=False   #These two lines need to be set manually

word_list = palavras

counts = Counter(word_list)

labels, values = zip(*counts.items())

# sort your values in descending order
indSort = np.argsort(values)[::-1]

# rearrange your data
labels = np.array(labels)[indSort][:10]
values = np.array(values)[indSort][:10]

indexes = np.arange(len(labels))

bar_width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(indexes, values)
#plt.title("Histograma")

# add labels
plt.xticks(indexes + bar_width, labels)
plt.xticks(rotation=35)
plt.ylabel("Quantidade")
plt.xlabel("Palavras")
#plt.savefig("histogramasw.png", bbox_inches='tight')
plt.show()

######## CLASSIFICACAO ###########
import lightgbm as lgb
#carrega modelo
loaded_model = pickle.load(open("modelo_lgb_tfidf_unigram_ingles.pkl", 'rb'))
# To load the TF-IDF vectorizer from the file using pickle
with open('tfidf_vec_unigram_ingles.pkl', 'rb') as file:
    tfidf_vec_unigram = pickle.load(file)

X_text = df_eventos['full_text']
X_tfidf_unigram = tfidf_vec_unigram.transform(X_text)
gb_X_tfidf_unigram = X_tfidf_unigram.astype(np.float64)
lgb_y_pred_tfidf_unigram = loaded_model.predict(lgb_X_tfidf_unigram)
df_eventos = df_eventos.reset_index(drop=True)
docs = df_eventos['full_text']
for i, doc in enumerate(docs):
    df_eventos.loc[i, 'probability'] = lgb_y_pred_tfidf_unigram[i]

df_eventos.to_excel("eventodwabordagem3.xlsx")

# PIE CHART WITH PERCENTAGES

# Defining the probability ranges
faixas = [0, 0.3, 0.7, 1.0]
nomes_faixas = ['Baixa', 'Média', 'Alta']

df_eventos['Relevância'] = pd.cut(df_eventos['probability'], bins=faixas, labels=nomes_faixas, right=False)

# Calculate the percentages of categories in the 'Relevance' column
percentuais_relevancia = df_eventos['Relevância'].value_counts(normalize=True) * 100
percentuais_relevancia

# Calculate the overall total
total_geral = len(df_eventos)

# Define the colors for each category
cores = ['green','yellow', 'red']

# Create the pie chart
plt.figure(figsize=(6, 6))  # Define o tamanho da figura
plt.pie(percentuais_relevancia, labels=percentuais_relevancia.index, autopct='%.1f%%', startangle=140, colors=cores)

# Add title
#plt.title('Percentuais de Relevância dos', total_geral, 'Posts Classificados')
#plt.title('Percentuais de Relevância dos Eventos Novos Classificados - Dark Web')

# Create a custom legend
legendas = ['Baixa', 'Média', 'Alta']
plt.legend(legendas, title="Relevância", loc="best")

plt.savefig("relevanciasw.png", bbox_inches='tight')

# Display the chart
plt.show()

df_dw = pd.read_excel('./eventodwabordagem3.xlsx')
df_sw = pd.read_excel('./eventoswabordagem3.xlsx')

faixas = [0, 0.3, 0.7, 1.0]
nomes_faixas = ['Baixa', 'Média', 'Alta']

df_dw['Relevância'] = pd.cut(df_dw['probability'], bins=faixas, labels=nomes_faixas, right=False)
df_sw['Relevância'] = pd.cut(df_sw['probability'], bins=faixas, labels=nomes_faixas, right=False)

full_text_col = 'full_text'
faixas = [0, 0.3, 0.7, 1.0]
def transform_into_array(df):
    palavras = []
    ptemp = [text for text in df[full_text_col]]
    for t in ptemp:
        splited = t.split()
        palavras+= splited
    return palavras

def transform_into_text(array):
    frase = ''
    for palavra in array:
        frase+= palavra + ' '
    return frase

#Geral
palavras_dw = transform_into_array(df_dw)
palavras_sw = transform_into_array(df_sw)
js = Jaccard_Similarity(palavras_dw, palavras_sw)
print('Similaridade de Jaccard Geral: ',js * 100, '%')

#Relevencia Baixa
palavras_dw_baixa = transform_into_array(df_dw.loc[(df_dw['Relevância'] == 'Baixa')])
palavras_sw_baixa = transform_into_array(df_sw.loc[(df_sw['Relevância'] == 'Baixa')])
js_baixa = Jaccard_Similarity(palavras_dw_baixa, palavras_sw_baixa)
print('Similaridade de Jaccard (Relevancia Baixa): ',js_baixa * 100, '%')

#Relevencia Media
palavras_dw_media = transform_into_array(df_dw.loc[(df_dw['Relevância'] == 'Média')])
palavras_sw_media = transform_into_array(df_sw.loc[(df_sw['Relevância'] == 'Média')])
js_media = Jaccard_Similarity(palavras_dw_media, palavras_sw_media)
print('Similaridade de Jaccard (Relevancia Media): ',js_media * 100, '%')

#Relevencia Alta
palavras_dw_alta = transform_into_array(df_dw.loc[(df_dw['Relevância'] == 'Alta')])
palavras_sw_alta = transform_into_array(df_sw.loc[(df_sw['Relevância'] == 'Alta')])
js_alta = Jaccard_Similarity(palavras_dw_alta, palavras_sw_alta)
print('Similaridade de Jaccard (Relevancia Alta): ',js_alta * 100, '%')

palavras_dw = transform_into_text(transform_into_array(df_dw))
palavras_sw = transform_into_text(transform_into_array(df_sw))
js = calcula_similaridade(palavras_dw, palavras_sw)
print('Similaridade por cosseno Geral: ',js)

#Relevencia Baixa
palavras_dw_baixa = transform_into_text(transform_into_array(df_dw.loc[(df_dw['Relevância'] == 'Baixa')]))
palavras_sw_baixa = transform_into_text(transform_into_array(df_sw.loc[(df_sw['Relevância'] == 'Baixa')]))
js_baixa = calcula_similaridade(palavras_dw_baixa, palavras_sw_baixa)
print('Similaridade por cosseno (Relevancia Baixa): ',js_baixa * 100, '%')

#Relevencia Media
palavras_dw_media = transform_into_text(transform_into_array(df_dw.loc[(df_dw['Relevância'] == 'Média')]))
palavras_sw_media = transform_into_text(transform_into_array(df_sw.loc[(df_sw['Relevância'] == 'Média')]))
js_media = calcula_similaridade(palavras_dw_media, palavras_sw_media)
print('Similaridade por cosseno (Relevancia Media): ',js_media * 100, '%')

#Relevencia Alta
palavras_dw_alta = transform_into_text(transform_into_array(df_dw.loc[(df_dw['Relevância'] == 'Alta')]))
palavras_sw_alta = transform_into_text(transform_into_array(df_sw.loc[(df_sw['Relevância'] == 'Alta')]))
js_alta = calcula_similaridade(palavras_dw_alta, palavras_sw_alta)
print('Similaridade por cosseno (Relevancia Alta): ',js_alta * 100, '%')

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
full_text_col = 'full_text'

# Função para gerar embeddings usando S-BERT
def get_embeddings(text):
    return model.encode(text)


# Textos a serem comparados - SBERT
start = time.time()
matrix_similaridades = [[0 for _ in range(len(df_sw))] for _ in range(len(df_dw))]
# print(len(df_dw))
# print(len(df_sw))
for index_dw, row_dw in df_dw.iterrows():
    palavras_dw = row_dw[full_text_col]
    embedding1 = get_embeddings(palavras_dw)
    for index_sw, row_sw in df_sw.iterrows():
        palavras_sw = row_sw[full_text_col]
        embedding2 = get_embeddings(palavras_sw)

        # Calcular a similaridade usando a Similaridade de Cosseno
        similarity = model.similarity(embedding1, embedding2).item()
        # print(f"Similaridade entre os textos: {similarity}")
        # print(index_dw,index_sw)
        matrix_similaridades[index_dw][index_sw] = similarity

end = time.time()
print(end - start)

matrix_similaridades = [[0 for _ in range(len(df_sw))] for _ in range(len(df_dw))]
#print(len(df_dw))
#print(len(df_sw))
for index_dw, row_dw in df_dw.iterrows():
    for index_sw, row_sw in df_sw.iterrows():
        palavras_dw = transform_into_text(transform_into_array(row_dw))
        palavras_sw = transform_into_text(transform_into_array(row_sw))
        sc = calcula_similaridade(palavras_dw, palavras_sw)
        matrix_similaridades[index_dw][index_sw] = sc