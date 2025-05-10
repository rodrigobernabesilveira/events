import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
import time
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

# Carregar arquivo
df = pd.read_csv('./dwenglish.csv')

import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from unidecode import unidecode

###############

df['title'] = df['title'].astype(str)
df['content'] = df['content'].astype(str)
df['created_at'] = pd.to_datetime(df["created_at"])

full_text_col = 'full_text'
required_columns = ["title", "content", full_text_col]
df[full_text_col] = df["title"] + " " + df["content"]

###############

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

df[full_text_col] = df[full_text_col].apply(preprocess_text)

from summa import keywords
def define_keywords(post):
    k = keywords.keywords(post[full_text_col], scores=True, language='english')
    k = [(a[0],sigmoid(a[1])) for a in k]
    return k

import numpy as np
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

from sklearn.cluster import KMeans
def aplica_k_means(X,n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(X)
    return kmeans

def pegar_posts_cluster_text(clustering, df):
    #retira os outliers
    labels = [l for l in clustering.labels_ if l > -1]
    clusters = set(labels)
    dados = []
    posts_cluster = []
    text = ''
    for cluster in clusters:
        indices = np.where(clustering.labels_ == cluster)
        for indice in np.nditer(indices):
            posts_cluster.append(df.iloc[indice])
        #ordena por data antes de adicionar na lista
        posts_cluster.sort(key=lambda x: x.created_at, reverse=False)
        dados.append(posts_cluster)
        posts_cluster = []
    return dados

#retorna a lista de keywords de cada post do cluster
def pegar_keywords_posts(posts):
    keyword_list = []
    for post in posts:
        keywords_ = define_keywords(post)
        keyword_list.append(keywords_)
    return keyword_list

def get_str_keywords(post1):
    list_post1 = list(zip(*post1))[0]
    str_1 = ''
    for st in list_post1:
        str_1 += ' ' + st
    return str_1

def compare_similarity_str(str1, str2):
    doc1 = nlp(str1)
    doc2 = nlp(str2)
    return doc1.similarity(doc2)

#retorna a lista de keywords de cada post dos clusters da semana
def pegar_keywords_posts_week(clustering):
    keywords_week = []
    # retira os outliers
    labels = [l for l in clustering.labels_ if l > -1]
    clusters = set(labels)
    dados = []
    text = ''
    keywords_cluster = []
    for cluster in clusters:
        indices = np.where(clustering.labels_ == cluster)
        for indice in np.nditer(indices):
            keywords_cluster.append(pegar_keywords_post(df.iloc[indice]))
        keywords_week.append(keywords_cluster)
        keywords_cluster = []
    return keywords_week

#retorna a lista de keywords de cada post do cluster
def pegar_keywords_post(post):
    return define_keywords(post)

def prepara_lista_week(keyword_list_week):
    lista = []
    for _,  week in enumerate(keyword_list_week):
        for _, cluster in enumerate(week):
            for _, tupla in enumerate(cluster):
                lista.append(tupla)
    return lista

def pega_cluster_semana(clustering, clusters_semana):
    labels = [l for l in clustering.labels_ if l > -1]
    clusters = set(labels)
    clusters_semana.append(len(clusters))

def pega_eventos_semana(qtd, eventos_semana):
    eventos_semana.append(qtd)

# surface web
# df_ = df.loc[(df['created_at'] >= '2021-08-24') & (df['created_at'] <= '2022-02-28')]
# dark web
df_ = df.loc[(df['created_at'] >= '2023-02-18') & (df['created_at'] <= '2024-02-18')]

#surface web
#qtd_posts = df.loc[(df['created_at'] >= '2021-08-24') & (df['created_at'] <= '2022-02-28')]
#dark web
qtd_posts = df.loc[(df['created_at'] >= '2023-02-18') & (df['created_at'] <= '2024-02-18')]
len(qtd_posts)

#divide o dataframe em periodos de 1 semana
df_.sort_values(by='created_at', ascending = True, inplace = True)
g = df_.groupby(pd.Grouper(key='created_at', freq='W'))
dfs = [group for _,group in g]
dfs = [df for df in dfs if len(df) > 0]

#usando gpu no spacy
spacy.require_gpu()
nlp = en_core_web_lg.load(exclude=["tagger", "parser", "senter", "attribute_ruler", "lemmatizer","ner"])

cs_thresold = 0.4
eventos_novos_geral = []
keyword_list_week = []
clusters_semana = []
eventos_semana = []
qtd_eventos_semana = 0

from sentence_transformers import SentenceTransformer
modelSBert = SentenceTransformer("all-MiniLM-L6-v2")
full_text_col = 'full_text'

# Função para gerar embeddings usando S-BERT
def get_embeddings(text):
    return modelSBert.encode(text)

########## BUSCA EVENTOS ##########
start = time.time()
for idx, df_ in enumerate(dfs):
    keywords_week = []
    df_ = df_.reset_index(drop=True)
    corpus = df_[full_text_col]
    n_clusters = 3  # default
    # tf_idf
    X = get_embeddings(corpus)
    if X.shape[0] < 3:
        n_clusters = X.shape[0]

    clustering = aplica_k_means(X, n_clusters)
    pega_cluster_semana(clustering, clusters_semana)

    # se for primeira semana, somente pega as keywords
    if idx == 0:
        keyword_list_week = pegar_keywords_posts_week(clustering)
        continue

    posts_cluster = pegar_posts_cluster_text(clustering, df_)
    for cluster in posts_cluster:
        eventos_novos = []
        # pega as keywords dos posts do cluster
        keyword_list = pegar_keywords_posts(cluster)
        # calcula a similaridade por coseno
        for idx, keywords_ in enumerate(keyword_list):
            if len(keywords_) > 0 and len(keyword_list[idx - 1]) > 0:
                keyword_list_week_temp = prepara_lista_week(keyword_list_week)
                # cs = compare_similarity(keyword_list_week_temp, keywords_)
                cs = modelSBert.similarity(get_embeddings(get_str_keywords(keyword_list_week_temp)),
                                           get_embeddings(get_str_keywords(keywords_))).item()
                if cs < cs_thresold:
                    eventos_novos.append([cluster[idx], keywords_])
        eventos_novos_geral.append(eventos_novos)
        qtd_eventos_semana += len(eventos_novos)
        keywords_week.append(keyword_list)

    # add na lista as keywords da semana atual para serem comparadas com os da semana seguinte
    keyword_list_week = []
    for kw in keywords_week:
        keyword_list_week.append(kw)
    pega_eventos_semana(qtd_eventos_semana, eventos_semana)
    qtd_eventos_semana = 0

end = time.time()
print(end - start)