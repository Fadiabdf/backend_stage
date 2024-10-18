from bson import ObjectId
from django.shortcuts import render, redirect
from django.http import JsonResponse
import json
from .models import Category, Corpus, Etiquette, Video, Commentaire, VideoCollection
from datetime import datetime
from django.views.decorators.http import require_POST
from django.core.paginator import Paginator
import uuid
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import requests
from itertools import combinations
from typing import Counter
import pandas as pd
import numpy as np
import re
from scipy.stats import chi2_contingency,chi2
from pymongo.errors import ServerSelectionTimeoutError
from .config import CORPUS
from pymongo import MongoClient
from collections import Counter
import os
from .forms import KeywordForm, URLForm
from googleapiclient.discovery import build
import youtube_dl
from yt_dlp import YoutubeDL
from openpyxl import Workbook
from django.http import HttpResponse
from io import BytesIO
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

client = MongoClient("mongodb+srv://maroua:maroua2003@cluster0.p6t0qwx.mongodb.net/")
db = client["projet_1cs"]


#----------------------------------------1ere statistique-------------------------------------------------
#-----------------------------------------------Recuperer les descripteurs-------------------------
def retrieve_descriptors(script_param,langues_param):
    count_Comment = 0
    corpus_id = get_corpus_id_by_title(CORPUS)
    data = []
    for video in video_collection.find({}, {"_id": 0, "videos.commentaires.descripteur": 1, "videos.commentaires.script": 1, "videos.commentaires.langue": 1, "videos.corpus":1}):
        for video_item in video.get('videos', []):
            if video_item.get('corpus')==corpus_id:
                for commentaire in video_item.get('commentaires', []):
                    if commentaire.get('script') == script_param:
                        commentaire_langues = commentaire.get('langue', [])
                        if set(commentaire_langues).issubset(set(langues_param)):
                            count_Comment += 1
                            descripteurs = commentaire.get('descripteur', [])
                            data.append({'descripteur': descripteurs})
    return pd.DataFrame(data)
    
#-------------------------------------Recuperer l'id d'une etiquette-------------------------------
def get_etiquette_id(collection,name):
    result = collection.find_one({"name": name.strip()}, {"_id": 1})
    return str(result["_id"]) if result else None

#--------------------------------------Fonction pour calculer mi et khi2 pour 2 descripteurs------------------
def calculate_mi_khi2_desc(df, x_names, y_names):
    etiquette_collection = db["etiquette"]
    x_set = {get_etiquette_id(etiquette_collection,name) for name in x_names if get_etiquette_id(etiquette_collection,name)}
    y_set = {get_etiquette_id(etiquette_collection,name) for name in y_names if get_etiquette_id(etiquette_collection,name)}

    all_annotations = x_set.union(y_set)
    for annotation in all_annotations:
        df[f'annotation_{annotation}'] = df['descripteur'].apply(lambda x: int(annotation in x))

    x_cols = [f'annotation_{x_id}' for x_id in x_set]
    y_cols = [f'annotation_{y_id}' for y_id in y_set]

    df['combined_x'] = df[x_cols].apply(lambda row: all(row), axis=1).astype(int)
    df['combined_y'] = df[y_cols].apply(lambda row: all(row), axis=1).astype(int)

    contingency_table = pd.crosstab(index=df['combined_x'], columns=df['combined_y'])
    contingency_table = contingency_table.reindex(index=[0, 1], columns=[0, 1], fill_value=0)

    try:
        khi2, _, _, _ = chi2_contingency(contingency_table, correction=False)
        chi2_result = khi2
    except ValueError:
        chi2_result = 0

    total_documents = len(df)
    print(total_documents)
    has_x = df['descripteur'].apply(lambda annotations: x_set.issubset(set(annotations)))
    has_y = df['descripteur'].apply(lambda annotations: y_set.issubset(set(annotations)))
    count_x = has_x.sum()
    count_y = has_y.sum()
    count_xy = (has_x & has_y).sum()
    print("tout ici",x_names,y_names,count_x, count_y, count_xy)
    mi_result = 0    
    if count_x > 0 and count_y > 0 and count_xy > 0:
        mi_result = np.log2(count_xy*total_documents / (count_x * count_y))

    return chi2_result, mi_result

#-------------------------Calculer tous les mi et khi2 pour les options selectionnees-------------------
def calculate_descriptors(list1, list2, script, langues):
    df = retrieve_descriptors(script, langues)
    results = [None] * len(list2)  
    results_khi2 = [None] * len(list2)

    list1_str = [item if isinstance(item, list) else [str(item)] for item in list1]
    list2_str = [item if isinstance(item, list) else [str(item)] for item in list2]

    flattened_list1 = [item for sublist in list1_str for item in sublist]
    flattened_list2 = [item for sublist in list2_str for item in sublist]

    def process_item(index, item2):
        result_khi2, result = calculate_mi_khi2_desc(df, flattened_list1, [item2])
        return index, result_khi2, result

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_item, i, item2) for i, item2 in enumerate(flattened_list2)]
        for future in concurrent.futures.as_completed(futures):
            index, result_khi2, result = future.result()
            results[index] = result
            results_khi2[index] = result_khi2

    result_khi2, result = calculate_mi_khi2_desc(df, flattened_list1, flattened_list2)
    results.append(result)
    results_khi2.append(result_khi2)

    return results_khi2, results

#------------------------------------Retourner les resultats des metriques (desc) au frontend------------------
def etqchoisie_view(request):
    if request.method == 'POST':
        try:
            payload = json.loads(request.body)
            selected_options_x = payload.get('selectedX', [])  
            selected_options_y = payload.get('selectedY', []) 
            selected_options_languages = payload.get('languages', [])
            selected_script = payload.get('script')
            khi2_theorique=chi2.ppf(1-0.05,1)
            print("chi2",khi2_theorique)
            descriptors_khi2, descriptors = calculate_descriptors(selected_options_x, selected_options_y,selected_script,selected_options_languages)
            response_data = {
                'message': 'Data received successfully',
                'selectedOptionsX': selected_options_x,
                'selectedOptionsY': selected_options_y,
                'descriptors': descriptors,  
                'descriptors_khi2': descriptors_khi2,
                'khi2_theoriques' : khi2_theorique,
            }
            return JsonResponse(response_data)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
    else:
        return JsonResponse({'error': 'Invalid HTTP method'}, status=405)
    
#------------------------------------------------2eme statistique---------------------------------------
def get_mots():
    video_collection = db["commentaires_nettoyes"]
    global CORPUS
    corpus_id = get_corpus_id_by_title(CORPUS)
    conflit_id = str(get_conflit_id()) 

    if not corpus_id:
        return []

    mots = set()
    for video in video_collection.find({}, {"_id": 0, "videos.commentaires.texte": 1, "videos.commentaires.descripteur": 1, "videos.corpus": 1}):
        videos = video.get('videos', [])
        for video_item in videos:
            if video_item.get('corpus') == corpus_id:
                commentaires = video_item.get('commentaires', [])
                for commentaire in commentaires:
                    descripteur = commentaire.get('descripteur', [])
                    if conflit_id in descripteur:
                        texte = commentaire.get('texte', '')
                        texte = texte.replace(',', '') 
                        mots.update(texte.split()) 
                        
    print("nombre de mots",len(mots))

    return list(mots)
#------------------------------Recuperer les mots-------------------------------------------
def retrieve_mots(script_param,langues_param):
    video_collection = db["video_collection"]
    corpus_id = get_corpus_id_by_title(CORPUS)
    conflit_id = str(get_conflit_id())
    data = []
    for video in video_collection.find({}, {"_id": 0, "videos.commentaires.texte": 1, "videos.commentaires.script": 1, "videos.commentaires.langue": 1, "videos.commentaires.descripteur": 1, "videos.corpus": corpus_id}):
        for video_item in video.get('videos', []):
            for commentaire in video_item.get('commentaires', []):
                texte = commentaire.get('texte', '')
                descripteur = commentaire.get('descripteur', [])
                if conflit_id in descripteur and commentaire.get('script') == script_param: 
                    commentaire_langues = commentaire.get('langue', [])
                    if set(commentaire_langues).issubset(set(langues_param)):
                        data.append({'texte': texte})
    return pd.DataFrame(data)

#------------------------------------Fonction pour calculer mi et khi2 pour 2 mots---------------------------
def calculate_mi_khi2_mots(df, x_mots, y_mots):
    if isinstance(x_mots, str):
        x_words = [word.strip() for word in x_mots.split(',')]
    else:
        x_words = [word.strip() for word in x_mots]

    if isinstance(y_mots, str):
        y_words = [word.strip() for word in y_mots.split(',')]
    else:
        y_words = [word.strip() for word in y_mots]

    all_words = set(x_words).union(set(y_words))
    for word in all_words:
        df[f'word_{word}'] = df['texte'].apply(lambda x: int(word in x))

    x_cols = [f'word_{word}' for word in x_words]
    y_cols = [f'word_{word}' for word in y_words]

    df['combined_x'] = df[x_cols].apply(lambda row: all(row), axis=1).astype(int)
    df['combined_y'] = df[y_cols].apply(lambda row: all(row), axis=1).astype(int)

    contingency_table = pd.crosstab(index=df['combined_x'], columns=df['combined_y'])
    contingency_table = contingency_table.reindex(index=[0, 1], columns=[0, 1], fill_value=0)

    try:
        khi2, _, _, _ = chi2_contingency(contingency_table, correction=False)
        chi2_result = khi2
    except ValueError:
        chi2_result = 0

    x_set = set(x_words)
    y_set = set(y_words)

    total_documents = len(df)
    print("total",total_documents)
    has_x = df['texte'].apply(lambda annotations: x_set.issubset(set(annotations.split())))
    has_y = df['texte'].apply(lambda annotations: y_set.issubset(set(annotations.split())))
    count_x = has_x.sum()
    count_y = has_y.sum()
    count_xy = (has_x & has_y).sum()
    print(count_x,count_y,count_xy)

    mi_result = 0 
    if count_x > 0 and count_y > 0 and count_xy > 0:
        mi_result += np.log2(count_xy * total_documents / (count_x * count_y))

    return chi2_result, mi_result

#------------------------------------Calculer mi et khi2 pour les options selectionnees-----------------------
def calculate_mots(list1, list2, script, langues):
    df = retrieve_mots(script, langues)
    results = [None] * len(list2) 
    results_khi2 = [None] * len(list2)

    list1_str = [item if isinstance(item, list) else [str(item)] for item in list1]
    list2_str = [item if isinstance(item, list) else [str(item)] for item in list2]

    flattened_list1 = [item for sublist in list1_str for item in sublist]
    flattened_list2 = [item for sublist in list2_str for item in sublist]

    def process_item(index, item2):
        result_khi2, result = calculate_mi_khi2_mots(df, flattened_list1, [item2])
        return index, result_khi2, result

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_item, idx, item2) for idx, item2 in enumerate(flattened_list2)]
        for future in concurrent.futures.as_completed(futures):
            idx, result_khi2, result = future.result()
            results[idx] = result
            results_khi2[idx] = result_khi2

    result_khi2, result = calculate_mi_khi2_mots(df, flattened_list1, flattened_list2)
    results.append(result)
    results_khi2.append(result_khi2)

    return results_khi2, results

#---------------------------------Requete pour envoyer les resultats(mots)----------------------------------
def motchoisie_view(request):
    if request.method == 'POST':
        try:
            payload = json.loads(request.body)
            selected_options_x = payload.get('selectedX', []) 
            selected_options_y = payload.get('selectedY', []) 
            selected_options_languages = payload.get('languages', [])
            selected_script = payload.get('script')
            khi2_theorique = chi2.ppf(1-0.05,1)
            print("le chi2",khi2_theorique)
            mots_khi2,mots = calculate_mots(selected_options_x, selected_options_y,selected_script,selected_options_languages)
            response_data = {
                'message': 'Data received successfully',
                'selectedOptionsX': selected_options_x,
                'selectedOptionsY': selected_options_y,
                'mots': mots, 
                'mots_khi2': mots_khi2,
                'khi2_theoriques' : khi2_theorique
                
            }
            return JsonResponse(response_data)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
    else:
        return JsonResponse({'error': 'Invalid HTTP method'}, status=405)
    
#--------------------------------------Statistique excel----------------------------------------------
#------------------------------------Recuperer tous les mots (peu importe script, langue)--------------------
def retrieve_mots_excel():
    video_collection = db["video_collection"]
    corpus_id = get_corpus_id_by_title(CORPUS)
    conflit_id = str(get_conflit_id())  
    data = []
    for video in video_collection.find({}, {"_id": 0, "videos.commentaires.texte": 1, "videos.commentaires.descripteur": 1, "videos.corpus": 1}):
        for video_item in video.get('videos', []):
            if video_item.get('corpus') == corpus_id:
                for commentaire in video_item.get('commentaires', []):
                    descripteur = commentaire.get('descripteur', [])
                    if conflit_id in descripteur:
                        texte = commentaire.get('texte', '')
                        data.append({'texte': texte})

    return pd.DataFrame(data)

#----------------------------Calculer mi et khi2 pour deux mots (excel)--------------------------------

def calculate_excel(df, x_mots, y_mots):
    def process_text(text):
        return set(text.split())
    
    if isinstance(x_mots, str):
        x_words = [word.strip() for word in x_mots.split(',')]
    else:
        x_words = [word.strip() for word in x_mots]

    if isinstance(y_mots, str):
        y_words = [word.strip() for word in y_mots.split(',')]
    else:
        y_words = [word.strip() for word in y_mots]

    all_words = set(x_words).union(set(y_words))
    
    columns_dict = {}
    for word in all_words:
        columns_dict[f'word_{word}'] = df['texte'].apply(lambda x: int(word in process_text(x)))

    new_df = pd.concat([df, pd.DataFrame(columns_dict)], axis=1)
    
    x_cols = [f'word_{word}' for word in x_words]
    y_cols = [f'word_{word}' for word in y_words]

    new_df['combined_x'] = new_df[x_cols].apply(lambda row: all(row), axis=1).astype(int)
    new_df['combined_y'] = new_df[y_cols].apply(lambda row: all(row), axis=1).astype(int)

    contingency_table = pd.crosstab(index=new_df['combined_x'], columns=new_df['combined_y'])
    contingency_table = contingency_table.reindex(index=[0, 1], columns=[0, 1], fill_value=0)

    try:
        chi2, _, _, _= chi2_contingency(contingency_table, correction=False)
        chi2_result = chi2
    except ValueError:
        chi2_result = 0

    x_set = set(x_words)
    y_set = set(y_words)

    total_documents = len(new_df)
    has_x = new_df['texte'].apply(lambda annotations: x_set.issubset(process_text(annotations)))
    has_y = new_df['texte'].apply(lambda annotations: y_set.issubset(process_text(annotations)))
    
    count_x = has_x.sum()
    count_y = has_y.sum()
    count_xy = (has_x & has_y).sum()

    mi_result = 0
    if count_x > 0 and count_y > 0 and count_xy > 0:
        mi_result += np.log2(count_xy * total_documents / (count_x * count_y))

    return chi2_result, mi_result

#--------------------------------------------Preparer le fichier excel----------------------------

def get_top_words(df, column='texte', limit=100):
    word_counter = Counter()

    def extract_words(text):
        return text.split()

    for text in df[column].dropna(): 
        words = extract_words(text)
        word_counter.update(words)
    
    top_words_with_counts = word_counter.most_common(limit)
    top_words = [word for word, _ in top_words_with_counts]
    
    return top_words


def calculate_all_excel():
    script = "Latin Script"
    langues = ["AZL", "FR", "TZL", "EN", "EMT"]
    df = retrieve_mots(script, langues)
    mots = get_top_words(df)
    results_mi = []
    results_khi2 = []

    def process_pair(df, mot1, mot2):
        try:
            khi2, mi = calculate_excel(df, mot1, mot2)
            return mot2, khi2, mi
        except Exception as e:
            print(f'Error in calculate_excel for pair ({mot1}, {mot2}): {e}')
            return mot2, None, None

    for i, mot1 in enumerate(mots):
        mi_values = []
        khi2_values = []

        with ThreadPoolExecutor(max_workers=32) as executor:
            future_to_mot2 = {
                executor.submit(process_pair, df, mot1, mot2): mot2
                for j, mot2 in enumerate(mots) if i != j
            }

            for future in as_completed(future_to_mot2):
                mot2 = future_to_mot2[future]
                try:
                    mot2, khi2, mi = future.result()
                    if mi is not None and khi2 is not None:
                        mi_values.append((mot2, mi))
                        khi2_values.append((mot2, khi2))
                    else:
                        print(f'Failed to process pair ({mot1}, {mot2})')
                except Exception as e:
                    print(f'Error processing pair ({mot1}, {mot2}): {e}')

        mi_values.sort(key=lambda x: x[1], reverse=True)
        khi2_values.sort(key=lambda x: x[1], reverse=True)

        results_mi.append([mot1] + [(x[0], x[1]) for x in mi_values])
        results_khi2.append([mot1] + [(x[0], x[1]) for x in khi2_values])

    return results_mi, results_khi2

#-------------------------------------------Retourner le fichier excel au front-----------------------
def download_excel(request):
    
    results_mi, results_khi2 = calculate_all_excel()
    df1 = pd.DataFrame(results_mi)
    df2 = pd.DataFrame(results_khi2)
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df1.to_excel(writer, sheet_name='Information mutuelle', index=False, header=False)  
        df2.to_excel(writer, sheet_name='Khi2', index=False, header=False)  


    output.seek(0)
    response = HttpResponse(output, content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename="associations_binaire.xlsx"'
    
    return response

#-------------------------------4eme statistique------------------------------------------------

def calculate_freq_perc(user_labels_list):
    video_collection = db["video_collection"]
    etiquette_collection = db["etiquette"]
    descripteurs = []
    corpus_id = get_corpus_id_by_title(CORPUS)
    
    for video in video_collection.find({}, {"_id": 0, "videos.commentaires.descripteur": 1, "videos.corpus": 1}):
        for video_item in video.get('videos', []):
            if video_item.get('corpus') == corpus_id:
                for commentaire in video_item.get('commentaires', []):
                    descripteurs.append(set(commentaire.get('descripteur', [])))
    
    descriptor_sets = [set(desc) for desc in descripteurs]

    etiquette_mapping = {}
    for etiquette in etiquette_collection.find({}, {"_id": 1, "name": 1}):
        etiquette_mapping[etiquette["name"].lower()] = str(etiquette["_id"])
    
    def labels_to_ids(labels):
        if isinstance(labels, str):
            return {etiquette_mapping.get(labels.strip().lower(), None)}
        elif isinstance(labels, list):
            return {etiquette_mapping.get(label.strip().lower(), None) for label in labels}
        return set()
    
    def calculate_stats(ids):
        ids = set(ids)
        if None in ids or not ids:
            return 0
        count = sum(1 for desc_set in descriptor_sets if ids.issubset(desc_set))
        return count
    
    combo_ids = labels_to_ids(user_labels_list)
    frequency = calculate_stats(combo_ids)
    total_descriptors = len(descriptor_sets)
    percentage = (frequency / total_descriptors * 100) if total_descriptors > 0 else 0
    
    return frequency, percentage, total_descriptors



video_collection = db["video_collection"]
def script_choisi(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'}, status=405)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    selected_option = data.get('selectedOption')
    if not selected_option:
        return JsonResponse({'error': 'selectedOption not provided'}, status=400)
    
    result = freq_lang(selected_option)
    if not isinstance(result, list):
        return JsonResponse({'error': 'Unexpected data format from freq_lang'}, status=500)

    labels = [item['label'] for item in result if 'label' in item]
    frequences = [item['frequency'] for item in result if 'frequency' in item]
    
    return JsonResponse({'labels': labels, 'frequences': frequences})


def get_freq_script(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'Invalid request method'}, status=405)

    frequences = freq_script()
    return JsonResponse(frequences, safe=False)

def get_conflit_id():
    etiquette_collection = db["etiquette"]
    conflit_doc = etiquette_collection.find_one({"name": "Conflit"}, {"_id": 1})
    return conflit_doc['_id'] if conflit_doc else None

def freq_script():
    script_counts = {
        'Latin Script': 0,
        'Arabic Script': 0,
        'Mixed Script': 0
    }

    corpus_id = get_corpus_id_by_title(CORPUS)
    conflit_id =  conflit_id = str(get_conflit_id())  

    for video in video_collection.find({}, {"_id": 0, "videos.commentaires.langue": 1, "videos.commentaires.script": 1, "videos.commentaires.descripteur": 1, "videos.corpus": 1}):
        for video_item in video.get('videos', []):
            if video_item.get('corpus') == corpus_id:
                for commentaire in video_item.get('commentaires', []):
                    descripteur = commentaire.get('descripteur', [])
                    script = commentaire.get('script', '')
                    if conflit_id in descripteur:
                        if script in script_counts:
                            script_counts[script] += 1

    total_comments = sum(script_counts.values())
    frequencies = {script: (count / total_comments) if total_comments > 0 else 0 
                   for script, count in script_counts.items()}

    return list(frequencies.values())


def get_titles_from_corpus():
    collection = db['corpus']  
    titles = collection.find({}, {"title": 1, "_id": 0})  
    title_list = [doc['title'] for doc in titles if 'title' in doc]
    
    return title_list


def get_corpus_id_by_title(title):
    corpus_collection = db['corpus']
    corpus = corpus_collection.find_one({'title': title})
    if corpus:
        return str(corpus['_id'])
    return None

def freq_lang(script_value):
    video_collection = db["video_collection"]
    langues = []
    corpus_id = get_corpus_id_by_title(CORPUS)
    conflit_id = str(get_conflit_id()) 
    
    for video in video_collection.find({}, {"_id": 0, "videos.commentaires.langue": 1, "videos.commentaires.script": 1, "videos.commentaires.descripteur": 1, "videos.corpus": 1}):
        for video_item in video.get('videos', []):
            if video_item.get('corpus') == corpus_id:
                for commentaire in video_item.get('commentaires', []):
                    descripteur = commentaire.get('descripteur', [])
                    if conflit_id in descripteur and commentaire.get('script') == script_value:
                        langues.append(commentaire.get('langue', []))

    if not langues:
        return []

    langue_sets = [set(commentaire_langues) for commentaire_langues in langues]
    all_langues = sorted(set(langue for sublist in langues for langue in sublist))

    def generate_combinations(langues):  
        combos = []
        for i in range(1, len(langues) + 1):
            for combo in combinations(langues, i):
                combos.append(frozenset(combo))
        return combos

    combinations_list = generate_combinations(all_langues)
    combination_counter = Counter()
    
    for lang_set in langue_sets:
        for combo in combinations_list:
            if combo == lang_set:
                combination_counter[combo] += 1

    total_comments = len(langue_sets)
    result = []

    for combo in combinations_list:
        combo_str = ' + '.join(sorted(combo))
        count = combination_counter.get(combo, 0)
        frequency = count / total_comments if total_comments > 0 else 0
        result.append({"label": combo_str, "frequency": frequency})
    
    return result



def get_etiquettes():
    video_collection = db["video_collection"]
    etiquette_collection = db["etiquette"]
    global CORPUS
    corpus_id = get_corpus_id_by_title(CORPUS)
    
    if not corpus_id:
        return [] 

    descripteurs = video_collection.find({}, {'videos.commentaires.descripteur': 1, 'videos.corpus': 1})
    ids = set()
    for video in descripteurs:
        videos = video.get('videos', [])
        for video_item in videos:
            if video_item.get('corpus')==corpus_id:
                commentaires = video_item.get('commentaires', [])
                for commentaire in commentaires:
                    descripteur = commentaire.get('descripteur', [])
                    ids.update(descripteur)

    ids_list = [ObjectId(id) for id in ids if ObjectId.is_valid(id)] 

    if not ids_list:
        return []
    etiquettes = etiquette_collection.find({'_id': {'$in': ids_list}})
    etiquette_dict = {}
    for etiquette in etiquettes:
        _id = str(etiquette['_id'])
        name = etiquette.get('name', None)
        if name:  
            etiquette_dict[_id] = name

    return list(etiquette_dict.values())

def get_liste_corpus_view(request):
    if request.method == 'GET':
        corpus = get_titles_from_corpus()
        return JsonResponse(corpus, safe=False)
    return JsonResponse({'error': 'Invalid request method'}, status=405)

def corpus_choisi(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        selected_option = data.get('selectedOption')
        global CORPUS
        CORPUS = selected_option
        return JsonResponse({'message': 'Option received', 'selectedOption': selected_option})
    
    return JsonResponse({'error': 'Invalid request'}, status=400)


def get_etiquettes_view(request):
    if request.method == 'GET':
        etiquettes = get_etiquettes()  
        return JsonResponse(etiquettes, safe=False)
    return JsonResponse({'error': 'Invalid request method'}, status=405)

def get_mots_view(request):
    if request.method == 'GET':
        mots = get_mots()  
        return JsonResponse(mots, safe=False)
    return JsonResponse({'error': 'Invalid request method'}, status=405)
    

def process_combinations(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            selected_combinations = data.get('selectedCombinations', [])

            modified_combinations_f= []
            modified_combinations_p= []

            for combination in selected_combinations:
                types_list = combination.get('types', [])
                frequency,percentage,total_desc = calculate_freq_perc(types_list)
                modified_combinations_f.append(frequency)
                modified_combinations_p.append(percentage)
                
            return JsonResponse({"frequencies": modified_combinations_f, "percentages": modified_combinations_p, "total": total_desc})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({"error": "Method not allowed"}, status=405)

JSON_FILE_PATH = 'validation/videos.json'

def load_json(request):
    try:
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as file:
            video_data = json.load(file)
            print("JSON data loaded successfully")
    except FileNotFoundError:
        print("File not found")
        return JsonResponse({'error': 'File not found'}, status=404)
    except json.JSONDecodeError:
        print("Error decoding JSON")
        return JsonResponse({'error': 'Error decoding JSON'}, status=400)

    paginator = Paginator(video_data, 10)  # Show 10 videos per page
    page_number = request.GET.get('page')
    try:
        page_obj = paginator.get_page(page_number)
    except Exception as e:
        print(f"Pagination error: {e}")
        return JsonResponse({'error': 'Pagination error'}, status=400)

    serialized_video_data = list(page_obj)

    return JsonResponse(serialized_video_data, safe=False)
 
@require_POST
def submit_validation(request):
    try:
        raw_post_data = request.body.decode('utf-8')
        try:
            data = json.loads(raw_post_data)
            action = data.get('action')
            video_data_str = data.get('video_data')
            try:
                video_data = json.loads(video_data_str) if isinstance(video_data_str, str) else video_data_str
            except json.JSONDecodeError as e:
                print(f"Video Data Parsing Error: {str(e)}")
                return JsonResponse({'error': f'Invalid video_data JSON: {str(e)}'}, status=400)
        except json.JSONDecodeError as e:
            print(f"JSON Parsing Error: {str(e)}")
            return JsonResponse({'error': f'Invalid JSON data: {str(e)}'}, status=400)

        if not action or not video_data:
            return JsonResponse({'error': 'Missing action or video_data in request'}, status=400)

        video_collection = VideoCollection.objects.first()
        if not video_collection:
            video_collection = VideoCollection(videos=[])

        if action == 'validate':
            for idx, video in enumerate(video_data):
                try:
                    if not isinstance(video, dict):
                        print(f"Invalid data type at index {idx}: {type(video)}")
                        raise ValueError("Each video entry should be a dictionary")

                    video_entry = Video(
                        id_video=video.get('id_video'),
                        titre_video=video.get('titre_video'),
                        description_video=video.get('description_video'),
                        hashtags=video.get('hashtags'),
                        date_publication=datetime.strptime(video.get('date_publication'), "%d-%m-%Y"),
                        lien_video=video.get('lien_video'),
                        annotation_video=video.get('annotation_video'),
                        commentaires=[
                            Commentaire(
                                id_commentaire=str(uuid.uuid4()),  # Generate a new unique ID for the comment
                                texte=comment.get('texte_commentaire'),
                                date_publication=datetime.strptime(comment.get('date_publication'), "%d-%m-%Y"),
                                auteur=comment.get('nom_utilisateur'),
                                langue=[comment.get('langue')] if comment.get('langue') else [],  # Ensure this is a list
                                descripteur=[]
                            ) for comment in video.get('commentaires', [])
                        ],
                        is_valid=True
                    )
                    
                    video_collection.videos.append(video_entry)
                except Exception as e:
                    print(f"Error processing video entry at index {idx}: {str(e)}")
                    return JsonResponse({'error': f'Error processing video entry at index {idx}: {str(e)}'}, status=500)
            video_collection.save()

        elif action == 'delete':
            try:
                video_ids_to_delete = {video.get('id_video') for video in video_data if isinstance(video, dict)}
                video_collection.videos = [
                    video for video in video_collection.videos
                    if video.id_video not in video_ids_to_delete
                ]
                video_collection.save()
            except Exception as e:
                print(f"Error deleting video entry: {str(e)}")
                return JsonResponse({'error': f'Error deleting video entry: {str(e)}'}, status=500)
        else:
            return JsonResponse({'error': 'Invalid action'}, status=400)

        return redirect('load_json')

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return JsonResponse({'error': f'An unexpected error occurred: {str(e)}'}, status=500)
     
 
@api_view(['GET'])
def get_comments(request):
    video_collection = VideoCollection.objects.first()
    comments = []

    if video_collection:
        for video in video_collection.videos:
            for comment in video.commentaires:
                comments.append({
                    'id_commentaire': comment.id_commentaire,
                    'texte': comment.texte,
                    'date_publication': comment.date_publication.strftime('%Y-%m-%d'),
                    'auteur': comment.auteur,
                    'langue': comment.langue,
                    'descripteur': comment.descripteur,
                })
    
    return JsonResponse(comments, safe=False)

@api_view(['POST'])
def update_comments(request):
    data = request.data
    comment_id = data.get('id_commentaire')
    new_text = data.get('texte')  # Example: if you want to update the text

    video_collection = VideoCollection.objects.first()
    if video_collection:
        for video in video_collection.videos:
            for comment in video.commentaires:
                if comment.id_commentaire == comment_id:
                    comment.texte = new_text  # Update the field accordingly
                    video_collection.save()
                    return JsonResponse({'status': 'success'})

    return JsonResponse({'status': 'error', 'message': 'Comment not found'}, status=404)

@api_view(['GET', 'POST'])
def manage_comments(request):
    if request.method == 'GET':
        collection = VideoCollection.objects.first()
        comments = [c for v in collection.videos for c in v.commentaires if 'conflict' in c.descripteur]  # Example condition
        return Response(comments)

    elif request.method == 'POST':
        comment_id = request.data.get('id_commentaire')
        new_descripteur = request.data.get('descripteur')

        collection = VideoCollection.objects.first()
        # Fetch language etiquettes dynamically from the database
        language_etiquettes = db.etiquette.find({"category": "66df0a114a1a2231ea4ed560"})
        id_to_name = {str(lang['_id']): lang['name'] for lang in language_etiquettes}


        for video in collection.videos:
            for comment in video.commentaires:
                if comment.id_commentaire == comment_id:
                    comment.descripteur = new_descripteur  # Example of what to update

                    # Filter out the language etiquettes from the descripteurs
                    langues = [id_to_name[desc_id] for desc_id in new_descripteur if desc_id in id_to_name]

                    # If there are any languages, update the langue field
                    if langues:
                        comment.langue = langues

                    collection.save()
                    return Response({'success': True})

        return Response({'success': False}, status=status.HTTP_400_BAD_REQUEST)

# @api_view(['POST'])
# def update_emotion(request):
#     try:
#         comment_id = request.data.get('comment_id')
#         emotions = request.data.get('emotion', [])
#         print("bbbbbbbbbbbbbbbbbbbb",emotions)
#         # Find and update the comment
#         video_collection = VideoCollection.objects.first()  # Adjust this according to your logic
#         for video in video_collection.videos:
#             for comment in video.commentaires:
#                 if comment.id_commentaire == comment_id:
#                     print("hhhhhh")
#                     comment.emotion = emotions
#                     print(comment.auteur)
#                     print(comment.emotion)
#                     video_collection.save()
#                     print("videosssss")
#                     print(video)
#                     return Response({'success': True})

#         return Response({'success': False, 'message': 'Comment not found'})
#     except Exception as e:
#         return Response({'success': False, 'message': str(e)})
    




 
@api_view(['GET', 'POST', 'PUT', 'DELETE'])
def manage_emotions(request, emotion_id=None):
    if request.method == 'GET':
        etiquettes = list(Etiquette.objects.all().only('id', 'name', 'color', 'category'))
        return Response([{'id': str(etiquette.id), 'name': etiquette.name, 'color': etiquette.color, 'category': etiquette.category} for etiquette in etiquettes])

    elif request.method == 'POST':
        data = request.data
        etiquette = Etiquette(name=data['name'], color=data['color'], category=data['category'])
        etiquette.save()
        return Response({'id': str(etiquette.id), 'name': etiquette.name, 'color': etiquette.color, 'category': etiquette.category})

    elif request.method == 'PUT':
        data = request.data
        etiquette = Etiquette.objects.get(id=data['id'])
        etiquette.name = data['name']
        etiquette.color = data['color']
        etiquette.save()
        return Response({'id': str(etiquette.id), 'name': etiquette.name, 'color': etiquette.color})

    elif request.method == 'DELETE':
        if not emotion_id:
            return Response({'error': 'Etiquette ID is required for deletion'}, status=400)

        try:
            etiquette = Etiquette.objects.get(id=emotion_id)
            
            if VideoCollection.objects.filter(videos__commentaires__descripteur=emotion_id).count() > 0:
                return Response({'error': 'Etiquette is used in a descripteur and cannot be deleted'}, status=400)
            
            etiquette.delete()
            return Response({'message': 'Etiquette deleted successfully'})
        except Etiquette.DoesNotExist:
            return Response({'error': 'Etiquette not found'}, status=404)
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import Category, Corpus, Etiquette  # Updated Emotion to Etiquette

@api_view(['GET'])
def get_comments(request):
    video_collection = VideoCollection.objects.first()
    comments = []

    if video_collection:
        for video in video_collection.videos:
            for comment in video.commentaires:
                comments.append({
                    'id_commentaire': comment.id_commentaire,
                    'texte': comment.texte,
                    'date_publication': comment.date_publication.strftime('%Y-%m-%d'),
                    'auteur': comment.auteur,
                    'langue': comment.langue,
                    'descripteur': comment.descripteur,
                })
    
    return JsonResponse(comments, safe=False)

@api_view(['POST'])
def update_comments(request):
    data = request.data
    comment_id = data.get('id_commentaire')
    descripteur = data.get('descripteur')

    video_collection = VideoCollection.objects.first()
    if video_collection:
        for video in video_collection.videos:
            for comment in video.commentaires:
                if comment.id_commentaire == comment_id:
                    comment.descripteur = descripteur
                    video_collection.save()
                    return JsonResponse({'status': 'success'})

    return JsonResponse({'status': 'error', 'message': 'Comment not found'}, status=404)

@api_view(['GET', 'POST', 'PUT', 'DELETE'])
def manage_categories(request, id=None):
    if request.method == 'GET':
        categories = Category.objects.all()
        categories_data = []
        for category in categories:
            etiquettes = Etiquette.objects.filter(id__in=category.emotions)  # Updated from Emotion to Etiquette
            etiquettes_data = [
                {'id': str(etiquette.id), 'name': etiquette.name, 'color': etiquette.color}
                for etiquette in etiquettes
            ]
            category_data = {
                'id': str(category.id),
                'name': category.name,
                'emotions': etiquettes_data,
                'multiple_choice': category.multiple_choice
            }
            categories_data.append(category_data)
        return Response(categories_data)

    elif request.method == 'POST':
        data = request.data
        category = Category(
            name=data['name'],
            multiple_choice=data.get('multiple_choice', False)
        )
        category.save()
        return Response({
            'id': str(category.id),
            'name': category.name,
            'emotions': [],
            'multiple_choice': category.multiple_choice
        }, status=status.HTTP_201_CREATED)

    elif request.method == 'PUT':
        data = request.data
        category = Category.objects.get(id=data['id'])
        category.name = data['name']
        category.multiple_choice = data.get('multiple_choice', category.multiple_choice)
        category.save()
        return Response({
            'id': str(category.id),
            'name': category.name,
            'multiple_choice': category.multiple_choice
        })

    elif request.method == 'DELETE':
        if id is None:
            return Response({'error': 'Category ID is required'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            category = Category.objects.get(id=id)
            
            # Check if the category is used in any Corpus
            corpus_with_category = Corpus.objects.filter(categories=category).first()
            if corpus_with_category:
                return Response(
                    {'error': 'Cannot delete the category as it is used in a corpus.'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            category.delete()
            return Response({'message': 'Category deleted successfully'}, status=status.HTTP_204_NO_CONTENT)

        except Category.DoesNotExist:
            return Response({'error': 'Category not found'}, status=status.HTTP_404_NOT_FOUND)

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def add_emotion_to_category(request):  # Updated from add_emotion_to_category to add_etiquette_to_category
    data = request.data
    category = Category.objects.get(id=data['category_id'])
    
    etiquette = Etiquette(
        name=data['name'],
        color=data['color'],
        category=str(category.id)  # Store the ID as a string
    )
    etiquette.save()
    
    # Append only the ID of the etiquette
    category.emotions.append(str(etiquette.id))
    category.save()
    
    return Response({
        'id': str(etiquette.id),
        'name': etiquette.name,
        'color': etiquette.color
    }, status=status.HTTP_201_CREATED)

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from  .models import Corpus, Category
from bson import ObjectId

from bson import ObjectId
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

@api_view(['GET', 'POST', 'DELETE','PUT'])
def manage_corpus(request, corpus_id=None):
    if request.method == 'GET':
        # Fetch all Corpus objects
        corpus_list = Corpus.objects.all()
        
        # Prepare response data
        data = []

        for corpus in corpus_list:
            # Fetch all categories in a single query
            category_ids = [ObjectId(cat_ref.id) for cat_ref in corpus.categories]
            categories = Category.objects(id__in=category_ids)
            category_dict = {str(cat.id): {'id': str(cat.id), 'name': cat.name} for cat in categories}

            # Map categories
            categories = [category_dict.get(str(cat_ref.id), {}) for cat_ref in corpus.categories]

            videos = []
            for video in corpus.videos:
                commentaires = [
                    {
                        'id_commentaire': comment.id_commentaire,
                        'texte': comment.texte,
                        'date_publication': comment.date_publication,
                        'auteur': comment.auteur,
                        'langue': comment.langue,
                        'descripteur': comment.descripteur
                    }
                    for comment in video.commentaires
                ]

                videos.append({
                    'id_video': video.id_video,
                    'titre_video': video.titre_video,
                    'description_video': video.description_video,
                    'hashtags': video.hashtags,
                    'date_publication': video.date_publication,
                    'lien_video': video.lien_video,
                    'annotation_video': video.annotation_video,
                    'commentaires': commentaires
                })

            data.append({
                'id': str(corpus.id),
                'title': corpus.title,
                'description': corpus.description,
                'categories': categories,
                'videos': videos
            })
        print("ffffffffffff")
        return Response(data)
    elif request.method == 'POST':
        data = request.data
        try:
            categories = [Category.objects.get(id=cat_id) for cat_id in data['categories']]
        except Category.DoesNotExist:
            return Response({'error': 'One or more categories do not exist'}, status=status.HTTP_400_BAD_REQUEST)
        
        corpus = Corpus(
            title=data['title'],
            description=data.get('description', ''),  # Handle the description field
            categories=categories
        )
        corpus.save()
        response_data = {
            'id': str(corpus.id),
            'title': corpus.title,
            'description': corpus.description,  # Include the description field
            'categories': [{'id': str(cat.id), 'name': cat.name} for cat in corpus.categories],
            'videos': []  # Adjust as needed
        }
        return Response(response_data, status=status.HTTP_201_CREATED)

    elif request.method == 'DELETE' and corpus_id:
        try:
            corpus_id = ObjectId(corpus_id)
            
            # Check if any video in VideoCollection references this corpus
            video_with_corpus = VideoCollection.objects.filter(videos__corpus=str(corpus_id)).first()
            if video_with_corpus:
                return JsonResponse({"error": "Cannot delete corpus. It is assigned to one or more videos."}, status=400)

            # Find and delete the corpus if no video references it
            corpus = Corpus.objects.get(id=corpus_id)
            
            # Optional: Check if corpus has videos before deleting
            if corpus.videos:
                return JsonResponse({"error": "Cannot delete corpus with associated videos. Please delete or reassign videos first."}, status=400)
            
            corpus.delete()

            return JsonResponse({"success": "Corpus deleted successfully"}, status=204)

        except DoesNotExist:
            return JsonResponse({"error": "Corpus not found"}, status=404)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    elif request.method == 'PUT' and corpus_id:
        # Handle the update logic
        try:
            data = request.data
            print("Received data:", data)
            title = data.get('title')
            description = data.get('description')
            categories = data.get('categories')

            if not title or not description or not categories:
                return Response({"error": "Missing title, description, or categories"}, status=status.HTTP_400_BAD_REQUEST)

            # Ensure categories are valid
            try:
                category_objects = [Category.objects.get(id=ObjectId(cat_id)) for cat_id in categories]
            except Category.DoesNotExist:
                return Response({"error": "One or more categories do not exist"}, status=status.HTTP_400_BAD_REQUEST)

            # Find and update the Corpus
            corpus = Corpus.objects.get(id=ObjectId(corpus_id))
            corpus.title = title
            corpus.description = description
            corpus.categories = category_objects
            corpus.save()

            return Response({"success": "Corpus updated successfully"}, status=status.HTTP_200_OK)
        except Corpus.DoesNotExist:
            return Response({"error": "Corpus not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    else:
        return Response({'error': 'Method not allowed'}, status=status.HTTP_405_METHOD_NOT_ALLOWED)
    


 
# from django.http import JsonResponse
# from django.views.decorators.http import require_http_methods
# import json
# from .models import Corpus, Category
# from bson import ObjectId
# from django.core.exceptions import ObjectDoesNotExist
# @require_http_methods(["POST", "PUT"])
# def update_corpus(request, corpus_id):
#     try:
#         data = json.loads(request.body)
#         corpus_id = data.get('corpus')
 
#         # Update the video with the new corpus
#         try:
#             print("video gotten")
#             video = Video.objects.get(id=video.id)
#             print("video gotten",video)
#             video.corpus = corpus_id
#             VideoCollection.save()
#         except ObjectDoesNotExist:
#             return JsonResponse({"error": "Video not found"}, status=404)

#         return JsonResponse({"success": "Video corpus updated successfully"}, status=200)
#     except json.JSONDecodeError:
#         return JsonResponse({"error": "Invalid JSON"}, status=400)
 
#     except Corpus.DoesNotExist:
#         return JsonResponse({"error": "Corpus not found"}, status=404)
#     except Exception as e:
#         return JsonResponse({"error": str(e)}, status=500)
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
import json
from bson import ObjectId
from mongoengine.errors import DoesNotExist

@require_http_methods(["PUT"])
def update_corpus(request, corpus_id):
    try:
        data = json.loads(request.body)
        video_id = data.get('id_video')
        corpus_id = ObjectId(corpus_id)
        
        # Find the corpus
        try:
            corpus = Corpus.objects.get(id=corpus_id)
            print(f"Corpus found: {corpus.title}")
        except DoesNotExist:
            return JsonResponse({"error": "Corpus not found"}, status=404)

        # Find the video in the VideoCollection
        try:
            video_collection = VideoCollection.objects.get(videos__id_video=video_id)
            video = next((v for v in video_collection.videos if v.id_video == video_id), None)
            if not video:
                return JsonResponse({"error": f"Video with ID {video_id} not found in VideoCollection"}, status=404)
            print(f"Video found: {video.titre_video}")

            # Update the video's corpus field
            video.corpus = str(corpus_id)
            video_collection.save()

            # Add the video to the Corpus's videos field if not already present
            if not any(v.id_video == video_id for v in corpus.videos):
                corpus.videos.append(video)
                corpus.save()

            return JsonResponse({"success": "Video corpus updated and added to Corpus"}, status=200)

        except DoesNotExist:
            return JsonResponse({"error": "VideoCollection not found"}, status=404)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

from mongoengine import DoesNotExist
def get_videos_by_corpus(request):
    corpus_id = request.GET.get('corpus_id')
    if not corpus_id:
        return JsonResponse({'error': 'No corpus ID provided'}, status=400)
    
    try:
        # Assuming you have only one VideoCollection document
        video_collection = VideoCollection.objects.first()
        if not video_collection:
            return JsonResponse({'error': 'No video collection found'}, status=404)
        
        # Filter videos by the corpus ID
        filtered_videos = [video for video in video_collection.videos if video.corpus == corpus_id]
        
        # Convert to dictionary
        videos_list = [video.to_mongo().to_dict() for video in filtered_videos]
        
        return JsonResponse(videos_list, safe=False)
    except DoesNotExist:
        return JsonResponse({'error': 'Corpus not found'}, status=404)
    


@api_view(['PATCH'])
def update_comment_descriptor(request):
    data = request.data
    comment_id = data.get('id_commentaire') 
    new_descriptors = data.get('descripteur')  # New descriptors to add

    if not comment_id or not isinstance(new_descriptors, list):
        return JsonResponse({"error": "Invalid comment ID or descriptors format"}, status=400)

    try:
        video_collection = VideoCollection.objects.first()
        if not video_collection:
            return JsonResponse({"error": "No video collection found"}, status=404)

        # Fetch language etiquettes dynamically 
        language_etiquettes = db.etiquette.find({"category": "66df0a114a1a2231ea4ed560"})
        id_to_name = {str(lang['_id']): lang['name'] for lang in language_etiquettes}
        
        categories = db.category.find()
        single_choice_categories = {}
        multiple_choice_categories = {}
        
        # We iterate over categories and classify them based on 'multiple_choice' field
        for cat in categories:
            category_id = str(cat['_id'])
            if cat['multiple_choice']:
                multiple_choice_categories[category_id] = cat
            else:
                single_choice_categories[category_id] = cat
        
    
        def get_category_id(descriptor_id):
            """map descriptor_id to its category"""
            descriptor = db.etiquette.find_one({"_id": ObjectId(descriptor_id)})
            if descriptor:
                category_id = str(descriptor['category'])
                return category_id
            else:
                return None
        
        for video in video_collection.videos:
            for comment in video.commentaires:
                if comment.id_commentaire == comment_id:
                    existing_descriptors = comment.descripteur or []
                    descriptor_map = {}  # this is a dictionary to store descriptors by category
                    
                    if existing_descriptors:
                        # We classify existing descriptors into single or multiple-choice categories
                        for descriptor in existing_descriptors:
                            category_id = get_category_id(descriptor)  # map descriptor to category
                            if category_id in single_choice_categories:
                                descriptor_map[category_id] = descriptor  # Single-choice, We replace directly
                            elif category_id in multiple_choice_categories:
                                descriptor_map[category_id] = []  
                 
                    # We process new descriptors
                    for new_descriptor in new_descriptors:
                        category_id = get_category_id(new_descriptor)
                        if category_id in single_choice_categories:
                            # Replace single-choice category descriptor
                            descriptor_map[category_id] = new_descriptor
                        elif category_id in multiple_choice_categories:
                            # Add to multiple-choice category descriptors
                            if not descriptor_map.get(category_id):
                                descriptor_map[category_id] = [] 
                            descriptor_map[category_id].append(new_descriptor)
                               
                    updated_descriptors = []
                    for desc_list in descriptor_map.values():
                        if isinstance(desc_list, list):
                            updated_descriptors.extend(desc_list)  # Combine all descriptors into a single list
                        else:
                            updated_descriptors.append(desc_list)  # Single descriptor

                    
                    
                    comment.descripteur = updated_descriptors  

                    # Filter out language etiquettes from the descriptors
                    langues = [id_to_name.get(desc_id) for desc_id in updated_descriptors if desc_id in id_to_name]

                    # If languages were found, update the langue field
                    if langues:
                        comment.langue = langues
                    
                    video_collection.save()
                    
                    return JsonResponse({"success": "Comment descriptor updated successfully"}, status=200)
        
        return JsonResponse({"error": "Comment not found"}, status=404)
    
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)





    



@api_view(['DELETE'])
def delete_comment(request):
    comment_id = request.data.get('id_commentaire')

    if not comment_id:
        return JsonResponse({"error": "Comment ID is required"}, status=400)

    try:
        video_collection = VideoCollection.objects.first()
        if not video_collection:
            return JsonResponse({"error": "No video collection found"}, status=404)

        for video in video_collection.videos:
            comment_to_delete = next((comment for comment in video.commentaires if comment.id_commentaire == comment_id), None)
            if comment_to_delete:
                video.commentaires.remove(comment_to_delete)
                video_collection.save()
                return JsonResponse({"success": "Comment deleted successfully"}, status=200)

        return JsonResponse({"error": "Comment not found"}, status=404)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
@api_view(['DELETE'])
def delete_video(request):
    video_id = request.data.get('id_video')

    if not video_id:
        return JsonResponse({"error": "Video ID is required"}, status=400)

    try:
        video_collection = VideoCollection.objects.first()
        if not video_collection:
            return JsonResponse({"error": "No video collection found"}, status=404)

        video_to_delete = next((video for video in video_collection.videos if video.id_video == video_id), None)
        if video_to_delete:
            video_collection.videos.remove(video_to_delete)
            video_collection.save()
            return JsonResponse({"success": "Video deleted successfully"}, status=200)

        return JsonResponse({"error": "Video not found"}, status=404)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

########################################################################################################################
########################################################################################################################
#                                  | cleaning endpoints |
########################################################################################################################
########################################################################################################################
# importing files & liberies
#-----------------------------------------------
from .cleaning_module import *
from rest_framework.decorators import api_view
from rest_framework.views import APIView
from rest_framework import status
from rest_framework.response import Response
#-----------------------------------------------
from django.http import JsonResponse
from django.shortcuts import render
#-----------------------------------------------
import json  
#-----------------------------------------------
# display the clusters
#--------------------------------------------------------------------------------
def display_clusters(request):
    # Récupérer tous les clusters de la collection 'clusters'
    clusters = list(db.clusters.find({}))

    # Passer les clusters au template
    return render(request, 'display_clusters.html', {'clusters': clusters})
#--------------------------------------------------------------------------------
# BaseCleaningText endpoint
#--------------------------------------------------------------------------------
class BaseCleaningText(APIView):
    def post(self, request, *args, **kwargs):
        try:   
            message = filtrage_auto()
            return Response({'cpt':message}, status=status.HTTP_200_OK)  
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
#--------------------------------------------------------------------------------
# view-comment-in the commentaires_nettoyés collection  endpoint
#--------------------------------------------------------------------------------
def comment_after(request):
    # Fetch the first VideoCollection document
    video_documents = db.video_collection.find()

    # Extract comments from the collection
    comments = []
    for video_document in video_documents:
        for video in video_document.get('videos', []):
            for comment in video.get('commentaires', []):
                comments.append(comment)
    # Render the template with comments
    number_comments = len(comments)
    return render(request, 'view_comments.html', {'comments': comments,'number_comments':number_comments})
#--------------------------------------------------------------------------------
def comment_before(request):
    # Fetch the first VideoCollection document
    video_documents = db.cleaning.find()

    # Extract comments from the collection
    comments = []
    for video_document in video_documents:
        for video in video_document.get('videos', []):
            for comment in video.get('commentaires', []):
                comments.append(comment)
    # Render the template with comments
    number_comments = len(comments)
    return render(request, 'view_comments.html', {'comments': comments,'number_comments':number_comments})

#--------------------------------------------------------------------------------
# DeepCleaningText endpoint
#--------------------------------------------------------------------------------
from concurrent.futures import ThreadPoolExecutor, as_completed

class DeepCleaningText(APIView):
    def post(self, request, *args, **kwargs):
        criteria = request.data.get('criteria', [])
        criteria = list(dict.fromkeys(criteria))
        print(f"Received Criteria: {criteria}")
        if not criteria:
            return Response({'message': 'No operations selected'}, status=status.HTTP_200_OK)
        processed_comments = []
        raw_and_cleaned = []
        video_documents = list(db.video_collection.find({}, {
                  '_id': 1,  
                  'videos.id_video': 1,  
                  'videos.commentaires.id_commentaire': 1,  
                  'videos.commentaires.texte': 1,  
                  'videos.commentaires.descripteur': 1,  
                  'videos.commentaires.cleaned': 1  
        }))
        total_comments = sum(len(video.get('commentaires', [])) for video_document in video_documents for video in video_document.get('videos', []))
        processed_count = 0
        conflit_etq_id = '66df07cf4a1a2231ea4ed55e'

        def process_comments(video_document):
            local_processed_comments = []
            local_raw_and_cleaned = []
            local_processed_count = 0

            try:
                for video in video_document.get('videos', []):
                    id_video = video.get('id_video')
                    filtred_comments = video.get('commentaires', [])

                    for comment in filtred_comments:
                        descripteurs = comment.get('descripteur', [])
                        if conflit_etq_id in descripteurs:
                            if detect_script(str(comment['texte'])) != 'Arabic Script':
                                criteria_appliquer = []
                                for crit in criteria:
                                    found = False
                                    for item in comment['cleaned']:
                                        if item[0] == crit:
                                            found = True
                                            if not item[1]:
                                                criteria_appliquer.append(crit)
                                            break

                                    if not found:
                                        criteria_appliquer.append(crit)
                                        comment['cleaned'].append([crit, False])

                                comment_text_before = str(comment['texte'])
                                cleaned_comment_text_after = apply_criteria(comment_text_before, criteria_appliquer)

                                if criteria_appliquer:
                                    comment['texte'] = cleaned_comment_text_after
                                    for crit in criteria_appliquer:
                                        for item in comment['cleaned']:
                                            if item[0] == crit:
                                                item[1] = True
                                    local_processed_comments.append(comment)
                                    local_raw_and_cleaned.append({
                                        'before': comment_text_before,
                                        'after': comment['texte']
                                    })

                                    filter = {
                                        '_id': video_document['_id'],
                                        'videos.id_video': id_video,
                                        'videos.commentaires.id_commentaire': comment['id_commentaire']
                                    }
                                    update = {
                                        '$set': {
                                            'videos.$.commentaires.$[comment].texte': comment['texte'],
                                            'videos.$.commentaires.$[comment].cleaned': comment['cleaned'],
                                        }
                                    }
                                    array_filters = [{'comment.id_commentaire': comment['id_commentaire']}]
                                    db.video_collection.update_one(filter, update, array_filters=array_filters)

                                    local_processed_count += 1
                                    progress = int((local_processed_count / total_comments) * 100) 
                                    print(f"progress : {progress} % | Processed Comment : {local_processed_count}")
            except Exception as e:
                print(f"Error processing video_document: {e}")

            return {
                'processed_comments': local_processed_comments,
                'raw_and_cleaned': local_raw_and_cleaned,
                'processed_count': local_processed_count,
            }

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_comments, video_document) for video_document in video_documents]
            for future in as_completed(futures):
                result = future.result()
                processed_comments.extend(result['processed_comments'])
                raw_and_cleaned.extend(result['raw_and_cleaned'])
                processed_count += result['processed_count']

        return Response({
            'criteria': criteria,
            'raw_and_cleaned': raw_and_cleaned,
            'processed_count': processed_count,
            'total_comments': total_comments
        }, status=status.HTTP_200_OK)
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------           

import os
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

# scraper/views.py
from django.shortcuts import render
from .forms import KeywordForm, URLForm
from googleapiclient.discovery import build
import youtube_dl
import json
from django.http import JsonResponse
from django.shortcuts import render
from yt_dlp import YoutubeDL
from .forms import KeywordForm, URLForm
 


from datetime import datetime

API_KEY = "AIzaSyCQSCrGIedJGhuXJRS60vGhP2ceYAVQ-vA"
 



"""
def search_videos(request):
    videos = []
    if request.method == "POST":
        form = KeywordForm(request.POST)
        if form.is_valid():
            keyword = form.cleaned_data["keyword"]
            youtube = build("youtube", "v3", developerKey=API_KEY)
            youtube_request = youtube.search().list(
                q=keyword, part="snippet", maxResults=10
            )  # Renamed variable
            response = youtube_request.execute()  # Using the renamed variable
            videos = response.get("items", [])
    else:
        form = KeywordForm()
    return render(request, "scraper/search.html", {"form": form, "videos": videos}) """



@csrf_exempt
def search_videos(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body.decode('utf-8'))
            keyword = data.get('keyword')
            if not keyword:
                return JsonResponse({'error': 'Keyword is required'}, status=400)

            youtube = build("youtube", "v3", developerKey=API_KEY)
            youtube_request = youtube.search().list(
                q=keyword, part="snippet", maxResults=200
            )
            response = youtube_request.execute()
            videos = response.get("items", [])

            return JsonResponse({'videos': videos})

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=405)


"""def scrape_video(request):
    data = {}
    if request.method == "POST":
        form = URLForm(request.POST)
        if form.is_valid():
            url = form.cleaned_data["url"]
            ydl_opts = {
                "skip_download": True,
                "writesubtitles": True,
                "writeautomaticsub": True,
                "writeinfojson": True,
                "getcomments": True,
            }
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)

                # Extracting the required information
                titre_video = info.get("title")
                description_video = info.get("description")
                hashtags = info.get("tags", [])
                date_publication = info.get("upload_date")
                if date_publication:
                    date_publication = datetime.strptime(
                        date_publication, "%Y%m%d"
                    ).strftime("%d-%m-%Y")
                lien_video = url

                # Extract comments if available
                comments = []
                if "comments" in info:
                    for comment in info["comments"]:
                        # Convert timestamp to DD-MM-YYYY format
                        timestamp = comment.get("timestamp")
                        if timestamp:
                            date_publication_comment = datetime.fromtimestamp(
                                timestamp
                            ).strftime("%d-%m-%Y")
                        else:
                            date_publication_comment = None

                        comments.append(
                            {
                                "nom_utilisateur": comment.get("author"),
                                "date_publication": date_publication_comment,
                                "texte_commentaire": comment.get("text"),
                                "langue": comment.get("language"),
                            }
                        )

                data = {
                    "titre_video": titre_video,
                    "description_video": description_video,
                    "hashtags": hashtags,
                    "date_publication": date_publication,
                    "lien_video": lien_video,
                    "nombre_commentaires": len(comments),
                    "commentaires": comments,
                }

                # Check if the JSON file exists and load existing data
                if os.path.exists(JSON_FILE_PATH):
                    if os.path.getsize(JSON_FILE_PATH) > 0:
                        with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
                            existing_data = json.load(f)
                    else:
                        existing_data = []
                else:
                    existing_data = []

                # Append the new data
                existing_data.append(data)

                # Save the updated data back to the JSON file
                with open(JSON_FILE_PATH, "w", encoding="utf-8") as f:
                    json.dump(existing_data, f, ensure_ascii=False, indent=4)

    else:
        form = URLForm()

    if request.method == "POST":
        return JsonResponse(data)
    else:
        return render(request, "scraper/scrape.html", {"form": form})"""

def scrape_video(request):
    data = []
    if request.method == "POST":
        body = json.loads(request.body)
        urls = body.get("urls", [])

        for url in urls:
            ydl_opts = {
                "skip_download": True,
                "writesubtitles": True,
                "writeautomaticsub": True,
                "writeinfojson": True,
                "getcomments": True,
            }
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)

                titre_video = info.get("title")
                description_video = info.get("description")
                hashtags = info.get("tags", [])
                date_publication = info.get("upload_date")
                if date_publication:
                    date_publication = datetime.strptime(date_publication, "%Y%m%d").strftime("%d-%m-%Y")
                lien_video = url

                comments = []
                if "comments" in info:
                    for comment in info["comments"]:
                        timestamp = comment.get("timestamp")
                        if timestamp:
                            date_publication_comment = datetime.fromtimestamp(timestamp).strftime("%d-%m-%Y")
                        else:
                            date_publication_comment = None

                        comments.append({
                            "nom_utilisateur": comment.get("author"),
                            "date_publication": date_publication_comment,
                            "texte_commentaire": comment.get("text"),
                            "langue": comment.get("language"),
                        })

                data.append({
                    "titre_video": titre_video,
                    "description_video": description_video,
                    "hashtags": hashtags,
                    "date_publication": date_publication,
                    "lien_video": lien_video,
                    "nombre_commentaires": len(comments),
                    "commentaires": comments,
                })

        # Overwrite the JSON file with the new data
        with open(JSON_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    return JsonResponse({"data": data})