from django.urls import path
from .views import  add_emotion_to_category, delete_comment, delete_video, get_videos_by_corpus, load_json, manage_categories, manage_corpus, manage_emotions, scrape_video, search_videos,  submit_validation, update_comment_descriptor,update_comments,get_comments, update_corpus
# for cleaning ----------------------------------------------------------
from .views import BaseCleaningText,DeepCleaningText,comment_before,comment_after,display_clusters
#------------------------------------------------------------------------
from .views import get_etiquettes_view,etqchoisie_view, get_mots_view, motchoisie_view, download_excel
from .views import process_combinations, get_liste_corpus_view, corpus_choisi, get_freq_script, script_choisi

urlpatterns = [
    path('', load_json, name='load_json'),
      #--------------------pour listes deroulantes---------------------------
    path('etiquettes/', get_etiquettes_view, name='get_etiquettes'),
    path('mots/', get_mots_view, name='get_mots'),
    #-------------------------statistiques-------------------------------
    path('etqchoisie/', etqchoisie_view, name='etqchoisie'),
    path('motchoisie/', motchoisie_view, name='motchoisie'),
   # path('langues/',get_langues_view, name='get_langues'),
    path('process-combinations/', process_combinations, name='process_combinations'),
    path('listecorpus/', get_liste_corpus_view, name='listecorpus'),
    path('corpuschoisi/', corpus_choisi, name='corpuschoisi'),
    path('freq_scripts/', get_freq_script, name='freq_scripts'),
    path('scriptchoisi/', script_choisi, name='scriptchoisi'),
    #path('languages/', get_langues, name='languages'),
    path('download_excel/', download_excel, name='download_excel_file'),
    path('submit_validation/', submit_validation, name='submit_validation'),   
    path('update_comments/',update_comments, name='update_comment'),
    path('get_comments/', get_comments, name='get_comments'),
    path('emotions/', manage_emotions, name='manage_emotions'),
    path('emotions/<str:emotion_id>/', manage_emotions, name='manage_emotion'),
    path('search/', search_videos, name='search_videos'),
    path('scrape/', scrape_video, name='scrape_video'),
    path('categories/', manage_categories, name='manage_categories'),
    path('categories/<str:id>/', manage_categories, name='manage_category_by_id'),  # For PUT and DELETE

    path('categories/add-emotion/', add_emotion_to_category, name='add_emotion_to_category'),
    path('corpus/', manage_corpus, name='manage_corpus'),
    path('corpus/<str:corpus_id>/', manage_corpus, name='manage_corpus'),
    path('update_corpus/<str:corpus_id>/', update_corpus, name='update_corpus'),

    path('get_videos_by_corpus/', get_videos_by_corpus, name='get_videos_by_corpus'),
    #path('comments/update/', update_comment_descriptor, name='update_comment_descriptor'),
    path('update/', update_comment_descriptor, name='update_comment_descriptor'),  
 
    path('comments/delete/', delete_comment, name='delete_comment'),  # Add this
    path('videos/delete/', delete_video, name='delete_video'),  # Add this
        #----------------------------------------------------------------------------
    path('base_clean_text/', BaseCleaningText.as_view(), name='base_clean_text'),
    path('deep_clean_text/', DeepCleaningText.as_view(), name='deep_clean_text'),
    path('before_comments/', comment_before, name='view_comment_before_preclean'),
    path('treated_comments/', comment_after, name='view_comment_after_preclean'),
    path('clusters/', display_clusters, name='display_clusters'),
    #----------------------------------------------------------------------------
     
]