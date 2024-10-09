# scraper/forms.py
from django import forms

class KeywordForm(forms.Form):
    keyword = forms.CharField(label='Keyword', required=True)

class URLForm(forms.Form):
    url = forms.URLField(label='YouTube URL', required=True)

class SearchForm(forms.Form):
    keyword = forms.CharField(label='Search Keyword', max_length=100)
