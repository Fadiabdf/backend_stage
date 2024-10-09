# cleaning module functions
#---------------------------
#  Import Python liberies  #
#---------------------------
import re
import emoji
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from num2words import num2words
from collections import Counter
from langdetect import detect
#-----------------------------------------------
from pymongo import MongoClient
#_________________________________________________________________________
# Load spaCy models for English and French
nlp_en = spacy.load('en_core_web_lg')
nlp_fr = spacy.load('fr_core_news_lg')
#-----------------------------------------------------
nltk.download('stopwords')
#-----------------------------------------------------
#___________________________________________________________________________________________
#----------------------
# 0. Language Detection 
#----------------------
def detect_language(text):
    try:
        language = detect(text)
        if language in ['fr','en','ar']:
           return language
        else:
           return "Langue non identifiée"
    except:
        return "Erreur de détection"
#--------------------------------------------------------
#----------------------
# 1. Script Detection ##
#----------------------
def detect_script(text):
  """
    Detects the type of script in the given text.
    Args:
    text (str): The input text to analyze.
    Returns:
    str: The detected script type - 'Latin Script', 'Arabic Script','Mixed Script'or 'Other Script' if there is only emojis in the text.
  """
  #We Define Unicode ranges for Latin and Arabic scripts
  latin_range = re.compile(r'[A-Za-zÀ-ÖØ-öø-ÿ]')
  arabic_range = re.compile(r'[\u0600-\u06FF\u0750-\u077F]')
  #We Check if the text contains Latin and Arabic characters
  has_latin = bool(latin_range.search(text))
  has_arabic = bool(arabic_range.search(text))
  if has_latin and not has_arabic:
      return 'Latin Script'
  elif has_arabic and not has_latin:
      return 'Arabic Script'
  elif has_latin and has_arabic:
      return 'Mixed Script'
  else:
      return 'Other Script'
#--------------------------------------------------------
#----------------------------
# 2. Remove URLs            ##
#----------------------------
def remove_urls(text):
    """
    Removes URLs from the given text.
    Args:
    text (str): The input text from which to remove URLs.
    Returns:
    str: The text with URLs removed.
    """
    url_pattern = re.compile(r'https?://\S+|www\.\S+|ftp?://\S+|http[s]?://\S+',flags=re.MULTILINE)
    return url_pattern.sub(r'', text)
#---------------------------------------------------------------------------------------------------------------
#-----------------------------
# 3. Remove HTML Tags        ##
#-----------------------------
def remove_html_tags(text):
    """
    Removes HTML tags from the given text.
    Args:
    text (str): The input text from which to remove HTML tags.
    Returns:
    str: The text with HTML tags removed.
    """
    html_tag_pattern = re.compile(r'<.*?>')
    return html_tag_pattern.sub(r'', text)
#---------------------------------------------------------------------------------------------------------------
#------------------------------------
# 4. Remove Mentions and Hashtags   ##
#------------------------------------
def Remove_mentions_hashtags(text):
    """
    Remove mentions and hashtags from the given text.
    Args:
    text (str): The input text from which to process mentions and hashtags.
    Returns:
    str: The text with mentions and hashtags removed.
    """
    text = re.sub(r'@\w+', '', text)  # To Remove mentions
    text = re.sub(r'#\w+', '', text)  # To Remove hashtags
    return text.strip()
#---------------------------------------------------------------------------------------------------------------
#------------------------------------
#  5. Emoji and Emoticon Conversion ##
#------------------------------------
# We Define a dictionary for emoticon
emoticon_dict = {
    ":)": "[happy]",       ":-)": "[happy]",        "<3": "[heart]",            "</3": "[broken heart]",
    ":D": "[very happy]",  ":-D": "[very happy]",    "-.-": "[annoyed]",
    ":(": "[sad]",         ":-(": "[sad]",
    ":/": "[confused]",    ":-/": "[confused]",
    ":|": "[neutral]",     ":-|": "[neutral]",
    ";)": "[wink]",        ";-)": "[wink]",
    ":P": "[playful]",     ":-P": "[playful]",
    ":O": "[surprised]",   ":-O": "[surprised]",
    "XD": "[laughing]",    "xD": "[laughing]",
    ":*": "[kiss]",        ":3": "[cute]",
    "^-^": "[happy]",      "-_-": "[annoyed]",
    ":'(": "[crying]",     ":'-)": "[happy tears]",
    ">:(": "[angry]",      ">:-(": "[angry]",
    "O_O": "[shocked]",    "o_O": "[confused]",
    ">.<": "[annoyed]",    "u_u": "[disappointed]",
    "^^": "[joy]",         "^_~": "[wink]",
    "T_T": "[crying]",     "Q_Q": "[crying]",
    "x_x": "[dead]",       "X_X": "[dead]",
    "=)": "[happy]",       "=(": "[sad]",
    "=D": "[very happy]",  "=P": "[playful]",
    ">:O": "[shocked]",    ">_<": "[annoyed]",
    ":&": "[sick]",        ":@": "[angry]",
    "*_*": "[starstruck]", "o_o": "[surprised]",
    ":c": "[disappointed]", ":'D": "[happy tears]",
    "B1": "[good]",        
}

def convert_emojis_and_emoticons_to_text(text):
    """
    Replace emojis and emoticons with their textual descriptions.
    If the same emoji or emoticon appears two or more times in a row, keep only two occurrences.
    """
    # We Convert emojis to textual descriptions
    text = emoji.demojize(text)
    text = re.sub(r':([a-zA-Z0-9_]+):', r' {\1} ', text)   # emojis will be written like this {happy} and emoticon like this [confused]

    # We Replace emoticons with their descriptions
    for emoticon, description in emoticon_dict.items():
        escaped_emoticon = re.escape(emoticon)
        # We Match emoticons even when they are adjacent to text or punctuation
        text = re.sub(rf"(?<!\w){ escaped_emoticon }(?!\w)", description, text)

    # We Limit consecutive repetitions to two for both emojis and emoticons
    text = re.sub(r'(\{\w+\})\1+', r'\1\1', text)  # For emojis
    text = re.sub(r'(\w+)\1+', r'\1\1', text)  # For emoticons

    return text
#---------------------------------------------------------------------------------------------------------------
#------------------------------------
# 6. Arabizi Number Conversion      ##
#------------------------------------
def normalize_arabizi(text):
    """
    Normalize Arabizi characters by converting different Latin representations
    into a consistent format. Normalization is applied only if Arabizi characters
    are part of a word surrounded by letters and does not contain consecutive numbers.
    Consecutive numbers will be removed entirely.
    """
    # We Define Arabizi normalization mappings
    arabizi_mappings = {
        "7": "h", "9": "q", "2": "a",
        "5": "kh", "3o": "o", "3a": "a",
        "3e": "e", "3i": "i", "i3": "i",
        "o3": "o", "a3": "a", "e3": "e", "3": "a"
    }

    # to normalize a single word
    def normalize_word(word):
        for original, normalized in arabizi_mappings.items():
            word = word.replace(original, normalized)
        return word

    # to process matches
    def process_match(match):
        word = match.group()
        
        # We Check if the word contains Arabizi characters and is flanked by letters
        if re.search(r'[a-zA-Z]', word):
            # We Remove consecutive numbers entirely
            if re.match(r'^\d+$', word):  # this Matches words that consist solely of digits
                return ''  # We Remove the number entirely
            
            # If the word contains Arabizi characters, We normalize it
            if any(c in word for c in arabizi_mappings):
                return normalize_word(word)

        # We Handle cases with numbers mixed with letters
        cleaned_word = re.sub(r'\d+', '', word)  # We Remove all digits
        return cleaned_word

    # We Apply normalization only to sequences of word characters that are flanked by letters
    text = re.sub(r'\b[a-zA-Z0-9]*[a-zA-Z]+[a-zA-Z0-9]*\b', process_match, text)
    return text

#---------------------------------------------------------------------------------------------------------------
#----------------------------------
# 7. Remove Non-Textual Element   ##
#----------------------------------
def remove_non_textual_elements(text):
    """
    Remove non-textual elements such as references to images, GIFs, and stickers
    from the given text.
    Args:
    text (str): The input text containing potential non-textual element indicators.
    Returns:
    str: The text with non-textual elements removed.
    """
    # We Define patterns for common non-textual element indicators
    non_textual_patterns = [
        r'\[image:.*?\]',  
        r'\[gif:.*?\]',    
        r'\[sticker:.*?\]',
        r'https?://\S+\.(jpg|jpeg|png|gif)',  
        r'https?://\S+\.(svg|webp|bmp)',      
        r'\b(image|gif|sticker):\S+',         
    ]
    # We Compile the patterns into a single regular expression
    combined_pattern = re.compile('|'.join(non_textual_patterns), flags=re.IGNORECASE)
    # We Remove matches
    cleaned_text = combined_pattern.sub('', text)
    return cleaned_text
#---------------------------------------------------------------------------------------------------------------
#----------------------------------
# 8. Remove Stop Word             ##
#----------------------------------
# We Fetch stop words from NLTK for French, and English
french_stop_words = set(stopwords.words('french'))
english_stop_words = set(stopwords.words('english'))

dz_arabizi_stopwords = {"w","fi","men","ya","ala","ila","ma","la","eli","ken","kol","enta","ano","ena","hoa",
                      "alik","fih","amin","bech","ama","rak","ay","wala","hata",
                       'ida','binma' ,'qalmen', 'akthar','homa','hia','hna','dok','doka','hia', 'hiya', 'hayt',
                      "hya",'dak','dik','douk','hadouk','tani','thani','hadi','hada','el','lik','lihom','liha','liya','ila','ina', 'ana', 'nta', 'nti', 'ntouma',
                      'ayn', 'aynma', 'ayh',"menek","mouch","ykoun","lih","biha","ki","alih","bin","maak","alihom","hak","ghir", 'mahma',
                      "fiha","kif","aandek","kont","khouya","khti","kho","menou","mech","had","kamel","dima", 'kyf', 'hatan','houwa',
                      "laken","youm","aya","lezem","awled","aliha","lyoum","wahda","maadech","ahla","fik","rahom","kima",'lah','hak',
                      "mazel","jey","li","wahed","hada","wi","deja","ach","ieh","ih","howa","heya","houma","ahna","ena","fl",'bi', 'bayn',
                      "fel","men","mn","ala","a","l","yaatek","yaatik","ah",'ay','ayh','hkdha',
                      "alech","kifeh","win","waktech","wino","win","cho","ti","alik","maak","mela","haja","yekhi","omkom","weli","okhra",
                      "aleh","fikom","kifek","tji","bara",'aha', 'aw', 'baad', 'baadh', 'bik', 'bina', 'bih', 'biha', 'bihoum','kl', 'lala',
                      'wash', 'bwash', 'bwash','maa','ntaa','kima','ili','lel','l','andek','andhom','andha','adna','lina','ando','andi',
                      'kthr', 'ali', 'layk', 'laykum', 'nitya', 'natuma', 'bsah', 'hum', 'wain', 'ambaad', 'awambaad', 'shwaya', 'shwia', 'wash',
                      'wala', 'walakin', 'walu','walo', 'wama', 'waman', 'wahuwa', 'ya', 'yamal','eh','ih'}


# We Combine all stop words into a single set
all_stop_words = dz_arabizi_stopwords | french_stop_words | english_stop_words 

def remove_stop_words(text):
    """
    Remove stop words from the given text using NLTK stop word lists for French, English,
    Args:
    text (str): The input text from which to remove stop words.
    Returns:
    str: The text with stop words removed.
    """
    # We Tokenize the text by splitting on spaces
    words = text.split()
    # We Remove stop words
    filtered_words = [word for word in words if word.lower() not in all_stop_words]
    # We Join the filtered words back into a string
    filtered_text = ' '.join(filtered_words)
    return filtered_text

#---------------------------------------------------------------------------------------------------------------
#--------------------------------------
#  9. Remove Whitespace               ##
#--------------------------------------
def remove_specific_whitespace(text):
    """
    Remplace les caractères de tabulation, de retour chariot, de saut de page et d'espaces verticaux par un seul espace,
    réduit les espaces multiples à un seul espace et réduit les sauts de ligne multiples à un seul saut de ligne.
    Args:
    text (str): Le texte d'entrée dont on veut supprimer les caractères d'espacement spécifiques.
    Returns:
    str: Le texte nettoyé des caractères d'espacement spécifiques.
    """
    # We Remplace les tabulations, retours chariot, sauts de page et espaces verticaux par un seul espace
    text = re.sub(r'[\t\r\f\v]+', ' ', text)
    # We Remplace les espaces multiples par un seul espace
    text = re.sub(r' +', ' ', text) 
    # We Remplace les sauts de ligne multiples par un seul saut de ligne
    text = re.sub(r'\n{2,}', '\n', text)
    
    return text
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------
#  10. Remove Personal Information     ##
#---------------------------------------
def remove_personal_info(text):
    """
    Remove personal information such as phone numbers and addresses from the given text.
    Args:
    text (str): The input text from which to remove personal information.
    Returns:
    str: The text with personal information removed or masked.
    """
    # We Define patterns for Algerian phone numbers
    phone_patterns = [
        r'\+213\s?\d{2,3}[\s\-]?\d{3}[\s\-]?\d{3}',  # this Matches +213 555 123 456 or +213-555-123-456
        r'0\d{9}',  # this Matches 0555123456
    ]
    # We Define patterns for typical address keywords and formats (can be expanded)
    address_patterns = [
        r'\b(Rue|Avenue|Cité|Lotissement|Lot|Immeuble|Quartier|Villa|Bâtiment|Bloc)\b[\s\w]*',  # Common address keywords
        r'\b\d{5}\b',  # this Matches postal codes (assuming 5-digit postal codes)
    ]
    # We Compile and combine patterns
    combined_pattern = re.compile('|'.join(phone_patterns + address_patterns), flags=re.IGNORECASE)
    # We Remove personal information
    cleaned_text = combined_pattern.sub('[PERSONAL INFO]', text)
    return cleaned_text
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------
#  11. Lemmatization      #
#---------------------------------------
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
# Initialisation du lemmatizer NLTK
wnl = WordNetLemmatizer()
def lemmatize(text, language='fr'):
    """
    Lemmatize the given text.
    Args:
    text (str): The input text.
    language (str): The language of the text ('en', 'fr').
    Returns:
    str: The lemmatized text.
    """
    if language == 'en':
        doc = nlp_en(text)
        lemmatized = [token.lemma_ for token in doc]
    elif language == 'fr':
        doc = nlp_fr(text)
        lemmatized = [token.lemma_ for token in doc]
    else:
        raise ValueError("Unsupported language. Choose 'en', 'fr'")

    return ' '.join(lemmatized)
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------
#  12. normalize arabic numbers        #
#---------------------------------------
def normalize_arabic_numbers(text):
    """
    Converts Arabic-Indic digits in the text to Western Arabic numerals.
    Args:
    text (str): The input text containing Arabic-Indic digits.
    Returns:
    str: The text with Arabic-Indic digits replaced by Western Arabic numerals.
    """
    arabic_to_western = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    return text.translate(arabic_to_western)
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------
#  13. Remove Punctuation              #
#---------------------------------------
def remove_punctuation(text):
    """
    Removes all punctuation from the given text except for conserve_chars, and limits consecutive occurrences of these characters to a maximum of two.
    Args:
    text (str): The input text from which to remove punctuation.
    Returns:
    str: The text without unnecessary punctuation, keeping only two consecutive '?' or '!'... characters.
    """
    # these are the characters to conserve
    conserve_chars = {'?', '!', '[', ']', '{', '}', '*',"'","`"}

    # We Remove all punctuation except for the conserved characters
    # string.punctuation = ! " # $ % & ' ( ) * + , - . / : ; < = > ? @ [ \ ] ^ _ ` { | } ~ 
    translation_table = str.maketrans('', '', ''.join(c for c in string.punctuation if c not in conserve_chars))
    text = text.translate(translation_table)

    # We Remove the specific characters '«' and '»' and "'" and "`" and "…" and "’" not included in string.ponct
    text = text.replace('«', '').replace('»', '')
    text = text.replace("'",' ')
    text = text.replace("…", '')
    text = text.replace('’',' ')
    text = text.replace('`',' ')
    # We limit consecutive conserved characters to two
    text = re.sub(r'\?\?+', '??', text) # text = re.sub(r'\?+', '?', text)
    text = re.sub(r'!!+', '!!', text) # text = re.sub(r'\!+', '!', text)
    text = re.sub(r'\[+', '[', text)
    text = re.sub(r'\]+', ']', text)
    text = re.sub(r'\{+', '{', text)
    text = re.sub(r'\}+', '}', text)
    text = re.sub(r'\*\*+', '**', text) # text = re.sub(r'\*+', '*', text)
    #text = re.sub(r'\.\.+', '..', text) # text = re.sub(r'\*+', '*', text)
    return text
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------
#  14. Remove Small Words              ##
#---------------------------------------
def remove_minlen_word(text, threshold):
    """
    Removes words shorter than the specified threshold length from the text while preserving punctuation.
    Args:
    text (str): The input text containing words and punctuation.
    threshold (int): The minimum length of words to keep.
    Returns:
    str: The text with words shorter than the threshold removed, preserving punctuation.
    """
    # pattern to match words and punctuation
    pattern = re.compile(r'\b\w+\b|[^\w\s]')   
    # words and punctuation in the text
    tokens = pattern.findall(text) 
    # short words to remove
    short_words = {token for token in tokens if token.isalpha() and len(token) <= threshold}
    
    # We Reconstruct the text
    result = []
    for token in tokens:
        if token in short_words:
            continue
        if token in string.punctuation:
            # If the token is punctuation,We join it without spaces
            if result and not result[-1].isspace():
                result.append(token)
            else:
                result.append(token)
        else:
            # If the token is a word, We join with a space
            if result and not result[-1].isspace():
                result.append(' ')
            result.append(token)
    
    # Join the result list into a single string
    return ''.join(result)
#---------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
#  15. Turn Text into lowercase and conserve the words that are entirely capitalized ##
#-------------------------------------------------------------------------------------
def smart_lowercase(text):
    """
    Converts text to lowercase, except for words that are entirely capitalized.
    Args:
    text (str): The input text.
    Returns:
    str: The text with lowercase conversion applied selectively.
    """
    words = text.split()
    # We Process each word to convert to lowercase unless it is fully capitalized
    processed_words = [
        word if word.isupper() else word.lower() for word in words
    ]

    # We Join the processed words back into a single string
    return " ".join(processed_words)
#---------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------
#  16. Convert number into text in any language used     ## not to use for the treatment
#--------------------------------------------------------
def convert_number_to_text(number, language='en'):
    """
    Converts a number to its textual representation in the specified language.
    Args:
    number (int): The number to convert.
    language (str): The language for conversion ('en' for English).
    Returns:
    str: The textual representation of the number.
    """
    try:
        return num2words(number, lang=language)
    except NotImplementedError:
        return 'Language not supported'
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------
#  17. Remove the Redundant Comments   ##
#---------------------------------------
from concurrent.futures import ThreadPoolExecutor, as_completed

def check_similarity(cosine_sim_matrix, i, j, similarity_threshold):
    """Check if two comments are redundant based on cosine similarity."""
    return cosine_sim_matrix[i, j] >= similarity_threshold

def detect_and_remove_redundant_messages(comments, similarity_threshold=0.8):
    """
    Detects and removes redundant messages in a list of comments, treating emojis as text.   
    Args:
        comments (list of str): List of comments to check for redundancy.
        similarity_threshold (float): Threshold for cosine similarity to consider comments as redundant. 
    Returns:
        list of str: List of comments with redundancies removed.
    """
    if len(comments) < 2:
        return comments
    # We Convert emojis to text descriptions
    comments_text = [emoji.demojize(comment) for comment in comments]
    # We Vectorize comments using TF-IDF
    vectorizer = TfidfVectorizer().fit_transform(comments_text)
    cosine_sim_matrix = cosine_similarity(vectorizer)
    # We Set to track redundant indices
    redundant_indices = set()

    # We use Parallel processing with 8 threads
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for i in range(len(cosine_sim_matrix)):
            if i in redundant_indices:
                continue
            for j in range(i + 1, len(cosine_sim_matrix)):
                futures.append(executor.submit(check_similarity, cosine_sim_matrix, i, j, similarity_threshold))

        # Collect results from parallel tasks
        for idx, future in enumerate(as_completed(futures)):
            i = idx // (len(cosine_sim_matrix) - 1)
            j = idx % (len(cosine_sim_matrix) - 1) + 1
            if future.result():
                redundant_indices.add(j)

    # We Keep only unique comments
    unique_comments_indices = [idx for idx in range(len(comments)) if idx not in redundant_indices]
    unique_comments = [comments[idx] for idx in unique_comments_indices]

    return unique_comments

#---------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------
#  18. Remove redundant words or expresssions in a comment #
#-----------------------------------------------------------
def remove_redundant_words(comment, min_count=2):
    """
    Detects and removes redundant words or expressions within a single comment, keeping only one instance.
    Args:
    comment (str): The comment text to analyze.
    min_count (int): Minimum count for a word or expression to be considered redundant.
    Returns:
    str: The comment text with redundant words or expressions removed.
    """
    # We Preprocess the text
    text = re.sub(r'\s+', ' ', comment)  # We Replace multiple spaces with a single space
    #text = re.sub(r'[^\w\s]', '', text)  # We Remove punctuation
    # We Tokenize the text
    words = text.split()
    # We Count the occurrences of each word
    word_counts = Counter(words)
    # We Find redundant words or expressions
    redundant_words = {word: count for word, count in word_counts.items() if count >= min_count}
    # We Create a list to store words while keeping track of the first occurrence of redundant words
    result = []
    seen = set()

    for word in words:
        if word in redundant_words and word in seen:
            continue
        result.append(word)
        seen.add(word)

    # We Join the words back into a string
    cleaned_comment = ' '.join(result)
    return cleaned_comment
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------
#  19. Normalize Abbreviations and Acronyms  #
#---------------------------------------------
# the abbreviation dictionary
abbreviation_dict = {
    "wlh": "wellah",              "vrm": "vraiment",               "btw": "by the way",              "omg": "oh my god","omgg": "oh my god",     #  "brb": "be right back",    "icymi": "in case you missed it",    "ttyl": "talk to you later",   "fwiw": "for what it's worth",  "imho": "in my humble opinion",   "tl;dr": "too long; didn't read", "hmu": "hit me up",     "smh": "shaking my head",             "ttyl": "talk to you later",    "ftw": "for the win",
    "t": "tu es",                 "lol": "laughing out loud","lool": "laughing out loud",      "asap": "as soon as possible",
    "aprèm": "après midi",           "tbh": "to be honest",            "nvm": "never mind",       "gl": "good luck",        # "gr8": "great", "hmu": "hit me up",  "jk": "just kidding",          "bbs": "be back soon",          "tba": "to be announced",          "bf": "boyfriend",               "bbl": "be back later", "faq": "frequently asked questions",  "w/e": "whatever", "gf": "girlfriend",
    "idk": "I don t know",        "bn8": "bonne nuit",             "dispo": "disponible",            "idc": "I don t care",  #   "xoxo": "hugs and kisses", "np": "no problem",                  "rofl": "rolling on the floor laughing",          "gg": "good game",
    "fyi": "for your information","tldr": "too long didn't read",                                    "bro": "brother","broo": "brother",       # "tmi": "too much information",   "w/e": "whatever",    "wth": "what the heck",   "bfn": "bye for now",
    "imo": "in my opinion",       "dz":"dzair",                    "fr":"france",                    "c1":"c est un", "C1":"C EST UN" ,"d1":"d un","D1":"D UN",
    "sis": "sister",
    "bn": "bon",                  "u": "you",                      "r": "are",
    "plz": "please",              "cc": "coucou",                  "bnj": "bonjour",
    "cya": "see you",             "dm": "direct message",          "bsr": "bonsoir",
    "atm": "at the moment",       "stp": "s il te plait",
    "ok": "okay",                 "rn": "right now",               "afk": "away from keyboard",
    "slt": "salut",               "mn": "minute",                  "lmk": "let me know",
    "cv": "ça va",                "jsp": "je sais pas",            "jk": "just kidding",
    "irl": "in real life",        "cbon": "c'est bon",             "c": "c'est",                     "ik": "I know",
    "p2q": "pas de quoi",         "mrc": "merci","mrcc": "merci",                 "cuz":"because", "cuzz":"because",

    "afaik": "as far as I know",                                   "btw": "by the way",              "diy": "do it yourself",
    "omw": "on my way",           "wth": "what the heck",          "idgaf": "I don t give a f***",
    "lmao": "laughing my ass off","ttfn": "ta ta for now",         "wyd": "what are you doing",
    "n/a": "not applicable",      "nvm": "never mind",
    "np": "no problem",           "wb": "welcome back",            "yw": "you're welcome",
    "sup": "what's up",           "wtf": "what the f***",
    "gtg": "got to go",
    "tl;dr": "too long; didn't read",                              "fomo": "fear of missing out",    "bday": "birthday",
    "ikr": "I know right",         "smh": "shaking my head",
    "ftl": "for the loss",        "brt": "be right there",         "gf": "girlfriend",
    "irl": "in real life",        "thx": "thanks",                 "ttyl": "talk to you later",
    "atm": "at the moment",       "gr8": "great",                  "idc": "I don t care",
    "imho": "in my humble opinion","l8r": "later",                 "rofl": "rolling on the floor laughing",

    "bff": "best friends forever",   "brb": "be right back",           "btw": "by the way",
    "cc": "carbon copy",          "cmiiw": "correct me if I'm wrong",  "cu": "see you",
    "diy": "do it yourself",      "dw": "don't worry",                 "fomo": "fear of missing out",
    "fyi": "for your information","gtg": "got to go",                  "hbd": "happy birthday",
    "ic": "I see",                "idk": "I don't know",                "iirc": "if I recall correctly",
    "imo": "in my opinion",
    "lmk": "let me know",
    "rofl": "rolling on the floor laughing","rsvp": "please respond",
    "tbc": "to be continued",     "tbd": "to be determined",
                                "yolo": "you only live once",     "dunno":"don t know",

    "pcq":"parce que",            "psq":"puisque",                 "rdv":"rendez-vous",              "qd":"quand",
    "bcp":"beaucoup","bcpp":"beaucoup",            "svp":"s il vous plaît","svpp":"s il vous plaît",         "ptdr":"pété de rire",            "mdr":"mort de rire", "mdrr":"mort de rire", "CQFD": "Ce Qu il Fallait Démontrer" ,"cqfd":"Ce Qu il Fallait Démontrer",
}

def normalize_abbreviations(text):
    #text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = text.split()
    expanded_words = []
    for word in words:
        if word.lower() in abbreviation_dict:
            expanded_words.append(abbreviation_dict[word.lower()])
        else:
            expanded_words.append(word)
    return " ".join(expanded_words)
#---------------------------------------------------------------------------------------------------------------
#------------------------------------
#  20. Remove Numbers               #
#------------------------------------
def remove_numbers(text):
    """
    This function takes a string and returns a new string with all numbers removed,
    except for the number 106 .
    Parameters:
    text (str): The input string from which numbers will be removed.
    Returns:
    str: A string with all numbers removed except 106.
    """
    # We match numbers, but not 106
    return re.sub(r'\b(?!106\b)\d+\b', '', text)
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------
#  21.Remove Redundant Emojis          ##
#---------------------------------------
def remove_redundant_emojis(text):
    # We Get a list of all emojis in the text 
    emojis = emoji.emoji_list(text)   
    # We Initialize an empty list to hold the non-redundant text
    clean_text = []
    last_emoji = None
    # index for text traversal
    i = 0

    # We Iterate through the emojis detected in the text
    while i < len(text):
        match_found = False

        for e in emojis:
            # We Check if the current position matches an emoji's start position
            if text[i:i + len(e['emoji'])] == e['emoji']:
                match_found = True
                current_emoji = e['emoji']
                
                # If it's different from the last seen emoji, we append it
                if current_emoji != last_emoji:
                    clean_text.append(current_emoji)
                    last_emoji = current_emoji
                
                # Skip the length of the emoji sequence
                i += len(current_emoji)
                break

        # If no emoji was found at this position,we just add the character to the clean_text
        if not match_found:
            clean_text.append(text[i])
            last_emoji = None
            i += 1
    
    return ''.join(clean_text)

#---------------------------------------------------------------------------------------------------------------
#---------------------------------------
#  22. remove arabic letters           ##
#---------------------------------------
def remove_arabic_letters(text):
    arabic_letters_pattern = re.compile('[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
    cleaned_text = re.sub(arabic_letters_pattern, '', text)
    return cleaned_text
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------
#  23. remove_exaggerations           ##
#---------------------------------------
def remove_exaggerations(text):
    # we find repeated characters and limit them to 3 occurrences
    normalized_text = re.sub(r'(.)\1{3,}', r'\1\1', text)
    return normalized_text
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------
#  24. detection des publicités       ##
#---------------------------------------
def is_advertisement(text):
    """Détecte si le texte semble être une publicité."""    
    max_length_for_ads = 300 
    # detect  pub
    ad_patterns = [
        r'\b(?:buy|purchase|discount|offer\s+now|sale|free\s+gift|limited time|act now|click here|call now|order now)\b',
        r'\b(?:acheter|promotion|réduction|offre\s+spéciale|soldes|gratuit|temps limité|agissez maintenant|cliquez ici|appelez maintenant|commandez maintenant)\b',
        r'\b(?:ishri|irba7|iphone|3ard|sell|free|limited|click|order|call)\b',
    ]   
    # Combine patterns 
    combined_pattern = '|'.join(ad_patterns)
    if len(text) > max_length_for_ads:
        return False
    matches = re.findall(combined_pattern, text, re.IGNORECASE)
    if len(matches) > 1:
        return True 
    return False
#---------------------------------------------------------------------------------------------------------------
#__________________________________________________________________________________________________________________________________________
# Connect to MongoDB
client = MongoClient('mongodb+srv://maroua:maroua2003@cluster0.p6t0qwx.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
db = client['projet_1cs']
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------
#  26. normalisation linguistique     ##
#---------------------------------------
from collections import defaultdict
import jellyfish
from fuzzywuzzy import fuzz
import spacy
from difflib import SequenceMatcher
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
#-----------------------------------------------------------------------------------
# Base de données des clusters
#-----------------------------------------------------------------------------------
#clusters_db = defaultdict(list)
clusters_db = db.clusters
#-----------------------------------------------------------------------------------
# phonetic similarity
#-----------------------------------------------------------------------------------
@lru_cache(maxsize=10000)
def soundex_code(word):
    """Calcule le code Soundex pour un mot donné."""
    return jellyfish.soundex(word)
#-----------------------------------------------------------------------------------
# ortographic similarity
#-----------------------------------------------------------------------------------
def levenshtein_similarity(word1, word2):
    """Calcule la similarité entre deux mots en utilisant la distance de Levenshtein."""
    if word1 == word2:
        return 100  # Exact match
    if abs(len(word1) - len(word2)) > 3:  
        return 0
    return fuzz.ratio(word1, word2)
#-----------------------------------------------------------------------------------
# update the clusters 
#-----------------------------------------------------------------------------------
def update_clusters_db(word, clusters_db):
    cleaned_word = re.sub(r'\W+', '', word).lower()
    word_soundex = soundex_code(cleaned_word)
    cluster = clusters_db.find_one({'soundex': word_soundex})
    if cluster:
        for candidate in cluster['words']:
            if cleaned_word == candidate:
                return
            phonetic_sim = 1
            ortho_sim = levenshtein_similarity(cleaned_word, candidate) / 100
            if phonetic_sim >= 0.8 and ortho_sim >= 0.7:
                if cleaned_word not in cluster['words']:
                    cluster['words'].append(cleaned_word)
                    clusters_db.update_one(
                        {'soundex': word_soundex},
                        {'$set': {'words': list(set(cluster['words']))}}
                    )
                return
        if cleaned_word not in cluster['words']:
            cluster['words'].append(cleaned_word)
            clusters_db.update_one(
                {'soundex': word_soundex},
                {'$set': {'words': list(set(cluster['words']))}}
            )
    else:
        new_cluster = {
            'soundex': word_soundex,
            'words': [cleaned_word]
        }
        clusters_db.insert_one(new_cluster)
#-----------------------------------------------------------------------------------
# selecting standard term using ortographic similarity
#-----------------------------------------------------------------------------------
def select_standard_term(cluster_words):
    """Sélectionne le terme le plus représentatif d'un cluster."""
    max_similarity = 0
    standard_term = cluster_words[0]
    for term1 in cluster_words:
        total_similarity = sum(levenshtein_similarity(term1, term2) for term2 in cluster_words)
        if total_similarity > max_similarity:
            max_similarity = total_similarity
            standard_term = term1
    return standard_term
#-----------------------------------------------------------------------------------
# to normalize a single word
#-----------------------------------------------------------------------------------
def normalize_word(word):
    """Normalise un mot en le remplaçant par son terme standardisé si disponible."""
    cleaned_word = re.sub(r'\W+', '', word).lower()
    cluster = clusters_db.find_one({'soundex': soundex_code(cleaned_word)})
    if cluster:
        return select_standard_term(cluster['words'])  
    return cleaned_word
#-----------------------------------------------------------------------------------
# to normalize a text
#-----------------------------------------------------------------------------------
def normalize_text(text):
    """Applique la normalisation à chaque mot d'un texte."""
    def is_emoji(word):
        return bool(re.match(r'^\{.*\}$', word))
    words = text.split()
    normalized_words = []
    # parallel processing with 8 threads
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(normalize_word, word.lower().translate(str.maketrans('', '', string.punctuation))): word for word in words if not is_emoji(word)}
        for future in as_completed(futures):
            normalized_word = future.result()
            normalized_words.append(normalized_word)
        normalized_words += [word for word in words if is_emoji(word)]
    return ' '.join(normalized_words)
#-----------------------------------------------------------------------------------
# update clusters with new words from new text
#-----------------------------------------------------------------------------------
def process_new_text(new_text):
    """Traite et normalise un nouveau texte, en mettant à jour les clusters."""
    new_words = new_text.split()
    with ThreadPoolExecutor() as executor:
        executor.map(lambda word: update_clusters_db(word, clusters_db), new_words)
    normalized_text = normalize_text(new_text)
    return normalized_text
#-----------------------------------------------------------------------------------

######################################################################################################################################################
                            ##############################
                            #   for  cleaning views      #
                            ##############################
######################################################################################################################################################

#---------------------------------------
#  filtrage automatique               ##
#---------------------------------------

from concurrent.futures import ThreadPoolExecutor
import threading

# Lock to control access to the count
lock = threading.Lock()

def process_video_document(video_document):
    try:
        cpt = 0  # Local counter for processed comments
        comment_map = {}  #  Store here the cleaned texts mapped to their comment ID
        comments_text = []  # a List of cleaned comment texts for redundancy detection
        comment_ids = []  # a List of comment IDs to track during processing

        for video in video_document.get('videos', []):
            id_video = video['id_video']

            for comment in video.get('commentaires', []):
                if comment['filtred']:  # Skip already filtered comments
                    continue

                with lock:
                    cpt += 1  # Increment the counter for processed comments

                comment_id = comment['id_commentaire']
                original_text = str(comment['texte'])

                # Cleaning and processing steps
                cleaned_text = remove_html_tags(original_text)
                cleaned_text = remove_urls(cleaned_text)
                cleaned_text = Remove_mentions_hashtags(cleaned_text)
                cleaned_text = remove_non_textual_elements(cleaned_text)
                cleaned_text = remove_personal_info(cleaned_text)
                cleaned_text = normalize_arabic_numbers(cleaned_text)
                cleaned_text = remove_redundant_emojis(cleaned_text)
                cleaned_text = smart_lowercase(cleaned_text)
                cleaned_text = remove_specific_whitespace(cleaned_text)

                # Add cleaned text and comment ID to tracking lists
                comments_text.append(cleaned_text)
                comment_ids.append(comment_id)
                comment_map[comment_id] = cleaned_text

            # We Apply redundancy detection after processing all comments in the video
            unique_comments_list = detect_and_remove_redundant_messages(comments_text, similarity_threshold=0.8)

            for comment_id in comment_ids:
                cleaned_text = comment_map[comment_id]

                # We Remove redundant or advertisement comments
                if cleaned_text not in unique_comments_list or is_advertisement(cleaned_text):
                    filter = {
                        '_id': video_document['_id'],
                        'videos.id_video': id_video
                    }
                    update = {
                        '$pull': {
                            'videos.$.commentaires': {
                                'id_commentaire': comment_id
                            }
                        }
                    }
                    db.video_collection.update_one(filter, update)

                # Otherwise, update the cleaned comment in the database
                else:
                    script_type = detect_script(cleaned_text)

                    filter = {
                        '_id': video_document['_id'],
                        'videos.id_video': id_video,
                        'videos.commentaires.id_commentaire': comment_id
                    }
                    update = {
                        '$set': {
                            'videos.$.commentaires.$[comment].texte': cleaned_text,
                            'videos.$.commentaires.$[comment].filtred': True,
                            'videos.$.commentaires.$[comment].script': script_type
                        }
                    }
                    array_filters = [{'comment.id_commentaire': comment_id}]
                    db.video_collection.update_one(filter, update, array_filters=array_filters)

        return cpt  # Return the count of processed comments for this document

    except Exception as e:
        return str(e)


def filtrage_auto():
    """Applique les étapes de prétraitement initial au texte ."""
    try:
        projection = {
            'videos.id_video': 1,
            'videos.commentaires.id_commentaire': 1,
            'videos.commentaires.texte': 1,
            'videos.commentaires.filtred': 1
        }
        video_documents = db.video_collection.find({}, projection)

        # Initialize total comment count
        total_cpt = 0

        # Process with a maximum of 8 threads
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_video_document, video_document) for video_document in video_documents]

            # We Retrieve the results from the threads
            for future in futures:
                result = future.result()
                if isinstance(result, int):
                    total_cpt += result

        return {'total_processed_comments': total_cpt}

    except Exception as e:
        return str(e)


#-------------------------------------------------------------for testing
'''
from concurrent.futures import ThreadPoolExecutor, as_completed

def filtrage_auto():
    """Applique les étapes de prétraitement initial au texte en parallèle."""
    try:
        video_documents = list(db.cleaning.find())
        cleaned_comments = []
        cpt = 0

        def process_video_document(video_document):
            local_cleaned_videos = []
            local_cpt = 0
            try:
                category = video_document.get('categorie', ['politique'])
                for video in video_document.get('videos', []):
                    cleaned_video_comments = []
                    for comment in video.get('commentaires', []):
                        local_cpt += 1
                        text = str(comment['texte'])
                        comment_id = comment['id_commentaire']
                        pub_date = comment['date_publication']
                        author = comment['auteur']
                        language = comment['langue']
                        descripteurs = comment['descripteur']

                        # Cleaning and processing steps
                        text = remove_html_tags(text)
                        text = remove_urls(text)
                        text = Remove_mentions_hashtags(text)
                        text = remove_non_textual_elements(text)
                        text = remove_personal_info(text)
                        text = normalize_arabic_numbers(text)
                        text = remove_redundant_emojis(text)
                        text = smart_lowercase(text)
                        text = remove_specific_whitespace(text)

                        script_type = detect_script(text)

                        #if not is_advertisement(text):
                        cleaned_video_comments.append({
                                'id_commentaire': comment_id,
                                'texte': text,
                                'date_publication': pub_date,
                                'auteur': author,
                                'langue': language,
                                'descripteur': descripteurs,
                                'filtred': True,
                                'cleaned': [],
                                'script': script_type
                            })
                        #else:
                            #print("text à éliminer: " + text)
                            #continue

                    if cleaned_video_comments:
                        # Detect and remove redundant messages within the same video
                        clean_comments_list = [comment['texte'] for comment in cleaned_video_comments]
                        #unique_comments_list = detect_and_remove_redundant_messages(clean_comments_list, similarity_threshold=0.8)
                        unique_comments_list = clean_comments_list
                        #unique_texts = set()
                        unique_cleaned_comments = []
                        for comment in cleaned_video_comments:
                            if comment['texte'] in unique_comments_list : #and comment['texte'] not in unique_texts:
                                unique_cleaned_comments.append(comment)
                                #unique_texts.add(comment['texte'])

                        local_cleaned_videos.append({
                            'id_video': video['id_video'],
                            'titre_video': video['titre_video'],
                            'description_video': video.get('description_video', ''),
                            'hashtags': video.get('hashtags', []),
                            'date_publication': video.get('date_publication'),
                            'lien_video': video.get('lien_video', ''),
                            'emotion': video.get('emotion', []),
                            'commentaires': unique_cleaned_comments,
                            'is_valid': video.get('is_valid'),
                            'corpus': video.get('corpus')
                        })

                if local_cleaned_videos:
                    return {
                        '_id': video_document['_id'],
                        'videos': local_cleaned_videos,
                        'categorie': category,
                        'local_cpt': local_cpt
                    }

            except Exception as e:
                print(f"Error processing document {video_document['_id']}: {e}")
                return None

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_video_document, video_document) for video_document in video_documents]

            for future in as_completed(futures):
                result = future.result()
                if result:
                    cleaned_comments.append({
                        '_id': result['_id'],
                        'videos': result['videos'],
                        'categorie': result['categorie']
                    })
                    cpt += result['local_cpt']

        for video_document in cleaned_comments:
            db.nettoyer.update_one(
                {'_id': video_document['_id']},
                {'$set': video_document},
                upsert=True
            )

        return {'cpt': cpt}

    except Exception as e:
        return str(e)
'''
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------
# sléction des critères  de nettoyage ##
#---------------------------------------
def apply_criteria(comment_text, criteria):
     
    script = detect_script(comment_text)
    
    if 'remove_arabic_letters' in criteria:
        if script == 'Mixed Script':
           comment_text = remove_arabic_letters(comment_text)
    if script != 'Arabic Script':     
        if 'remove_punctuation' in criteria:
            comment_text = remove_punctuation(comment_text)
        if 'convert_emojis' in criteria:
            comment_text = convert_emojis_and_emoticons_to_text(comment_text)
        if 'remove_redundent_words_and_expressions' in criteria:
            comment_text = remove_redundant_words(comment_text, min_count=2)
        if 'remove_stop_words' in criteria:
            comment_text = remove_stop_words(comment_text)
        if 'remove_exaggeration' in criteria:
            comment_text = remove_exaggerations(comment_text)
        if 'replace_abbreviations' in criteria:
            comment_text = normalize_abbreviations(comment_text)
        if 'lemmatize' in criteria:
            comment_text = lemmatize(comment_text,'fr')
            comment_text = lemmatize(comment_text,'en')
        if 'arabizi_lettres_conversion' in criteria:     
            comment_text = normalize_arabizi(comment_text)
        if 'remove_numbers' in criteria:
            comment_text = remove_numbers(comment_text)
        if 'remove_small_words' in criteria:        
            comment_text = remove_minlen_word(comment_text, 2)
        if "normalizing text" in criteria:
            comment_text = process_new_text(comment_text)
        
    return comment_text

#---------------------------------------------------------------------------------------------------------------
