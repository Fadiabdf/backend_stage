from mongoengine import Document, EmbeddedDocument, fields
class Commentaire(EmbeddedDocument):
    id_commentaire = fields.StringField(required=True)
    texte = fields.StringField()
    date_publication = fields.DateTimeField()
    auteur = fields.StringField()
    langue = fields.ListField(fields.StringField())  # Array of strings
    descripteur = fields.ListField(fields.StringField())  # Array of strings
    cleaned = fields.ListField(fields.DictField())  # Array of dictionaries with str and bool
    filtred= fields.BooleanField(default=False)
    script= fields.StringField(default="script")

class Etiquette(Document):  # Renamed from Emotion
    name = fields.StringField(required=True)
    color = fields.StringField(required=True)
    category = fields.StringField(required=True)  # New field for category

class Video(EmbeddedDocument):
    id_video = fields.StringField(required=True)
    titre_video = fields.StringField()
    description_video = fields.StringField()
    hashtags = fields.ListField(fields.StringField())  # Array of strings
    date_publication = fields.DateTimeField()
    lien_video = fields.URLField()
    annotation_video = fields.StringField()
    emotion = fields.ListField(fields.StringField())  # Array of strings
    commentaires = fields.EmbeddedDocumentListField(Commentaire)
    is_valid = fields.BooleanField(default=False)
    corpus=fields.StringField()

class VideoCollection(Document):
    videos = fields.EmbeddedDocumentListField(Video)
     
    categorie = fields.ListField(fields.StringField())  

class Category(Document):
    name = fields.StringField(required=True)
    emotions = fields.ListField(fields.StringField(), default=[]) 
    multiple_choice = fields.BooleanField(default=False)


class Corpus(Document):
    title = fields.StringField(required=True)
    description = fields.StringField(required=False) 
    categories = fields.ListField(fields.ReferenceField(Category))  # List of Category references
    videos = fields.ListField(fields.EmbeddedDocumentField(Video))  # List of Video embedded documents