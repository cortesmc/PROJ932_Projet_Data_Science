from django.contrib import admin
from .models import Node, Edge

# Enregistre les modèles pour les rendre disponibles dans l'admin
admin.site.register(Node)
admin.site.register(Edge)
