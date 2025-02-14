from django.db import models

class Node(models.Model):
    NODE_TYPES = [
        ('person', 'Personne'),
        ('organization', 'Organisation'),
        ('location', 'Lieu'),
    ]
    name = models.CharField(max_length=100, unique=True) 
    node_type = models.CharField(max_length=50, choices=NODE_TYPES)
    timestamp = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"{self.name} ({self.get_node_type_display()})"


class Edge(models.Model):
    RELATION_TYPES = [
        ('diplomatic', 'Diplomatique'),
        ('economic', 'Économique'),
        ('humanitarian', 'Humanitaire'),
        ('undefined', 'Non défini'),  # Par défaut si un type n'est pas spécifié
    ]
    from_node = models.ForeignKey(Node, related_name='from_edges', on_delete=models.CASCADE)
    to_node = models.ForeignKey(Node, related_name='to_edges', on_delete=models.CASCADE)
    relation_type = models.CharField(max_length=50, choices=RELATION_TYPES, default='undefined')
    weight = models.FloatField(default=1.0)

    class Meta:
        unique_together = ('from_node', 'to_node', 'relation_type')  # Empêcher les doublons exacts

    def __str__(self):
        return f"{self.from_node} -> {self.to_node} ({self.get_relation_type_display()}, poids: {self.weight})"
