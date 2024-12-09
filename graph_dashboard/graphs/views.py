from django.http import JsonResponse
from django.shortcuts import render
from .models import Node, Edge

def graph_data(request):
    # Récupérer les nœuds
    nodes = list(Node.objects.values('id', 'name', 'node_type'))

    # Récupérer les relations
    edges = list(Edge.objects.values('from_node_id', 'to_node_id', 'relation_type', 'weight'))

    return JsonResponse({'nodes': nodes, 'edges': edges})

def graph_view(request):
    return render(request, 'graph.html')