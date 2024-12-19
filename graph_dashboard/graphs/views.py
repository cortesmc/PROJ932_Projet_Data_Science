import os
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render

def graph_dashboard(request):
    data_dir = os.path.join(settings.BASE_DIR, 'graphs', 'data')
    
    file_names = []
    if os.path.exists(data_dir):
        file_names = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

    return render(request, 'upload_graph.html', {'file_names': file_names})

def upload_file(request):
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        file_name = uploaded_file.name

        # Check if the file is a valid type
        if not (file_name.endswith('.json') or file_name.endswith('.txt')):
            return JsonResponse({'success': False, 'message': 'Invalid file type. Only .json and .txt files are allowed.'})

        # Save the uploaded file
        save_path = os.path.join(settings.BASE_DIR, 'graphs', 'data')
        os.makedirs(save_path, exist_ok=True)

        file_path = os.path.join(save_path, file_name)
        with open(file_path, 'wb') as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)

        # Get the updated file list
        file_names = [f for f in os.listdir(save_path) if os.path.isfile(os.path.join(save_path, f))]

        # Return the success message and updated file list
        return JsonResponse({'success': True, 'message': f'File "{file_name}" uploaded successfully.', 'file_names': file_names})

    return JsonResponse({'success': False, 'message': 'Invalid request. File not found.'})
