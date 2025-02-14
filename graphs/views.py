import os
import subprocess
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render

def graph_dashboard(request):
    # Construct absolute paths
    data_dir = os.path.join(settings.BASE_DIR, 'graphs', 'data')
    python_functions_dir = os.path.join(settings.BASE_DIR, 'graphs', 'python_functions')

    file_names = []
    python_files = []

    # Ensure data directory exists before listing files
    if os.path.exists(data_dir):
        file_names = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

    # Ensure python_functions_dir exists before listing files
    if os.path.exists(python_functions_dir):
        python_files = [f for f in os.listdir(python_functions_dir) if f.endswith('.py')]

    return render(request, 'upload_graph.html', {
        'file_names': file_names,
        'python_files': python_files
    })

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

def generate_graph(request):
    if request.method == 'POST':
        file_name = request.POST.get('file_name')
        python_file = request.POST.get('python_file')
        
        if not file_name:
            return JsonResponse({'success': False, 'message': 'No file selected.'})
        
        file_path = os.path.join(settings.BASE_DIR, 'graphs', 'data', file_name)
        if not os.path.exists(file_path):
            return JsonResponse({'success': False, 'message': f'File {file_name} not found.'})

        name_only, _ = os.path.splitext(file_name)

        # Directory to save .gexf files
        save_dir = os.path.join(settings.BASE_DIR, 'static', 'gexf')
        os.makedirs(save_dir, exist_ok=True)

        # Generate unique file name with index if necessary
        index = 0
        save_name = f"{name_only}.gexf"
        save_path = os.path.join(save_dir, save_name)
        
        while os.path.exists(save_path):
            index += 1
            save_name = f"{name_only}_{index}.gexf"
            save_path = os.path.join(save_dir, save_name)

        try:
            # Run the Python script with the correct save path
            command = [
                'python',
                f'graphs/python_functions/{python_file}',
                '--json_path', file_path,
                '--save_path', save_path
            ]
            
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            
            return JsonResponse({
                'success': True,
                'message': f'Graph generated successfully: {save_name}',
                'output': result.stdout,
                'gexf_file': save_name
            })
        
        except subprocess.CalledProcessError as e:
            error_message = f"Error generating graph:\n{e.stderr if e.stderr else e.stdout}"
            return JsonResponse({'success': False, 'message': error_message})

    return JsonResponse({'success': False, 'message': 'Invalid request.'})