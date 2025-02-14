# PROJ932 - Projet Data Science

## Overview
This project is developed as part of the **PROJ932: Projet Data Science - Data Science Project** course. It provides a **React web application** that integrates with **Gephi Lite** and a **Django backend**, enabling users to run Python scripts dynamically and visualize network graphs.

## Features
- **React Frontend**: A modern web application for user interaction.
- **Django Backend**: Handles the execution of Python scripts and data management.
- **Gephi Lite Integration**: Visualizes network graphs using `.gexf` files.
- **Dynamic Python Script Execution**: Any Python script added to `graphs/python_functions/` will automatically appear in the application.
- **Automated Graph Generation**: Clicking on a script's button executes it, generating a `.gexf` file in `static/gexf/`, which is then displayed in Gephi Lite.

## Technologies Used
- **React**: Frontend UI framework.
- **Django**: Backend framework to handle script execution and data processing.
- **Gephi Lite**: Lightweight graph visualization engine.
- **Docker & Docker Compose**: Containerization for easy deployment.

## Installation and Setup
### Prerequisites
Ensure you have the following installed:
- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/install/)

### Cloning the Repository
```bash
git clone https://github.com/cortesmc/PROJ932_Projet_Data_Science.git
cd PROJ932_Projet_Data_Science
```

### Building and Running the Application
```bash
docker compose build
docker compose up --build -d
```
This will start the React frontend, Django backend, and Gephi Lite containerized services.

### Accessing the Application
Once the services are running, access the dashboard at:
```
http://localhost:8000/dashboard/
```

## Usage Guide
1. **Adding Python Scripts**: Place any `.py` script inside `graphs/python_functions/`.
2. **Script Format Requirement**: Python scripts should follow the `argparse` format for inputting a JSON data file and saving the output `.gexf` file.
   
   Example script:
   ```
   python ./your_python_file.py --json_path "./path_to_folder/your_data.json" --save_path "./path_to_folder/new_gexf.gexf"
   ```

3. **Running a Script**: The React app will automatically detect the new script and display a button with the script’s filename.
4. **Generating a Graph**: Click on a script button to execute it. The script will generate a `.gexf` file in `static/gexf/`.
5. **Viewing the Graph**: The `.gexf` file will be loaded into Gephi Lite for visualization.


