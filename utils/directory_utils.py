import os

def create_new_run_directory(model_path: str = None):
    # Directory where your PPO runs are stored
    runs_dir = 'chess_ppo_model' if not model_path else model_path

    # Get the current highest run number
    current_run_number = 0
    for folder_name in os.listdir(runs_dir):
        if folder_name.startswith('run_'):
            run_number = int(folder_name.split('_')[1])
            current_run_number = max(current_run_number, run_number)

    # Create new run folder
    new_run_number = current_run_number + 1
    new_run_folder = os.path.join(runs_dir, f'run_{new_run_number}')
    os.makedirs(new_run_folder)

    return new_run_folder

