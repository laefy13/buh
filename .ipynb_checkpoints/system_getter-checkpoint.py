import subprocess
from datetime import datetime

timenow = datetime.now()

with open(f'./logs/system_info{timenow}.txt','w') as f:
    nvidia_smi_output = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    f.write("Nvidia SMI Output:")
    f.write(nvidia_smi_output.stdout)

    # Get the output of python --version
    python_version_output = subprocess.run(['python', '--version'], capture_output=True, text=True)
    f.write("\nPython Version:")
    f.write(python_version_output.stdout)

    # Get the output of pip list
    pip_list_output = subprocess.run(['pip', 'list'], capture_output=True, text=True)
    f.write("\nPIP List:")
    f.write(pip_list_output.stdout)

    # Get the output of nvcc --version
    nvcc_version_output = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
    f.write("\nNVCC Version:")
    f.write(nvcc_version_output.stdout)