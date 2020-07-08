import os
os.system("pip install -r ./requirements.txt")
os.chdir("./lib")
print("Install APEX [y/n]:")
key = input()
if key == 'y':
    os.system("git clone https://github.com/NVIDIA/apex.git")
    os.system("python ./apex/setup.py install --cpp_ext --cuda_ext")
elif key == 'n':
    pass
else:
    raise ValueError('Invalid Input')

