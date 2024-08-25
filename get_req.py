import re

# match python lib version: lib==1.3.0
pattern = r"^([a-zA-Z0-9._-]+)==([\d.]+)"

with open("requirements.txt", "r") as f:
    lines = f.readlines()
    
data = []
for line in lines:
    match = re.match(pattern, line)
    if match:
        lib, version = match.groups()
        print(f"Library: {lib}, Version: {version}")
        data.append("{lib}=={version}".format(lib=lib, version=version))
        
with open("new_requirements.txt", "w") as f:
    f.write("\n".join(data))