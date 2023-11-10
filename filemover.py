import os

with open('./txt/all.txt','r') as f:
    all_lines = f.readlines()
if not os.path.exists('img/0'):
    os.makedirs('img/0')
if not os.path.exists('img/1'):
    os.makedirs('img/1')
for line in all_lines:
    file,classifier = line.split()
    try:
        curr_path = f'./resized_224/{file}'
        out_path = f'./img/{classifier}/{file}'
        os.rename(curr_path,out_path)

    except Exception as e:
        print(e)
    # os.rename(out_path,curr_path)
    # break
