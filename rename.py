import os
path = '/home/shreya/Documents/envs/mountains/bin/pics/dhaulagiri'
files = os.listdir(path)
i = 1

for file in files:
    os.rename(os.path.join(path, file), os.path.join(path,'dhaulagiri-'+str(i)+'.jpg'))
    i = i+1