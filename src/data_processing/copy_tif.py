import shutil
def list_tif_predict(file):
    root = '/gpfs/projects/LynchGroup/Orthoed/'
    imlist =[]
    imnamelist =[]
    f = open(file,'r')
    while True:
        line = f.readline()
        if not line:break
        imnamelist.append(line.split()[0] )
    print(imnamelist)

    for name in imnamelist :
        try:
            shutil.copyfile(root+name,'./tif_data/'+name)
        except:
            print('error')

list_tif_predict('full.txt')
