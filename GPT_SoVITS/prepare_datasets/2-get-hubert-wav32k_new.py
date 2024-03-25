# -*- coding: utf-8 -*-

from queue import Queue
from threading import Lock, Thread
import sys,os
inp_text=                           os.environ.get("inp_text")
inp_wav_dir=                        os.environ.get("inp_wav_dir")
exp_name=                           os.environ.get("exp_name")
i_part=                             os.environ.get("i_part")
all_parts=                          os.environ.get("all_parts")
os.environ["CUDA_VISIBLE_DEVICES"]= os.environ.get("_CUDA_VISIBLE_DEVICES")
from feature_extractor import cnhubert
opt_dir=                            os.environ.get("opt_dir")
cnhubert.cnhubert_base_path=                os.environ.get("cnhubert_base_dir")
is_half=eval(os.environ.get("is_half","True"))

import pdb,traceback,numpy as np,logging
from scipy.io import wavfile
import librosa,torch
now_dir = os.getcwd()
sys.path.append(now_dir)
from my_utils import load_audio

# from config import cnhubert_base_path
# cnhubert.cnhubert_base_path=cnhubert_base_path
# inp_text=sys.argv[1]
# inp_wav_dir=sys.argv[2]
# exp_name=sys.argv[3]
# i_part=sys.argv[4]
# all_parts=sys.argv[5]
# os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[6]
# cnhubert.cnhubert_base_path=sys.argv[7]
# opt_dir="/data/docker/liujing04/gpt-vits/fine_tune_dataset/%s"%exp_name

from time import sleep, time as ttime
import shutil
def my_save(fea,path):#####fix issue: torch.save doesn't support chinese path
    dir=os.path.dirname(path)
    name=os.path.basename(path)
    # tmp_path="%s/%s%s.pth"%(dir,ttime(),i_part)
    tmp_path="%s%s.pth"%(ttime(),i_part)
    torch.save(fea,tmp_path)
    shutil.move(tmp_path,"%s/%s"%(dir,name))

hubert_dir="%s/4-cnhubert"%(opt_dir)
wav32dir="%s/5-wav32k"%(opt_dir)
os.makedirs(opt_dir,exist_ok=True)
os.makedirs(hubert_dir,exist_ok=True)
os.makedirs(wav32dir,exist_ok=True)

maxx=0.95
alpha=0.5
if torch.cuda.is_available():
    device = "cuda:0"
# elif torch.backends.mps.is_available():
#     device = "mps"
else:
    device = "cpu"
model=cnhubert.get_model()
# is_half=False
if(is_half==True):
    model=model.half().to(device)
else:
    model = model.to(device)
    
data_queue = Queue(100)
pre_queue = Queue(100)
post_queue = Queue(100)
lock = Lock()
nan_fails=[]
num_tasks=0
def name2go(wav_name,wav_path):
    hubert_path="%s/%s.pt"%(hubert_dir,wav_name)
    if(os.path.exists(hubert_path)):return
    tmp_audio = load_audio(wav_path, 32000)
    tmp_max = np.abs(tmp_audio).max()
    if tmp_max > 2.2:
        print("%s-filtered,%s" % (wav_name, tmp_max))
        return
    tmp_audio32 = (tmp_audio / tmp_max * (maxx * alpha*32768)) + ((1 - alpha)*32768) * tmp_audio
    tmp_audio32b = (tmp_audio / tmp_max * (maxx * alpha*1145.14)) + ((1 - alpha)*1145.14) * tmp_audio
    tmp_audio = librosa.resample(
        tmp_audio32b, orig_sr=32000, target_sr=16000
    )#不是重采样问题
    tensor_wav16 = torch.from_numpy(tmp_audio)
    if (is_half == True):
        tensor_wav16=tensor_wav16.half().to(device)
    else:
        tensor_wav16 = tensor_wav16.to(device)
    ssl=model.model(tensor_wav16.unsqueeze(0))["last_hidden_state"].transpose(1,2).cpu()#torch.Size([1, 768, 215])
    if np.isnan(ssl.detach().numpy()).sum()!= 0:
        nan_fails.append(wav_name)
        print("nan filtered:%s"%wav_name)
        return
    wavfile.write(
        "%s/%s"%(wav32dir,wav_name),
        32000,
        tmp_audio32.astype("int16"),
    )
    my_save(ssl,hubert_path )
    
def pre_process():
    print("pre_process start")
    counter=0
    while counter!=num_tasks:
        if data_queue.empty():
            sleep(0.01)
            continue
        input=data_queue.get()
        counter+=1
        wav_name = input["wav_name"]
        wav_path = input["wav_path"]
        
        hubert_path="%s/%s.pt"%(hubert_dir,wav_name)
        if(os.path.exists(hubert_path)):continue
        tmp_audio = load_audio(wav_path, 32000)
        tmp_max = np.abs(tmp_audio).max()
        if tmp_max > 2.2:
            print("%s-filtered,%s" % (wav_name, tmp_max))
            continue
        tmp_audio32 = (tmp_audio / tmp_max * (maxx * alpha*32768)) + ((1 - alpha)*32768) * tmp_audio
        tmp_audio32b = (tmp_audio / tmp_max * (maxx * alpha*1145.14)) + ((1 - alpha)*1145.14) * tmp_audio
        tmp_audio = librosa.resample(
            tmp_audio32b, orig_sr=32000, target_sr=16000
        )#不是重采样问题
        tensor_wav16 = torch.from_numpy(tmp_audio)
        if (is_half == True):
            tensor_wav16=tensor_wav16.half().to(device)
        else:
            tensor_wav16 = tensor_wav16.to(device)
            
        res={
            "hubert_path":hubert_path,
            "tmp_audio32":tmp_audio32,
            "wav_name":wav_name,
            "tensor_wav16":tensor_wav16,
        }
        pre_queue.put(res)

def extract_freature():
    print("extract_freature start")
    counter=0
    while counter!=num_tasks:
        if pre_queue.empty():
            sleep(0.01)
            continue
        res=pre_queue.get()
        
        ssl=model.model(res["tensor_wav16"].unsqueeze(0))["last_hidden_state"].transpose(1,2).cpu()#torch.Size([1, 768, 215])
        
        _res={
            "hubert_path":res["hubert_path"],
            "tmp_audio32":res["tmp_audio32"],
            "wav_name":res["wav_name"],
            "tensor_wav16":res["tensor_wav16"],
            "ssl":ssl
        }
        post_queue.put(_res)
        del res
        counter+=1
final = False
def post_process():
    print("post_process start")
    global final
    counter=0
    while counter!=num_tasks:
        if post_queue.empty():
            sleep(0.01)
            continue
        res=post_queue.get()
        
        counter+=1
        if np.isnan(res["ssl"].detach().numpy()).sum()!= 0:
            nan_fails.append(res["wav_name"])
            print("nan filtered:%s"%res["wav_name"])
            del res
            continue
        
        wavfile.write(
            "%s/%s"%(wav32dir,res["wav_name"]),
            32000,
            res["tmp_audio32"].astype("int16"),
        )
        my_save(res["ssl"],res["hubert_path"] )
        del res
    final = True

with open(inp_text,"r",encoding="utf8")as f:
    lines=f.read().strip("\n").split("\n")
    
num_tasks=len(lines[int(i_part)::int(all_parts)])

pre_task = Thread(target=pre_process)
extract_task = Thread(target=extract_freature)
post_task = Thread(target=post_process)
pre_task.start()
extract_task.start()
post_task.start()

for line in lines[int(i_part)::int(all_parts)]:
    try:
        # wav_name,text=line.split("\t")
        wav_name, spk_name, language, text = line.split("|")
        if (inp_wav_dir != "" and inp_wav_dir != None):
            wav_name = os.path.basename(wav_name)
            wav_path = "%s/%s"%(inp_wav_dir, wav_name)

        else:
            wav_path=wav_name
            wav_name = os.path.basename(wav_name)
        # name2go(wav_name,wav_path)
        data_queue.put({"wav_name":wav_name,"wav_path":wav_path})
    except:
        print(line,traceback.format_exc())
while not final:
    sleep(1)


final = False
num_tasks=len(nan_fails)
pre_task = Thread(target=pre_process)
extract_task = Thread(target=extract_freature)
post_task = Thread(target=post_process)
pre_task.start()
extract_task.start()
post_task.start()
if(len(nan_fails)>0 and is_half==True):
    is_half=False
    model=model.float()
    for wav_name in nan_fails:
        try:
            # name2go(wav_name)
            data_queue.put({"wav_name":wav_name,"wav_path":wav_path})
        except:
            print(wav_name,traceback.format_exc())
while not final:
    sleep(1)