# Learning to optimize ML models throughput
This repository contains code to develop and test optimized scripts to run YOLO models. My primary goal is to learn and document how to use various SOTA techniques to squeeze performance out of these models. I am starting with YOLO v3 as that is a relatively simple model to code (from scratch). 
All FPS reported here is dataloading + forward + prediction + writing results unless mentioned otherwise. I am testing on my laptop's 1660ti GTX (6GB).

For Torchscript model see the torchscript branch, I didn't noticed any speedup running the scripted+traced pipeline yet, still working on it. (model.forward is traced and prediction function is scripted).

Some things that has worked so far:
- Optimizing transpose and reshape operations (in logits_to_preds function) (7 FPS -> 9 FPS)
- Caching mesh_grid for YOLO layers (9 FPS -> 11 FPS)

Some tools I am using :
- [Line Profiler](https://pypi.org/project/line-profiler/) : For checking which code piece is taking unreasonable amount of time.

In case you are interested pure forward + prediction FPS is about 18 after all optimizations done above.

Yolov3 official weights : https://pjreddie.com/media/files/yolov3.weights
<br>
<br>
WIP, you can look around or visit later for more details!
