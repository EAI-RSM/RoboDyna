import sys, os, traceback, yaml
os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
os.environ.pop("DISPLAY", None)
os.chdir("/shared_work/markhsp/DOMINO")
sys.path.insert(0, "/shared_work/markhsp/DOMINO")
sys.path.insert(0, "/shared_work/markhsp/DOMINO/script")
import collect_data as cd

task_name, task_config, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
task = cd.class_decorator(task_name)
args = yaml.load(open(f"./task_config/{task_config}.yml").read(), Loader=yaml.FullLoader)
args["task_name"] = task_name
emb = args.get("embodiment")
_emb = yaml.load(open(os.path.join(cd.CONFIGS_PATH, "_embodiment_config.yml")).read(), Loader=yaml.FullLoader)
gef = lambda e: _emb[e]["file_path"]
if len(emb) == 1:
    args["left_robot_file"] = gef(emb[0]); args["right_robot_file"] = gef(emb[0]); args["dual_arm_embodied"] = True
    args["embodiment_name"] = str(emb[0])
else:
    args["left_robot_file"] = gef(emb[0]); args["right_robot_file"] = gef(emb[1])
    args["embodiment_dis"] = emb[2]; args["dual_arm_embodied"] = False
    args["embodiment_name"] = f"{emb[0]}+{emb[1]}"
args["left_embodiment_config"] = cd.get_embodiment_config(args["left_robot_file"])
args["right_embodiment_config"] = cd.get_embodiment_config(args["right_robot_file"])
args["task_config"] = task_config
args["save_path"] = os.path.join(args["save_path"], task_name, task_config)
try:
    task.setup_demo(now_ep_num=0, seed=seed, **args)
    task.play_once()
    print(f"REPRO_OK plan_success={getattr(task,'plan_success',None)} check_success={task.check_success()}")
except Exception:
    print("REPRO_EXC:")
    traceback.print_exc()
finally:
    try: task.close_env()
    except Exception: pass
