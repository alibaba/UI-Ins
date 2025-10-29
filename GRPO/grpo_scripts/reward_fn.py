import numpy as np
import os
import datetime
import re

def parse_coordinates(raw_string):
    matches = re.findall(r'\[(\d+),(\d+)\]', raw_string)
    matches = [tuple(map(int, match)) for match in matches]
    if len(matches) > 1:
        return -1,-1
    else:
        return matches[0]

def click_reward(data_source, solution_str, ground_truth, extra_info=None):
    def isin(x,y, gt):
        boxs,ratio_h,ratio_w = gt[0:4], gt[4], gt[5]
        if not isinstance(boxs[0], list):
            boxs = [boxs]
        for box in boxs:
            x0,y0,x1,y1 = box
            x0 = int(x0*ratio_w)
            y0 = int(y0*ratio_h)
            x1 = int(x1*ratio_w)
            y1 = int(y1*ratio_h)
            if x<=x1 and x>=x0:
                if y<=y1 and y>=y0:
                    return True
        return False

    reward = 0.0
    solution_str_list = solution_str.split("\n")
    print_str = "solution str: "
    for part in solution_str_list:
        print_str += part + " "
    print_str += "ground truth: " + str(ground_truth) 

    try:
        pred_x, pred_y = parse_coordinates(solution_str)
        if isin(pred_x, pred_y, ground_truth):
            reward = 1.0
        print("reward: ",reward)
    except Exception:
        print("parse coordinates error")
        pass 
                
    if os.getenv("DEBUG_MODE") == "true":
        with open("/workspace/verl/verl/gta1/debug.txt", "a") as f:
            timestamp = datetime.datetime.now().isoformat()
            f.write(f"{timestamp}\tsolution_str: {repr(solution_str)}\tground_truth: {repr(ground_truth)}\treward: {reward}\n")
    return reward
