import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/braincoder-extreme-nas/hanzhang/llama-factory/evaluation_ID/generated_predictions_lr2e7.jsonl')
args = parser.parse_args()

# print(json.dumps(data[0], indent=2))
# print(data[0]['predict'].replace('<tool_call>', '').replace('</tool_call>', ''))
data_path = args.data_path
data = [json.loads(line) for line in open(data_path, 'r')]
def extract_actions(text):
    """从文本中提取所有动作信息"""
    actions = []
    # 分割多个tool_call
    parts = text.split('</tool_call>')
    for part in parts:
        if '<tool_call>' not in part:
            continue
        # 提取JSON部分
        json_str = part.split('<tool_call>')[1].strip()
        try:
            data = json.loads(json_str)
            # 支持多种工具名称：computer_use, mobile_use 等
            if data['name'] in ['computer_use', 'mobile_use']:
                action_data = {
                    'action': data['arguments']['action'],
                    'coordinate': data['arguments'].get('coordinate', None)
                }
                actions.append(action_data)
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
    print(actions)
    return actions

def compare_actions(predict_actions, label_actions):
    """比较预测动作和标签动作"""
    comparisons = []
    for i, (pred, lbl) in enumerate(zip(predict_actions, label_actions), 1):
        comparison = {
            'action_number': i,
            'action_type_match': pred['action'] == lbl['action'],
            'coordinate_comparison': None  # 默认设置为None
        }

        if pred['coordinate'] and lbl['coordinate']:
            comparison['coordinate_comparison'] = {
                'pred_coordinate': pred['coordinate'],
                'label_coordinate': lbl['coordinate'],
                'x_diff': abs(pred['coordinate'][0] - lbl['coordinate'][0]),
                'y_diff': abs(pred['coordinate'][1] - lbl['coordinate'][1]),
                'total_diff': ((pred['coordinate'][0] - lbl['coordinate'][0])**2 +
                              (pred['coordinate'][1] - lbl['coordinate'][1])**2)**0.5
            }

        comparisons.append(comparison)
    return comparisons
exact_match = 0
soft_match_10 = 0
soft_match_20 = 0
action_match = 0
total_action = 0
total_coordinate = 0
total_euclidean_distance = 0  # 欧氏距离总和
diff_gap_10 = 10
diff_gap_20 = 20

for tmp_data in data[:]:
    predict_actions = extract_actions(tmp_data['predict'])
    label_actions = extract_actions(tmp_data['label'])
    results = compare_actions(predict_actions, label_actions)
    for item in results:
        total_action += 1
        if item['action_type_match']:
            action_match += 1
        if item['coordinate_comparison'] is not None:
            total_coordinate += 1
            # 累加欧氏距离
            total_euclidean_distance += item['coordinate_comparison']['total_diff']
            if item['coordinate_comparison']['x_diff'] == 0 and item['coordinate_comparison']['y_diff'] == 0:
                exact_match += 1
            if item['coordinate_comparison']['x_diff'] <= diff_gap_10 and item['coordinate_comparison']['y_diff'] <= diff_gap_10:
                soft_match_10 += 1
            if item['coordinate_comparison']['x_diff'] <= diff_gap_20 and item['coordinate_comparison']['y_diff'] <= diff_gap_20:
                soft_match_20 += 1
# 计算平均欧氏距离
avg_euclidean_distance = total_euclidean_distance / total_coordinate if total_coordinate > 0 else 0

print(total_coordinate, exact_match, soft_match_10, soft_match_20, action_match, total_action)
print(f'exact_match_rate: {exact_match/total_coordinate}, soft_match_rate_10: {soft_match_10/total_coordinate}, soft_match_rate_20: {soft_match_20/total_coordinate}, action_match_rate: {action_match/total_action}, total_action: {total_action}, total_coordinate: {total_coordinate}')
print(f'average_euclidean_distance: {avg_euclidean_distance:.2f} pixels')