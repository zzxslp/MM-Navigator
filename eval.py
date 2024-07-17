'''
Example evaluation code for reference, one may need to modify the code given how they save the results.
'''

import json
import numpy as np
import jax.numpy as jnp
import pickle
import re

import action_type

def is_tap_action(normalized_start_yx, normalized_end_yx):
	distance = np.linalg.norm(
		np.array(normalized_start_yx) - np.array(normalized_end_yx)
	)
	return distance <= 0.04

def _check_drag_actions_match(
		drag_touch_yx,
		drag_lift_yx,
):
		"""Determines if two drag actions are the same."""
		# Store drag deltas (the change in the y and x coordinates from touch to
		# lift), magnitudes, and the index of the main axis, which is the axis with
		# the greatest change in coordinate value (e.g. a drag starting at (0, 0) and
		# ending at (0.3, 0.5) has a main axis index of 1).
		drag_1_deltas = drag_lift_yx - drag_touch_yx
		drag_1_magnitudes = jnp.abs(drag_1_deltas)
		drag_1_main_axis = np.argmax(drag_1_magnitudes)

		# y axis
		if drag_1_main_axis == 0:
				if drag_1_deltas[0] < 0:
						scroll = "down"
				else:
						scroll = "up"
		elif drag_1_main_axis == 1:
				if drag_1_deltas[1] < 0:
						scroll = "left"
				else:
						scroll = "right"
						
		return scroll

def find_answer(input_text):
	pattern = r'{"action_type": "click", "idx": \d+}|' \
					r'{"action_type": "type", "text": "[^"]+"}|' \
					r'{"action_type": "navigate_home"}|' \
					r'{"action_type": "navigate_back"}|' \
					r'{"action_type": "scroll", "direction": "(?:up|down|left|right)"}'

	action_output = re.search(pattern, input_text)
	if action_output: ## if action with pattern can be found in GPT-4 text
		return action_output.group(0)
	else:
		return None

def _resize_annotation_bounding_boxes(
	annotation_positions, annotation_width_augment_fraction=1.4,
	annotation_height_augment_fraction=1.4):
	"""Resize the bounding boxes by the given fractions.

	Args:
	annotation_positions: Array of shape (N, 4), where each row represents the
	  (y, x, height, width) of the bounding boxes.
	annotation_width_augment_fraction: The fraction to augment the box widths,
	  E.g., 1.4 == 240% total increase.
	annotation_height_augment_fraction: Same as described for width, but for box
	  height.

	Returns:
	Resized bounding box.

	"""
	return annotation_positions


def check_location_match(result_touch_yx, click_box):
	_TAP_DISTANCE_THRESHOLD = 0.14  # Fraction of the screen
	top, left, h, w = click_box
	bottom, right = top+h, left+w
	# print (result_touch_yx, [top, left, bottom, right])
	y1, x1 = result_touch_yx
	## click in box
	# exit()
	if0 = jnp.logical_and(y1 >= top, y1 <= bottom) & jnp.logical_and(x1 >= left, x1 <= right)
	## if within distance
	if1 = (jnp.linalg.norm(jnp.array(result_touch_yx) - jnp.array([top, left]))) <= _TAP_DISTANCE_THRESHOLD
	if2 = (jnp.linalg.norm(jnp.array(result_touch_yx) - jnp.array([top+h,left+w]))) <= _TAP_DISTANCE_THRESHOLD
	if3 = (jnp.linalg.norm(jnp.array(result_touch_yx) - jnp.array([top+h/2,left+w/2]))) <= _TAP_DISTANCE_THRESHOLD
	return if0 or if1 or if2 or if3

def eval_answer(action_output, step_data):
	'''
	action_output: will be converted into dict, in the format {action_type: "", ...}
	step_data: data from current step, dict with keys ui_positions, ui_text, result_touch_yx, ...
	See data_utils.py on how data is saved
	'''
	if not action_output: ## no predictions found from GPT-4 output
		action_output = "{'action_type': 'invalid'}"
	action_output = eval(action_output)
	# print (step_data.keys())
	result_touch_yx = step_data["result_touch_yx"]
	result_lift_yx = step_data["result_lift_yx"]
	result_action = step_data["result_action"][0]
	result_text = step_data["result_action"][1]
	result_text = result_text.replace("\\", "").replace('"','').replace("'","")

	if_action, if_text, if_action_type = 0, 0, 0
	if result_action in ["STATUS_TASK_COMPLETE", "STATUS_TASK_IMPOSSIBLE", "PRESS_ENTER"]:
		return 1, 1, 1
	elif result_action == "DUAL_POINT":
		action_touch_yx = jnp.asarray(result_touch_yx)
		action_lift_yx = jnp.asarray(result_lift_yx)
		if is_tap_action(action_touch_yx, action_lift_yx):
			result_touch_yx = [round(axis, 4) for axis in result_touch_yx]
			# if click, the lift can be the same as touch
			if action_output["action_type"] == "click":
				if_action_type += 1 
				if_text += 1 # no text to compare
				resized_positions = _resize_annotation_bounding_boxes(step_data["ui_positions"])
				try:
					click_box = resized_positions[action_output["idx"]-1]
				except:
					click_box = resized_positions[0]
				if check_location_match(result_touch_yx, click_box) or check_location_match(result_lift_yx, click_box):
					
					if_action += 1
				
		else: ## if scroll
			if action_output["action_type"] == "scroll":
				if_action_type += 1
				if_text += 1
				if action_output["direction"] == _check_drag_actions_match(action_touch_yx, action_lift_yx):
					if_action += 1
	elif result_action == "TYPE":
		if action_output["action_type"] == "type":
			if_action_type += 1
			if_action += 1
			if action_output["text"].lower() == result_text: 
				if_text += 1
	elif result_action == "PRESS_BACK" and action_output["action_type"]=="navigate_back":
		if_action += 1
		if_action_type += 1
		if_text += 1
	elif result_action == "PRESS_HOME" and action_output["action_type"]=="navigate_home":
		if_action += 1
		if_action_type += 1
		if_text += 1
	return if_action, if_text, if_action_type

results_dir = "gpt4_results/"
data_dir = "test_set/"
gpt_preds = {}
for category in ["general", "google_apps", "install", "single", "web_shopping"]: 
	with open(f'{results_dir}/test_results_{category}.json', 'r') as file:		
		tmp_preds = json.load(file)
		tmp_preds[category] = dict(list(tmp_preds[category].items())[:10])
		gpt_preds.update(tmp_preds) 


## first compute partial matching score of each episode, then average over episodes
total_num = 0
category_action_scores, category_text_scores, category_type_scores = {}, {}, {}

for category in gpt_preds.keys():
	category_results = gpt_preds[category]
	category_dir = f"{data_dir}/{category}"
	action_scores, text_scores, type_scores = [], [], []
	print (f"Evaluate {category}", len(category_results))
	for ep_id in category_results.keys():
		action_correct, text_correct, type_correct, step_cnt = 0, 0, 0, 0
		ep_dir = f"{category_dir}/{ep_id}/"
		with open(f"{ep_dir}/data.obj", "rb") as rp:
			ep_data = pickle.load(rp)["data"]

		for step_id in range(len(ep_data)):
			if category_results[ep_id][step_id] == "response filtered":
				continue
			action_output = find_answer(category_results[ep_id][step_id])
			step_data = ep_data[step_id]
			if_action, if_text, if_action_type = eval_answer(action_output, step_data)
			action_correct += if_action
			text_correct += if_text
			type_correct += if_action_type
			step_cnt += 1
			total_num += 1
		if step_cnt != 0:
			action_scores.append(action_correct/step_cnt)
			text_scores.append(text_correct/step_cnt)
			type_scores.append(type_correct/step_cnt)
	category_action_scores[category] = (sum(action_scores)/len(action_scores))
	category_text_scores[category] = (sum(text_scores)/len(text_scores))
	category_type_scores[category] = (sum(type_scores)/len(type_scores))

for k,v in category_action_scores.items():
	print (k, v)

metrics = {}
metrics["avg accuracy"] = "{:.2f}".format(sum(category_action_scores.values())/len(category_action_scores) * 100)
metrics["text_acc"] = "{:.2f}".format(sum(category_text_scores.values())/len(category_text_scores) * 100)
metrics["type_acc"] = "{:.2f}".format(sum(category_type_scores.values())/len(category_type_scores) * 100)
metrics["total_episodes"] = len(action_scores)
metrics["total_steps"] = total_num

for k,v in metrics.items():
	print (k, v)