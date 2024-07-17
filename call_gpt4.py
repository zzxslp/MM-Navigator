'''
Example code to call gpt4
Input: two screens + self-summary of history actions (in a 2-step manner, action + summary)
'''
import os
import json
import cv2
import base64
import requests
import pickle
import time

api_url = ''
headers = {}


def run_api(body):
    '''
    API call, check https://platform.openai.com/docs/guides/vision for the latest api usage. 
    '''
    response = requests.post(api_url, headers=headers, json=body)
    response_json = response.json()

    return response_json["choices"][0]["message"]["content"]

def load_image(img_path):
    '''
    load png images
    '''
    img = cv2.imread(img_path)
    img_encoded_bytes = base64.b64encode(cv2.imencode('.jpg', img)[1])
    img_encoded_str = img_encoded_bytes.decode('utf-8')
    return img_encoded_str

def load_screen(step_data):
    '''
    conver screen information into html format
    '''
    screen_info = ""
    for idx, (ui_type, ui_text) in enumerate(zip(step_data["ui_type"], step_data["ui_text"]), 1):
        if ui_type == "TEXT":
            screen_info += f'''<p id={idx} class="text" alt="{ui_text}"> {ui_text} </p>\n'''
        else:
            screen_info += f'''<img id={idx} class="{ui_type}" alt="{ui_text}"> </img>\n'''
    return screen_info


def build_input_body(ep_dir, step_id, tag_mode, INSTRUCTION, history):
    if not history:
        history = "This is step 0 so no history."
    img_raw = load_image(f"{ep_dir}/raw/{step_id}.png")
    img_tag = load_image(f"{ep_dir}/{tag_mode}/{step_id}.png")

    body = [{ 'role' : 'system',
               'content' : ['''You are an expert at completing instructions on Android phone screens. 
               You will be presented with two images. The first is the original screenshot. The second is the same screenshot with some numeric tags.
               If you decide to click somewhere, you should choose the numeric idx that is the closest to the location you want to click.  
               The screenshot are most likely an intermediate step of this intruction, so in most cases there is no need to navigate back home. 
               You should decide the action to continue this instruction.
               Here are the available actions:
{"action_type": "click", "idx": <element_idx chosen from the second screen>}
{"action_type": "type", "text": <the text to enter>}
{"action_type": "navigate_home"}
{"action_type": "navigate_back"}
{"action_type": "scroll", "direction": "up"}
{"action_type": "scroll", "direction": "down"}
{"action_type": "scroll", "direction": "left"}
{"action_type": "scroll", "direction": "right"}.
Your final answer must be in the above format.
'''],},
    { 'role' : 'user',
      'content' : [f'''
      The instruction is to {INSTRUCTION}. 
      History actions:
      {history}
      Think about what you need to do with current screen, and output the action in the required format in the end. ''', 
      {"image": img_raw}, {"image": img_tag}],  
    },
    ]
    return body

def continue_chat(body, gpt_output):
    body.append({ 'role' : 'assistant',
      'content': [gpt_output]},)
    body.append({ 'role' : 'user',
      'content' : ["Summarize your actions so far (history actions + the action you just take) in 1-2 sentences. Be as concise as possible."]
    })
    return body

root_dir = "test_set/"

results_dir = "gpt4_results/"
tag_mode = 'tagscenter_nobox' 
for category in ["general", "google_apps", "install", "single", "web_shopping"]:
    json_dir = f'{results_dir}/test_results_{category}.json'
    if os.path.exists(json_dir):
        with open(json_dir, 'r') as file:
            gpt_preds = json.load(file)
    else:
        gpt_preds = {}
    print (f"Processing {category}")
    category_dir = f"{root_dir}/{category}"
    if category not in gpt_preds:
        gpt_preds[category] = {}
    for i, ep_id in enumerate(os.listdir(category_dir), 1):
        if ep_id in gpt_preds[category].keys():
            print (f"Skip {ep_id} idx:{i}")
            continue
        ep_dir = f"{category_dir}/{ep_id}/"
        with open(f"{ep_dir}/data.obj", "rb") as rp:
            data = pickle.load(rp)
            goal = data["goal"]
            episode_data = data["data"]
        episode_len = len(episode_data)
        print (f"Processing {ep_id}, Goal:{goal}, idx:{i}")
        gpt_preds[category][ep_id] = []
        history = None
        for step_id in range(episode_len):
            inputs = {}
            inputs['messages'] = build_input_body(ep_dir, step_id, tag_mode, goal, history)
            inputs['max_tokens'] = 1024
            gpt_output = run_api(inputs)

            gpt_preds[category][ep_id].append(gpt_output)

            next_inputs = {} 
            next_inputs['messages'] = continue_chat(inputs['messages'], gpt_output)
            next_inputs['max_tokens'] = 1024
            history = run_api(next_inputs, ep_id, step_id, goal)

        with open(json_dir, 'w') as file:
            json.dump(gpt_preds, file)


