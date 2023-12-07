import requests
import json

API_URL = "http://localhost:8001/filter"

def call_filter_api(url, html_page, objective):
    data = {
        "html_page": html_page,
        "objective": objective
    }
    response = requests.post(url, json=data)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

def constuct_obs(obs, intent):
    try:
        results = call_filter_api(API_URL, obs, intent)
        results_list = results['results']

    except Exception as e:
        print(f"API call failed: {e}")
        return None
    
    obs_list = obs.split('\n')
    obs_list_stripped = [obs.strip() for obs in obs_list]

    result_index = []
    for stripped_obs in results_list:
        if stripped_obs not in obs_list_stripped:
            continue
        obs_index = obs_list_stripped.index(stripped_obs)
        result_index.append(obs_index)

    result_index.sort()
    final_list = [obs_list[i] for i in result_index]
    
    return "\n".join(final_list)

if __name__ == "__main__":
    api_url = "http://localhost:8003/filter"
    
    example_html = open('/data/webarena_acc_tree/render_41_tree_0.txt', 'r').read()
    example_objective = "Extract text"

    try:
        print("Calling API")
        results = call_filter_api(api_url, example_html, example_objective)
        print("API call successful. Results:")
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"API call failed: {e}")
