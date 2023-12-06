import requests
import json

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

if __name__ == "__main__":
    api_url = "http://localhost:8001/filter"
    
    example_html = open('/data/webarena_acc_tree/render_41_tree_0.txt', 'r').read()
    example_objective = "Extract text"

    try:
        print("Calling API")
        print(example_html)
        results = call_filter_api(api_url, example_html, example_objective)
        print("API call successful. Results:")
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"API call failed: {e}")
