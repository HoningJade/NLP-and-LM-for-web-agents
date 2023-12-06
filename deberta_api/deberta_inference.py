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
    # URL of the FastAPI endpoint
    api_url = "http://localhost:8001/filter"
    
    # Example HTML content and objective
    # read 

    example_html = open('/home/zhitongg/11711-webarena/data/webarena_acc_tree/render_41_tree_0.txt', 'r').read()
    example_objective = "Extract text"

    # Call the API
    try:
        results = call_filter_api(api_url, example_html, example_objective)
        print("API call successful. Results:")
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"API call failed: {e}")