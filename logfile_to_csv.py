import re
import pandas as pd
import matplotlib.pyplot as plt


def extract_result(log_file_name):
    results = []
    test_indices = []

    test_index_pattern = r'\[Config file\]: config_files/(\d+).json'

    result_pattern = re.compile(r'\[Result\] \((\w+)\) config_files/(\d+)\.json')

    error_pattern = re.compile(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+ - INFO - \[.*Error\] (.+)')

    with open(log_file_name, 'r') as log_file:
        lines = log_file.readlines()

    current_test_index = None

    for line in lines:
        test_index_match = re.search(test_index_pattern, line)
        if test_index_match:
            current_test_index = int(test_index_match.group(1))
            continue

        result_match = result_pattern.search(line)
        if result_match:
            result = result_match.group(1)
            results.append(result)
            test_indices.append(current_test_index)
        else:
            error_match = error_pattern.search(line)
            if error_match:
                error_message = error_match.group(1)
                results.append(f'Error: {error_message}')
                test_indices.append(current_test_index)
                
    df = pd.DataFrame({'task_id': test_indices, 'result': results})
    return df


def read_logfiles(testdf, logpaths):
    with open(logpaths, 'r') as log_paths:
        lines = log_paths.readlines()
    df_lists = []

    for logfile in lines:
        temp = extract_result(logfile.strip("\n"))
        df_lists.append(temp)
    
    df_sum = pd.concat(df_lists, ignore_index=True, axis = 0)
    
    merged_df = pd.merge(testdf, df_sum, on='task_id', how='inner')

    return df_sum, merged_df

if __name__ == "__main__":
    testdf = pd.read_json('config_files/test.json')
    baseline_results, baseline_meta = read_logfiles("/Users/guozhitong/webarena/results/log_files.txt") 
    err_analysis, err_analysis_meta = read_logfiles("/Users/guozhitong/webarena/error_analysis/log_files.txt") 

    ## merge two dataframes
    df1 = baseline_results.sort_index()
    df2 = err_analysis.sort_index()

    df1.set_index('task_id', inplace=True)
    df2.set_index('task_id', inplace=True)

    df2.sort_index(inplace=True)

    df1.update(df2)

    print(df1.result.value_counts()['PASS']/df1.shape[0])
    baseline_meta = pd.merge(testdf, df1, on='task_id', how='inner')
    baseline_meta.to_csv("baseline_metadata.csv")


    ## result visualization
    df = baseline_meta
    recall_rate = df[df['result'] == 'PASS'].groupby('sites').size() / df.groupby('sites').size()
    recall_rate = recall_rate.sort_values()

    plt.figure(figsize = (20,6))
    plt.bar(recall_rate.index, recall_rate.values)
    plt.xlabel('Sites')
    plt.ylabel('Recall Rate')
    plt.title('Recall Rate by Sites')
    plt.ylim(0, 0.2) 
    plt.show()

    