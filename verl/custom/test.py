import pandas as pd
import os

def read_and_print_test_scores(directory_path):
    """
    读取指定目录下的所有 .parquet 文件，合并它们，并打印 'test_score' 列。

    参数:
        directory_path (str): 包含 Parquet 文件的目录路径。
    """
    if not os.path.isdir(directory_path):
        print(f"错误: 目录不存在: {directory_path}")
        return

    parquet_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.parquet')]
    
    if not parquet_files:
        print(f"在目录 '{directory_path}' 中未找到 Parquet 文件。")
        return

    try:
        # 读取所有 parquet 文件并合并成一个 DataFrame
        df_list = [pd.read_parquet(file) for file in parquet_files]
        full_df = pd.concat(df_list, ignore_index=True)
        
        if 'test_score' in full_df.columns:
            print(f"--- 'test_score' 列的内容 ---")
            # 使用 to_string() 以确保所有行都被打印出来
            print(full_df['test_score'].to_string())
            
            # 如果您还想计算并打印所有批次的总体平均分
            if not full_df['test_score'].empty:
                # 假设 'test_score' 列中的每个元素都是一个包含 'mean_score' 的字典
                try:
                    all_mean_scores = full_df['test_score'].apply(lambda x: x.get('mean_score') if isinstance(x, dict) else None).dropna()
                    if not all_mean_scores.empty:
                        overall_average = all_mean_scores.mean()
                        print("\n--- 总体平均分 ---")
                        print(f"Overall Average Score: {overall_average}")
                except Exception as e:
                    print(f"\n计算总体平均分时出错: {e}")

        else:
            print("错误: 在合并的 DataFrame 中未找到 'test_score' 列。")
            print("可用列:", full_df.columns.tolist())

    except Exception as e:
        print(f"读取或处理 Parquet 文件时出错: {e}")

# --- 使用示例 ---
# 指定包含所有批次结果的目录
results_directory = "/home/jwangxgroup/zhwang730/LLM/Train/verl/logs/eval_Qwen2.5-Math-7B-16k-think_omni/"
read_and_print_test_scores(results_directory)