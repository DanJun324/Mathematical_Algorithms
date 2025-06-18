using MAT, Random, LinearAlgebra

println("--- 诊断脚本开始 ---")

try
    # 步骤 1: 加载数据
    data_file = "data.mat"
    if !isfile(data_file)
        println("错误: data.mat 文件不存在于当前目录!")
        return
    end
    mat_data = matread(data_file)
    input_raw = Matrix(mat_data["input"]) # 确保是 Matrix
    println("步骤 1: 原始输入已加载。维度: $(size(input_raw)), 类型: $(typeof(input_raw))")

    # 步骤 2: 判断方向并创建清晰、物化的矩阵
    local data_processed
    if size(input_raw, 1) > size(input_raw, 2)
        println("步骤 2: 检测到 (样本数, 特征数) 格式。正在转置并物化...")
        data_processed = Matrix(input_raw')
    else
        println("步骤 2: 检测到 (特征数, 样本数) 格式。无需转置。")
        data_processed = input_raw
    end
    println("步骤 2: 用于打乱的数据已备好。维度: $(size(data_processed)), 类型: $(typeof(data_processed))")
    
    # 步骤 3: 打乱列索引
    num_samples = size(data_processed, 2)
    println("步骤 3: 待打乱的样本/列数: $(num_samples)")
    shuffled_indices = shuffle(1:num_samples)
    println("步骤 3: 已成功创建随机索引。")

    # 步骤 4: 执行一直以来出问题的索引操作
    println("步骤 4: 准备执行 data_processed[:, shuffled_indices] ...")
    final_shuffled_matrix = data_processed[:, shuffled_indices]
    println("步骤 4: 成功！按列打乱操作已完成。")
    println("最终矩阵维度: $(size(final_shuffled_matrix)), 类型: $(typeof(final_shuffled_matrix))")

catch e
    println("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    println("!!!      诊断脚本失败     !!!")
    println("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    println("错误类型: $(typeof(e))")
    println("错误信息: $(e)")
    println("\n--- 请将以上包括虚线的全部输出信息发给我 ---")
    Base.show_backtrace(stdout, catch_backtrace())
end

println("\n--- 诊断脚本结束 ---")