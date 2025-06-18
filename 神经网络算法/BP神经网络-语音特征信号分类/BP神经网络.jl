using Flux          # 用于构建神经网络
using Plots         # 用于绘图
using Random        # 用于随机数生成和设置种子
using Statistics    # 用于数据归一化时的均值和标准差
using MLUtils       # 用于数据批处理和 one-hot 编码
using MAT           # 用于读取 .mat 文件
using LinearAlgebra # 用于矩阵乘法 (可选，但无害)

# 设置随机种子以确保结果可复现
Random.seed!(1234);

# --- 数据加载 ---
# 请将这些路径替换为您的实际 .mat 文件路径！
mat_file_path_data1 = "data1.mat"
mat_file_path_data2 = "data2.mat"
mat_file_path_data3 = "data3.mat"
mat_file_path_data4 = "data4.mat"

try
    # 读取 .mat 文件并提取变量
    c1 = matread(mat_file_path_data1)["c1"]
    c2 = matread(mat_file_path_data2)["c2"]
    c3 = matread(mat_file_path_data3)["c3"]
    c4 = matread(mat_file_path_data4)["c4"]

    println("读取成功")

    # --- 数据合并 ---
    data = vcat(c1[1:500,:], c2[1:500,:], c3[1:500,:], c4[1:500,:])
    println("合并成功，data维度：$(size(data))")

    # --- 数据随机排序 ---
    shuffled_indices = Random.shuffle(1:size(data, 1))
    shuffled_data = data[shuffled_indices, :]

    # --- 输入输出数据分离 ---
    input_raw = Matrix(shuffled_data[:, 2:end]')
    output_raw_labels = Vector(shuffled_data[:, 1])

    println("原始输入数据 dimensions (features x samples): $(size(input_raw))")
    println("原始输出标签 dimensions (samples): $(size(output_raw_labels))")
    
    # --- One-Hot 编码 ---
    num_classes = 4
    output_onehot = Flux.onehotbatch(output_raw_labels, 1:num_classes)
    println("One-Hot 编码后输出 dimensions (classes x samples): $(size(output_onehot))")

    # --- 划分训练集和测试集 ---
    num_train_samples = 1500
    
    input_train = input_raw[:, 1:num_train_samples]
    output_train = output_onehot[:, 1:num_train_samples]
    input_test = input_raw[:, (num_train_samples + 1):end]
    output_test_labels = output_raw_labels[(num_train_samples + 1):end]
    output_test_onehot = output_onehot[:, (num_train_samples + 1):end]

    println("\n训练集输入维度 (features x samples): $(size(input_train))")
    println("训练集输出维度 (classes x samples): $(size(output_train))")
    println("测试集输入维度 (features x samples): $(size(input_test))")
    println("测试集输出标签维度 (samples): $(size(output_test_labels))")

    # --- 输入数据归一化 ---
    input_min = minimum(input_train, dims=2)
    input_max = maximum(input_train, dims=2)

    function minmax_normalize(data_matrix, min_vals, max_vals)
        range_vals = max_vals .- min_vals
        range_vals[range_vals .== 0] .= 1.0
        return (data_matrix .- min_vals) ./ range_vals
    end

    input_train_norm = minmax_normalize(input_train, input_min, input_max)
    input_test_norm = minmax_normalize(input_test, input_min, input_max)
    
    # =========================================================================
    # --- 第二部分：网络结构初始化与训练 ---
    # =========================================================================

    innum = 24
    midnum = 25
    outnum = 4

    model = Chain(
        Dense(innum, midnum, Flux.sigmoid),
        Dense(midnum, outnum)
    )

    loss(m, x, y) = Flux.Losses.logitcrossentropy(m(x), y)

    learning_rate = 0.8
    momentum_rate = 0.9

    opt = Flux.Optimisers.Momentum(learning_rate, momentum_rate)
    optimizer_state = Flux.setup(opt, model)

    loopNumber = 5000
    train_losses = Float64[]

    println("\n开始训练神经网络...")
    for epoch in 1:loopNumber
        x_train = Float32.(input_train_norm)
        y_train = Float32.(output_train)

        val, grads = Flux.withgradient(model) do m
            loss(m, x_train, y_train)
        end
        
        Flux.update!(optimizer_state, model, grads[1])
        
        if isnan(val) || isinf(val)
            println("警告: 在 Epoch $(epoch) 损失变为 NaN 或 Inf。训练中止。")
            break
        end

        push!(train_losses, val)
        println("Epoch $(epoch)/$(loopNumber), Loss: $(val)")
    end
    println("神经网络训练完成。")

    # =========================================================================
    # --- 第三部分：网络预测与结果分析 ---
    # =========================================================================

    x_test = Float32.(input_test_norm)
    y_test_labels = output_test_labels

    predictions_raw = model(x_test)
    println("\n预测输出维度 (classes x samples): $(size(predictions_raw))")

    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★★★ 修正点：移除转置 '，确保结果是 Vector ★★★
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    predicted_labels = [argmax(predictions_raw[:, i]) for i in 1:size(predictions_raw, 2)]

    println("预测类别标签 (Vector with $(length(predicted_labels)) samples)")
    println("预测类别标签前5个样本:\n", predicted_labels[1:5])

    # 现在 error 会自动成为一个 Vector
    error = predicted_labels .- y_test_labels
    println("\n预测误差 (预测类别 - 实际类别) 前5个样本:\n", error[1:5])

    plot_data_range = 1:length(predicted_labels)

    p1 = plot(plot_data_range, predicted_labels, # 不再需要 [:]
              label="Predicted Voice Class", linecolor=:red, linewidth=1,
              xlabel="Sample Index", ylabel="Class", title="BP Network Classification")
    plot!(p1, plot_data_range, y_test_labels, # 不再需要 [:]
          label="Actual Voice Class", linecolor=:blue, linewidth=1)
    display(p1)
    savefig(p1, "bp_classification_comparison.png")
    println("分类对比图已保存为 bp_classification_comparison.png")

    p2 = plot(plot_data_range, error, # 不再需要 [:]
              label="Classification Error", linecolor=:green, linewidth=1,
              xlabel="Sample Index", ylabel="Error (Predicted - Actual)", title="BP Network Classification Error")
    display(p2)
    savefig(p2, "bp_classification_error.png")
    println("误差图已保存为 bp_classification_error.png")

    correct_predictions = sum(predicted_labels .== y_test_labels)
    total_test_samples = length(y_test_labels)
    overall_accuracy = correct_predictions / total_test_samples
    println("\n总体准确率: $(overall_accuracy * 100) %")

    k_errors_per_class = zeros(Int, num_classes)
    kk_total_per_class = zeros(Int, num_classes)

    for i in 1:total_test_samples
        actual_label = Int(y_test_labels[i])
        predicted_label = predicted_labels[i] # 已经是 Int

        kk_total_per_class[actual_label] += 1

        if actual_label != predicted_label
            k_errors_per_class[actual_label] += 1
        end
    end

    right_ratio = zeros(Float64, num_classes)
    for class_idx in 1:num_classes
        if kk_total_per_class[class_idx] > 0
            right_ratio[class_idx] = (kk_total_per_class[class_idx] - k_errors_per_class[class_idx]) / kk_total_per_class[class_idx]
        else
            right_ratio[class_idx] = NaN
        end
    end

    println("\n每类语音信号的正确率:")
    for class_idx in 1:num_classes
        if !isnan(right_ratio[class_idx])
            println("  类别 $(class_idx): $(right_ratio[class_idx] * 100) %")
        else
            println("  类别 $(class_idx): (测试集中无样本)")
        end
    end

catch e
    println("执行代码时发生错误：", e)
    Base.show_backtrace(stdout, catch_backtrace())
end