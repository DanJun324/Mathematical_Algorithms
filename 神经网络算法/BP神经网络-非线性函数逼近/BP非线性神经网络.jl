using Flux
using Plots
using Random
using Statistics
using MAT
using LinearAlgebra
plotlyjs()

# 设置随机种子以确保结果可复现
Random.seed!(1234)

try
    # =========================================================================
    # --- 第1部分：数据加载与预处理 ---
    # =========================================================================
    println("正在从 data.mat 加载数据...")
    data_file = "data.mat"
    mat_data = matread(data_file)
    
    input_raw = Matrix(mat_data["input"])
    output_raw = Matrix(mat_data["output"])

    local input_processed, output_processed
    if size(input_raw, 1) > size(input_raw, 2)
        input_processed = Matrix(input_raw')
    else
        input_processed = input_raw
    end
    if size(output_raw, 2) != size(input_processed, 2)
        output_processed = Matrix(output_raw')
    else
        output_processed = output_raw
    end
    
    num_samples = size(input_processed, 2)
    shuffled_column_indices = shuffle(1:num_samples)
    input_shuffled = input_processed[:, shuffled_column_indices]
    output_shuffled = output_processed[:, shuffled_column_indices]
    num_train_samples = 1900
    input_train = input_shuffled[:, 1:num_train_samples]
    output_train = output_shuffled[:, 1:num_train_samples]
    input_test = input_shuffled[:, (num_train_samples + 1):end]
    output_test = output_shuffled[:, (num_train_samples + 1):end]
    
    input_min = minimum(input_train, dims=2); input_max = maximum(input_train, dims=2)
    output_min = minimum(output_train, dims=2); output_max = maximum(output_train, dims=2)

    function minmax_normalize(data, min_vals, max_vals)
        range_vals = max_vals .- min_vals; range_vals[range_vals .== 0] .= 1.0
        return (data .- min_vals) ./ range_vals
    end
    function minmax_denormalize(normalized_data, min_vals, max_vals)
        range_vals = max_vals .- min_vals
        return normalized_data .* range_vals .+ min_vals
    end

    input_train_norm = minmax_normalize(input_train, input_min, input_max)
    output_train_norm = minmax_normalize(output_train, output_min, output_max)
    input_test_norm = minmax_normalize(input_test, input_min, input_max)

    # =========================================================================
    # --- 第2部分：BP网络训练 (采用现代Flux API) ---
    # =========================================================================

    # --- 2.1 优化网络结构与训练参数 ---
    innum = size(input_train_norm, 1)
    midnum1 = 10
    midnum2 = 10
    outnum = size(output_train_norm, 1)

    model = Chain(
        Dense(innum, midnum1, tanh),
        Dense(midnum1, midnum2, tanh),
        Dense(midnum2, outnum)
    )

    loss(m, x, y) = Flux.Losses.mse(m(x), y)
    
    learning_rate = 0.005
    epochs = 5000
    batch_size = 32
    
    # --- 2.2 【核心API修正】采用新的 Flux/Optimisers.jl 训练流程 ---

    # 1. 初始化优化器状态，它将同时包含优化器信息和模型参数
    optimizer_state = Flux.setup(Flux.Optimisers.Adam(learning_rate), model)

    # 2. 创建 DataLoader 实现小批量训练
    train_data_loader = Flux.DataLoader(
        (Float32.(input_train_norm), Float32.(output_train_norm)),
        batchsize=batch_size,
        shuffle=true
    )

    println("\n开始优化训练神经网络...")
    for epoch in 1:epochs
        for (x_batch, y_batch) in train_data_loader
            # 3. 计算梯度，直接作用于模型。返回的 grads 是一个元组(Tuple)。
            grads = Flux.gradient(m -> loss(m, x_batch, y_batch), model)
            
            # 4. 使用 optimizer_state 更新模型参数。注意要传入 grads[1]。
            Flux.update!(optimizer_state, model, grads[1])
        end
        
        # 每个 epoch 结束后，计算在整个训练集上的损失以供参考
        current_loss = loss(model, Float32.(input_train_norm), Float32.(output_train_norm))
        println("Epoch $(epoch)/$(epochs), Loss: $(current_loss)")
    end
    println("神经网络训练完成。")

    # =========================================================================
    # --- 第3部分：BP网络预测与结果分析 ---
    # =========================================================================
    x_test_flux = Float32.(input_test_norm); predictions_norm = model(x_test_flux)
    BPoutput = minmax_denormalize(predictions_norm, output_min, output_max)
    BPoutput_flat = vec(BPoutput); output_test_flat = vec(output_test)
    println("\n反归一化后的预测输出 (前5个样本):\n", BPoutput_flat[1:5])
    println("真实的测试集输出 (前5个样本):\n", output_test_flat[1:5])
    
    p1 = plot(1:length(output_test_flat), [BPoutput_flat output_test_flat], 
              title="BP神经网络函数逼近效果 ", label=["网络预测输出" "真实输出"], 
              lw=1.5, xlabel="测试样本序号", ylabel="函数输出值")
    display(p1); savefig(p1, "bp_comparison_optimized_cn.png")
    
    p2 = plot(1:length(output_test_flat), BPoutput_flat .- output_test_flat, 
              title="BP神经网络预测误差 ", label="预测误差", 
              c=:red, lw=1.5, xlabel="测试样本序号", ylabel="误差")
    display(p2); savefig(p2, "bp_error_optimized_cn.png")
    
    total_abs_error = sum(abs.(BPoutput_flat .- output_test_flat))
    println("\n总绝对误差和: $(total_abs_error)")

catch e
    println("\n执行代码时发生严重错误：", e)
    Base.show_backtrace(stdout, catch_backtrace())
end