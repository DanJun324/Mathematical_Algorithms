using Flux
using Plots
using Random
using Statistics
using Evolutionary
using Optimisers
using LinearAlgebra

# 将所有逻辑包装在 main 函数中
function main()
    # 设置随机种子以确保结果可复现
    Random.seed!(1234)
    plotlyjs()

    # =========================================================================
    # --- 第1部分：数据生成与预处理 (无变化) ---
    # =========================================================================
    println("正在生成模拟数据 (y = x1^2 + x2^2)...")
    num_total_samples = 2000
    input_raw_data = rand(Float32, 2, num_total_samples) .* 10.0f0 .- 5.0f0
    output_raw_data = mapslices(x -> x[1]^2 + x[2]^2, input_raw_data, dims=1)
    
    shuffled_indices = shuffle(1:num_total_samples)
    input_shuffled = input_raw_data[:, shuffled_indices]
    output_shuffled = output_raw_data[:, shuffled_indices]

    num_train_samples = 1900
    input_train = input_shuffled[:, 1:num_train_samples]
    output_train = output_shuffled[:, 1:num_train_samples]
    input_test = input_shuffled[:, (num_train_samples + 1):end]
    output_test = output_shuffled[:, (num_train_samples + 1):end]

    input_min, input_max = minimum(input_train, dims=2), maximum(input_train, dims=2)
    output_min, output_max = minimum(output_train, dims=2), maximum(output_train, dims=2)

    function minmax_normalize(data, min_vals, max_vals)
        range_vals = max_vals .- min_vals; range_vals[range_vals .== 0] .= 1.0f0
        return (data .- min_vals) ./ range_vals
    end
    function minmax_denormalize(normalized_data, min_vals, max_vals)
        range_vals = max_vals .- min_vals
        return normalized_data .* range_vals .+ min_vals
    end

    input_train_norm = minmax_normalize(input_train, input_min, input_max)
    output_train_norm = minmax_normalize(output_train, output_min, output_max)
    input_test_norm = minmax_normalize(input_test, input_min, input_max)
    println("\n数据预处理完成。")

    # =========================================================================
    # --- 第2部分：GA 全局搜索最优初始权重 ---
    # =========================================================================
    innum = size(input_train_norm, 1); hiddennum = 9; outnum = size(output_train_norm, 1)
    create_bp_model(in_dim, hid_dim, out_dim) = Chain(Dense(in_dim, hid_dim, tanh), Dense(hid_dim, out_dim))
    
    reference_model = create_bp_model(innum, hiddennum, outnum)
    p_init, re = Flux.destructure(reference_model)
    numsum = length(p_init)
    println("BP网络参数总数 (遗传算法染色体长度): $(numsum)")

    # 【核心修正】适应度函数：只计算初始权重下的损失，不进行训练
    function ga_fitness_function(params_vector, re_fn, x_train, y_train)
        model_eval = re_fn(Float32.(params_vector))
        loss = Flux.mse(model_eval(x_train), y_train)
        return Float64(loss)
    end

    # GA 参数 (增加迭代和种群以进行更充分的搜索)
    maxgen = 500; sizepop = 60; pcross = 0.9; pmutation = 0.1
    lower_bound = -3.0; upper_bound = 3.0
    
    println("\n阶段一：开始运行遗传算法进行全局搜索...")
    ga_options = Evolutionary.Options(iterations=maxgen, show_trace=true, show_every=10)
    constraints = Evolutionary.BoxConstraints(fill(lower_bound, numsum), fill(upper_bound, numsum))
    
    ga_algorithm = GA(
        populationSize = sizepop,
        selection = roulette,
        crossover = TPX,
        mutation  = gaussian(0.1)
    )

    fitness_func_ga = x -> ga_fitness_function(x, re, input_train_norm, output_train_norm)
    
    ga_result = Evolutionary.optimize(fitness_func_ga, constraints, ga_algorithm, ga_options)
    best_initial_params = Evolutionary.minimizer(ga_result)
    println("\n阶段一完成。遗传算法找到的最佳初始权重损失: $(Evolutionary.minimum(ga_result))")

    # =========================================================================
    # --- 第3部分：BP 局部精调最优权重 ---
    # =========================================================================
    println("\n阶段二：使用GA找到的最优参数进行BP网络局部精调...")
    
    # 使用GA找到的最佳参数来构建最终模型
    final_model = re(Float32.(best_initial_params))
    
    # 定义BP网络的训练参数
    bp_epochs = 6000
    bp_lr = 0.005
    bp_batch_size = 32

    bp_optimizer_state = Flux.setup(Optimisers.Adam(bp_lr), final_model)
    train_loader = Flux.DataLoader((input_train_norm, output_train_norm), batchsize=bp_batch_size, shuffle=true)
    
    loss_bp(m,x,y) = Flux.mse(m(x),y)

    for epoch in 1:bp_epochs
        for (x_batch, y_batch) in train_loader
            grads = Flux.gradient(m -> loss_bp(m, x_batch, y_batch), final_model)
            Flux.update!(bp_optimizer_state, final_model, grads[1])
        end
        if epoch % 50 == 0
            current_loss = loss_bp(final_model, input_train_norm, output_train_norm)
            println("BP训练 - Epoch $(epoch), 损失: $(current_loss)")
        end
    end
    println("阶段二完成。")

    # =========================================================================
    # --- 第4部分：结果分析 ---
    # =========================================================================
    predictions_norm = final_model(input_test_norm)
    BPoutput = minmax_denormalize(predictions_norm, output_min, output_max)
    BPoutput_flat, output_test_flat = vec(BPoutput), vec(output_test)

    println("\n反归一化后的预测输出 (前5个样本):\n", BPoutput_flat[1:5])
    println("真实的测试集输出 (前5个样本):\n", output_test_flat[1:5])

    p1 = plot(1:length(output_test_flat), [BPoutput_flat output_test_flat],
              title="GA-BP Optimized Network Approximation", label=["Predicted" "Actual"],
              lw=1.5, xlabel="Sample Index", ylabel="Function Output")
    display(p1); savefig(p1, "ga_bp_final_comparison.png")

    error = BPoutput_flat .- output_test_flat
    p2 = plot(1:length(output_test_flat), error, title="Prediction Error", label="Error", c=:red, lw=1.5)
    display(p2); savefig(p2, "ga_bp_final_error.png")

    println("\n总绝对误差和: $(sum(abs.(error)))")
end

# 运行主函数
try
    main()
catch e
    println("\n执行代码时发生严重错误：")
    showerror(stdout, e)
    println("\n")
    Base.show_backtrace(stdout, catch_backtrace())
end