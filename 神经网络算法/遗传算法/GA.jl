using Evolutionary
using Plots
using Printf # 用于格式化输出
using Random # 引入 Random 模块，以防后续需要明确的 RNG

# 1. 定义目标函数 (Fitness Function)
function target_function_for_optimization(x::Vector{Float64})
    # x 是一个向量，即使我们的问题只有一个变量 x，它也会作为 x[1] 传入
    val = x[1] * sin(10 * π * x[1]) + 2.0
    return -val # 最小化负的适应度，等同于最大化原始适应度
end

# 2. 定义问题的搜索空间边界
lower_bound_val = -1.0
upper_bound_val = 2.0

# 使用 Evolutionary.BoxConstraints 来明确定义搜索范围
# BoxConstraints 期望两个向量作为输入，即使是单变量也需要用方括号括起来
constraints = Evolutionary.BoxConstraints([lower_bound_val], [upper_bound_val])

# 3. 配置通用优化选项 (Options)
# 移除 rng = nothing，让它使用默认的全局 RNG
options = Evolutionary.Options(
    iterations = 100,              # 迭代次数 (Generation)
    # 其他通用选项可以在这里添加，例如：
    # show_trace = true,             # 是否在优化过程中显示跟踪信息
    # store_trace = true,            # 是否存储每次迭代的跟踪信息
)

# 4. 配置遗传算法的具体参数 (GA algorithm instance)
algorithm = GA(
    populationSize = 50,           # 种群大小
    crossoverRate = 0.8,           # 交叉概率
    mutationRate = 0.1,            # 变异概率
    # 您也可以在这里指定选择、交叉和变异算子，例如：
    # selection = Evolutionary.rouletteSelection,
    # crossover = Evolutionary.blendCrossover(0.5), # 0.5是alpha参数，需要传入
    # mutation = Evolutionary.gaussianMutation(0.1), # 0.1是标准差，需要传入
)

# 5. 运行遗传算法
# optimize 函数的参数： 适应度函数， 约束， 算法实例， 算法选项
ga_result = Evolutionary.optimize(target_function_for_optimization, constraints, algorithm, options)

# 6. 获取结果 (保持不变)
best_individual = Evolutionary.minimizer(ga_result) # 最优解的变量值
min_negative_fitness = Evolutionary.minimum(ga_result) # 最小化的负适应度值
best_original_fitness = -min_negative_fitness # 转换回原始最大适应度

println("\n使用 Evolutionary.jl 找到的最佳个体：x = $(@sprintf("%.4f", best_individual[1]))")
println("对应最大适应度值：f(x) = $(@sprintf("%.4f", best_original_fitness))")

# 7. 绘制结果 (保持不变)
x_vals = range(lower_bound_val, stop=upper_bound_val, length=500)
y_vals = [x * sin(10 * π * x) + 2.0 for x in x_vals] # 直接用原始函数绘图
plot(x_vals, y_vals, label="Target Function f(x)", xlabel="x", ylabel="f(x)",
     title="Target Function and GA Result (using Evolutionary.jl)")
scatter!([best_individual[1]], [best_original_fitness], label="GA Best Solution",
         markersize=8, markercolor=:red)
savefig("Evolutionary_result.png")
println("结果图片已保存为 Evolutionary_result.png")

# 如果您想获取每次迭代的最佳适应度历史，并绘制收敛曲线，可以尝试以下代码：
# 注意：这需要 options 中设置 store_trace = true，并且 trace 的结构可能因版本而异
# 建议查阅您本地安装的 Evolutionary.jl 版本文档以获取准确的 trace 访问方法。

# options = Evolutionary.Options(
#     iterations = 100,
#     store_trace = true, # 确保存储跟踪信息
#     show_trace = false, # 可以设置为 true 在运行时显示进度
# )

# ga_result = Evolutionary.optimize(target_function_for_optimization, constraints, algorithm, options)

# try
#     # 尝试从 trace 中提取适应度历史。这可能是最容易因版本而异的部分。
#     # 一种常见的模式是 ga_result.trace 是一个包含 Evolutionary.OptimizationState 对象的向量
#     # 每个 state 包含 f_minimum (当前最佳适应度) 和 iter (迭代次数)
#     fitness_history = [state.f_minimum for state in ga_result.trace]
#     # 由于我们是最小化负适应度，这里的 f_minimum 是负值，需要取反才能表示原始函数的最大值
#     best_fitness_history_original = [-f for f in fitness_history]

#     plot(best_fitness_history_original, label="Best Fitness per Generation", xlabel="Generation", ylabel="Fitness", title="Genetic Algorithm Convergence (Evolutionary.jl)")
#     savefig("ga_evolutionary_convergence.png")
#     println("收敛曲线图已保存为 ga_evolutionary_convergence.png")
# catch e
#     println("无法绘制收敛曲线图，可能 Evolutionary.jl 版本不支持或访问 trace 方式不同。错误：", e)
# end