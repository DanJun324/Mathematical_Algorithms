using Plots
using Printf # 用于格式化输出
function target_function(x)
    return x * sin(10* π * x) + 2.0
end

function initialize_population(pop_size::Int, lower_bound::Real,upper_bound::Real)
    #生成在[lower_bound, upper_bound]范围内的随机初始种群
    population = lower_bound .+ (upper_bound - lower_bound) .* rand(pop_size)
    return population
end

pop_size = 100
lower_bound = 1.5
upper_bound = 2.0

population = initialize_population(pop_size,lower_bound,upper_bound)

println("初始种群前5个个体:",population[1:5])

function calculate_fitness(population::Vector{Float64})
    fitness_values = [target_function(x) for x in population]
    return fitness_values
end

fitness_values = calculate_fitness(population)
println("初始种群的适应度前5个值:", fitness_values[1:5])

#轮盘赌选择
function selection(population::Vector{Float64}, fitness_values::Vector{Float64}, num_parents::Int)
    #确保适应度为正
    min_fit = minimum(fitness_values)
    adjusted_fitness = fitness_values .- min_fit .+ 1e-6 #避免适应度为0
    
    total_fitness = sum(adjusted_fitness)
    probabilities = adjusted_fitness ./ total_fitness

    #累积概率分布
    cumulative_probabilities = cumsum(probabilities)

    selected_parents = Vector{Float64}(undef, num_parents)
    for i in 1:num_parents
        r = rand()
        idx = findfirst(x -> x >= r, cumulative_probabilities)
        selected_parents[i] = population[idx]
    end
    return selected_parents
end

num_parents = pop_size # 通常选择与种群大小相同的父代
parents = selection(population, fitness_values, num_parents)
println("选择后的父代（前5个）：", parents[1:5])

# 交叉操作
# 使用线性交叉方法生成子代
function crossover(parents::Vector{Float64}, crossover_rate::Float64, lower_bound::Real, upper_bound::Real)
    num_parents = length(parents)
    offspring = Vector{Float64}(undef, num_parents)

    for i in 1:2:num_parents-1 # 每两个父代进行交叉
        parent1 = parents[i]
        parent2 = parents[i+1]

        if rand() < crossover_rate
            alpha = rand() # 随机混合因子
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = (1 - alpha) * parent1 + alpha * parent2

            # 确保子代在有效范围内
            offspring[i] = clamp(child1, lower_bound, upper_bound)
            offspring[i+1] = clamp(child2, lower_bound, upper_bound)
        else
            offspring[i] = parent1
            offspring[i+1] = parent2
        end
    end
    # 如果种群大小是奇数，最后一个个体直接复制
    if isodd(num_parents)
        offspring[num_parents] = parents[num_parents]
    end
    return offspring
end

crossover_rate = 0.8
offspring_crossed = crossover(parents, crossover_rate, lower_bound, upper_bound)
println("交叉后的子代（前5个）：", offspring_crossed[1:5])

#变异操作
function mutation(offspring::Vector{Float64}, mutation_rate::Float64, lower_bound::Real, upper_bound::Real, mutation_strength::Real)
    mutated_offspring = deepcopy(offspring) # 创建副本，避免修改原始数组
    for i in eachindex(mutated_offspring)
        if rand() < mutation_rate
            # 添加高斯噪声或均匀随机数
            mutated_offspring[i] += mutation_strength * (rand() * 2 - 1) # 随机扰动在 [-mutation_strength, mutation_strength] 之间
            mutated_offspring[i] = clamp(mutated_offspring[i], lower_bound, upper_bound) # 确保在范围内
        end
    end
    return mutated_offspring
end

mutation_rate = 0.1
mutation_strength = 0.1 * (upper_bound - lower_bound) # 变异强度，与区间大小相关
offspring_mutated = mutation(offspring_crossed, mutation_rate, lower_bound, upper_bound, mutation_strength)
println("变异后的子代（前5个）：", offspring_mutated[1:5])

# 遗传算法主函数
function genetic_algorithm(
    target_function::Function,
    pop_size::Int,
    lower_bound::Real,
    upper_bound::Real,
    crossover_rate::Float64,
    mutation_rate::Float64,
    mutation_strength::Real,
    max_generations::Int
)
    population = initialize_population(pop_size, lower_bound, upper_bound)
    best_fitness_history = Float64[]
    best_individual_history = Float64[]

    for gen in 1:max_generations
        fitness_values = calculate_fitness(population)

        # 记录当前代的最优个体和适应度
        best_idx = argmax(fitness_values)
        current_best_individual = population[best_idx]
        current_best_fitness = fitness_values[best_idx]

        push!(best_fitness_history, current_best_fitness)
        push!(best_individual_history, current_best_individual)

        # 打印进度
        if gen % 10 == 0 || gen == 1 || gen == max_generations
            println("Generation $gen: Best Fitness = $(@sprintf("%.4f", current_best_fitness)), Best Individual = $(@sprintf("%.4f", current_best_individual))")
        end

        # 选择
        parents = selection(population, fitness_values, pop_size)

        # 交叉
        offspring_crossed = crossover(parents, crossover_rate, lower_bound, upper_bound)

        # 变异
        offspring_mutated = mutation(offspring_crossed, mutation_rate, lower_bound, upper_bound, mutation_strength)

        # 更新种群
        population = offspring_mutated
    end

    # 最终结果
    fitness_values = calculate_fitness(population)
    best_idx = argmax(fitness_values)
    global_best_individual = population[best_idx]
    global_best_fitness = fitness_values[best_idx]

    return global_best_individual, global_best_fitness, best_fitness_history, best_individual_history
end


# 运行遗传算法
max_generations = 500   #设置最大迭代次数
global_best_individual, global_best_fitness, best_fitness_history, best_individual_history = genetic_algorithm(
    target_function,
    pop_size,
    lower_bound,
    upper_bound,
    crossover_rate,
    mutation_rate,
    mutation_strength,
    max_generations
)

println("\n遗传算法找到的最佳个体：x = $(@sprintf("%.4f", global_best_individual))")
println("对应最大适应度值：f(x) = $(@sprintf("%.4f", global_best_fitness))")

# 绘制适应度历史曲线
plot(best_fitness_history, label="Best Fitness per Generation", xlabel="Generation", ylabel="Fitness", title="Genetic Algorithm Convergence")
savefig("适应度历史曲线.png") # 保存图片

# 绘制函数曲线和找到的最优解
x_vals = range(lower_bound, stop=upper_bound, length=500)
y_vals = [target_function(x) for x in x_vals]
plot(x_vals, y_vals, label="Target Function f(x)", xlabel="x", ylabel="f(x)", title="Target Function and GA Result")
scatter!([global_best_individual], [global_best_fitness], label="GA Best Solution", markersize=8, markercolor=:red)
savefig("函数曲线VS最优解.png") # 保存图片