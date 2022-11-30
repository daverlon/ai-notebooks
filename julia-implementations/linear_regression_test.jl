using Plots
using CSV
using DataFrames
using Random
using LinearAlgebra
using Format

#Random.seed!(42)

df = CSV.read("datasets/Salary_Data.csv", DataFrame)
df = shuffle(df)
m = 20
#display(df)

X_train, X_test = df[1:20, 1], df[21:30, 1]
y_train, y_test = df[1:20, 2], df[21:30, 2]

#p = scatter(X, y) 
#display(p)

function mse(preds, targets)
    return (1/(2*m))*sum([(preds-targets)[i,1]^2 for i in 1:m])
end

function forward(X, params)
    y = (X * params[1,2])
    return [i + params[1,1] for i in y]
end

params = rand(1,2) # bias, weight
println("Params: ", params, " size: ", size(params))

function train(params, X, y)
    println("Training started.")
    losses = []
    lr = 0.01
    n_epochs = 10000
    for epoch in 1:n_epochs
        preds = forward(X, params)
        loss = mse(preds, y)
        #println("[", epoch, "] Loss: ", format(loss, commas=true))

        dd_h0 = (-1/m)*sum([(preds-y)[i,1] for i in 1:m])
        dd_h1 = (-1/m) * sum(
            transpose([(preds-y)[i, 1] for i in 1:m]) * X)

        params[1,1] += dd_h0 * lr
        params[1,2] += dd_h1 * lr
        append!(losses, loss)
    end
    println("Training finished ", format(n_epochs, commas=true), " epochs.")
    println("Argmin: ", format(argmin(losses), commas=true))
    println("Params: ", params, " size: ", size(params))
    #display(plot(losses, title="Loss over time"))
end

function eval(params, X_test, y_test)
    preds = forward(X_test, params)
    scatter(X_test, preds)
    scatter!(X_test, y_test)
    plot!(X_test, [i*params[1,2] + params[1,1] for i in X_test])
end

@time train(params, X_train, y_train)
eval(params, X_test, y_test)
