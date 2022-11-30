using Flux
using CSV, DataFrames
using LinearAlgebra
using Plots
using Random
using Flux, Flux

df = CSV.read("datasets/Salary_Data.csv", DataFrame)
df = shuffle(df)
println(df)

train_size = 20

X_train, X_test = df[1:train_size, 1], df[train_size+1:30, 1]
y_train, y_test = df[1:train_size, 2], df[train_size+1:30, 2]


predict = Dense(1 => 1)
parameters = Flux.params(predict)

loss(x, y) = Flux.Losses.mse(predict(transpose(x)), transpose(y))
optim = Descent(0.01)

n_epochs = 10000
losses = []

data = [(X_train, y_train)]

for epoch in 1:n_epochs
    append!(losses, loss(X_train, y_train))
    Flux.train!(loss, parameters, data, optim)
    println(epoch)
end

plot(losses)


#
#   julia> println(Flux.params(predict))
#   Params([Float32[9608.36;;], Float32[25417.484]])
#
#   closely matching the manual implementation I made: linear_regression_test.jl
#

