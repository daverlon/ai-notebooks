using SyntheticDatasets
using PlotlyJS
using Flux, NNlib

blobs = SyntheticDatasets.make_blobs(
    n_samples=1000,
    n_features=3,
    shuffle=true,
    random_state=30
)

X, y = Matrix(blobs[:, 1:3]), float(Array(blobs[:, 4]))
println(typeof(y))

function PlotData()
    PlotlyJS.plot(
        PlotlyJS.scatter(
            x=X[:,1], y=X[:,2], z=X[:,3], 
            type="scatter3d", 
            color=y,
            mode="markers",
            marker=attr(
                color=y,
                opacity=0.6,
            )
        )
    )
end

println(size(X), size(y))


m = 1000
train_size = 800

X_train, X_test = X[1:train_size, 1:3], X[train_size+1:m, 1:3]
y_train, y_test = y[1:train_size, 1], y[train_size+1:m, 1]

println(size(X_train), size(y_train))
println(size(X_test), size(y_test))


predict = Chain(
    Dense(3 => 3)
)
parameters = Flux.params(predict)
loss(y_pred, y_true) = Flux.Losses.logitcrossentropy(y_pred, y_true, dims=3)
optim = Flux.Descent(0.01)

function TestSampleForwardPass()
    X_sample = X_train[1, :]
    y_sample = Flux.onehotbatch(y_train[1, :], 0:2)
    println("\nSample Data: ", X_sample, "\nLabel: ", y_sample)

    pred = predict(X_sample)
    println("Prabilities: ", softmax(pred))
    
    yhat = softmax(pred)
    l = loss(yhat, y_sample)
    println(l)
end

function TestForwardPass()
    preds = predict(transpose(X_train))
    println(size(preds))

    l = loss(transpose(preds), X_train)
    println(l)
end

function GetAccuracy(y_pred, y_true)
    preds = softmax(y_pred)
    preds = argmax(preds, dims=1)
    preds = reshape(preds, (800,))
    return sum(preds .== y_true)/800
end

function TrainModel()
    epochs = 500
    losses = []
    accuracies = []

    xd = transpose(X_train)
    hot = Flux.onehotbatch(y_train, 0:2)
    

    data = [(xd, hot)]

    for epoch in 1:epochs

        Flux.train!(loss, parameters, data, optim)

        #preds = predict(transpose(X_train))

        #println("Epoch: ", epoch)
        display(parameters)
        #println("     - Loss: ", loss(preds, hot))
        #println("     - Accuracy: ", GetAccuracy(preds, y_train))

        

    end

end

#TrainModel()
#display(Flux.onehotbatch(y_train, 0:2))
unique(y)

# not learning
# need to figure out problem

