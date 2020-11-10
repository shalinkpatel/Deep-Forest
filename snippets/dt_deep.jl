using Flux, Zygote, StatsBase, Plots, SliceMap
using Flux: @epochs

mutable struct Leaf
    c
end;

mutable struct Node
    splitter
    best
    l :: Union{Leaf, Node}
    r :: Union{Leaf, Node}
end;

function flatten(tree :: Node)
    vcat(tree.splitter, flatten(tree.l), flatten(tree.r))
end;

function flatten(leaf :: Leaf)
    [leaf.c]
end;

function Node(in_channels, hidden, out_channels, depth)
    if depth == 0
        Leaf(rand(out_channels))
    else
        Node(Chain(Dense(in_channels, hidden, leakyrelu), Dense(hidden, 1)), -1, Node(in_channels, hidden, out_channels, depth - 1), Node(in_channels, hidden, out_channels, depth - 1))
    end
end;

function eval(model :: Node, x :: Vector{Float64})
    val = model.splitter(x)
    if val[1] < 0
        eval(model.l, x)
    else
        eval(model.r, x)
    end
end;

function eval(model :: Leaf, x :: Vector{Float64})
    model.c
end;

function eval(model :: Union{Node, Leaf}, x :: Array{Float64, 2})
    mapslices((batch) -> eval(model, batch), x, dims=2)
end

function get_part(model :: Node, x :: Vector{Float64})
    rep = []
    if model.splitter(x)[1] < 0
        rep = vcat(rep, [0], get_part(model.l, x))
    else
        rep = vcat(rep, [1], get_part(model.r, x))
    end
    ret = 0
    for i ∈ 0:length(rep)-1
        ret += rep[i+1] * 2^i
    end
    return ret
end;

function get_part(L :: Leaf, x :: Vector{Float64})
    []
end;

function get_part(model :: Node, x :: Array{Float64, 2})
    mapslices((batch) -> get_part(model, batch), x, dims=2)
end

model = Node(2, 10, 2, 4)
x = rand(1000, 2);
x[:, 1] .*= 2*π;
x[:, 1] .-= π;
x[:, 2] .*= 2;
x[:, 2] .-= 1;
y = Int64.(x[:, 2] .< sin.(x[:, 1]));
data = [(x, y)]

function mode(x)
    if x == []
        return 0
    else
        return StatsBase.mode(x)
    end
end

function set_best!(model :: Node, x, y)
    best = mode(y)
    model.best = best

    splt = [model.splitter(x[i, :])[1] < 0 for i ∈ 1:length(y)]
    set_best!(model.l, x[splt, :], y[splt])
    set_best!(model.r, x[.!splt, :], y[.!splt])
end

function set_best!(model :: Leaf, x, y)
    model.c = Flux.onehot(mode(y), [0, 1])
end

function loss(model :: Node, x :: Vector{Float64}, y :: Int64)
    l = Flux.logitcrossentropy(eval(model, x), Flux.onehot(model.best, [0,1]))
    if model.splitter(x)[1] < 0
        l += loss(model.l, x, y)
    else
        l += loss(model.r, x, y)
    end
    return l
end

function loss(model :: Leaf, x :: Vector{Float64}, y :: Int64)
    Flux.logitcrossentropy(model.c, Flux.onehot(y, [0,1]))
end

function loss(model :: Node, x :: Array{Float64, 2}, y :: Vector{Int64})
    l = 0
    for i ∈ 1:length(y)
        l += loss(model, x[i, :], y[i])
    end
    l / length(y)
end

function loss(x, y)
    loss(model, x, y)
end

total_acc(x, y) = mean([argmax(eval(model, x[i, :])) for i ∈ 1:size(x, 1)] == y)

set_best!(model, x, y)
loss(x, y)

function train!(loss, ps, data, opt)
    local training_loss
    ps = Params(ps)
    for d in data
        set_best!(model, d...)
        gs = gradient(ps) do
            training_loss = loss(d...)
            return training_loss
        end
        @show ps
        @show gs
        @show training_loss
        Flux.update!(opt, ps, gs)
    end
end

@epochs 10 train!(loss, params(flatten(model)), data, ADAM())

x̄ = rand(500, 2);
x̄[:, 1] .*= 2*π;
x̄[:, 1] .-= π;
x̄[:, 2] .*= 2;
x̄[:, 2] .-= 1;
ȳ = Int64.(x̄[:, 2] .< sin.(x̄[:, 1]));

plot(sin, xlims=(-π, π), ylims=(-1,1))
scatter!(x̄[:, 1], x̄[:, 2], color=[argmax(eval(model, x̄[i, :])) for i ∈ 1:size(x̄, 1)])
total_acc(x̄, ȳ)

plot(sin, xlims=(-π, π), ylims=(-1,1), legend=false)
scatter!(x̄[:, 1], x̄[:, 2], color=map((i) -> get_part(model, x̄[i, :]), 1:size(x̄, 1)))