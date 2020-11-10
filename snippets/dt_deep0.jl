using Flux, Zygote, Statistics, Plots
using Flux: @epochs

struct Leaf
    pred
end

struct Node
    splitter
    l :: Union{Leaf, Node}
    r :: Union{Leaf, Node}
end

function flatten(tree :: Node)
    vcat(tree.splitter, flatten(tree.l), flatten(tree.r))
end;

function flatten(leaf :: Leaf)
    [leaf.pred]
end;

function Node(in_channels, hidden, out_channels, depth)
    if depth == 0
        Leaf(Dense(in_channels, out_channels))
    else
        Node(Chain(Dense(in_channels, hidden, leakyrelu), Dense(hidden, 1)), Node(in_channels, hidden, out_channels, depth - 1), Node(in_channels, hidden, out_channels, depth - 1))
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
    model.pred(x)
end;

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

model = Node(2, 10, 2, 4);
x = rand(1000, 2);
x[:, 1] .*= 2*π;
x[:, 1] .-= π;
x[:, 2] .*= 2;
x[:, 2] .-= 1;
y = Int64.(x[:, 2] .< sin.(x[:, 1]));

loss(x, y) = Flux.logitcrossentropy(eval(model, x), Flux.onehot(y, [0, 1]));
data = [(x[i, :], y[i]) for i ∈ 1:1000];
total_loss(x, y) = mean([loss(x[i, :], y[i]) for i ∈ 1:1000]);
total_acc(x, y) = mean([argmax(eval(model, x[i, :])) - 1 == y[i] for i ∈ 1:size(y, 1)]);

total_loss(x, y)
@epochs 250 Flux.train!(loss, params(flatten(model)), data, ADAM());
total_loss(x, y)

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