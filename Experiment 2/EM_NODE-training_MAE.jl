# Adapted from the original CRNN implementation by Ji & Deng
# (https://github.com/DENG-MIT/CRNN)

# Imports
using OrdinaryDiffEq, Flux, Optim, Random
using Zygote
using LinearAlgebra, SparseArrays, Statistics
using ProgressBars, Printf
using Flux.Losses: mse, mae
using NPZ

# Method 1: iCRNN (per-experiment collocation)
function predict_integral(X_i, p, nr, ns, J, lb, ub)
    w_in, w_b, w_out = p2vec(p, nr, ns)

    X_clamped = clamp.(X_i, lb, ub)
    log_X = log.(X_clamped)

    n_pts = size(X_i, 2)
    ones_T = ones(Float32, 1, n_pts)
    b_matrix = w_b * ones_T

    linear_term = log_X' * w_in
    R_X = exp.(linear_term .+ b_matrix')

    dX = R_X * w_out'
    dX = dX'

    X0 = X_i[:, 1:1]
    X0_all = X0 * ones_T
    X_pred = X0_all + dX * J

    return X_pred
end

function loss_integral(p, X_i, nr, ns, J, lb, ub)
    X_pred = predict_integral(X_i, p, nr, ns, J, lb, ub)
    loss = mae(X_pred, X_i)
    return loss
end

function update_integral(p, n_exp_train, ode_data_list, opt_state, nr, ns, J, lb, ub)
    train_indices = shuffle(1:n_exp_train)
    X_train = ode_data_list[train_indices, :, :]

    for i_exp in 1:n_exp_train
        X_train_i = X_train[i_exp, :, :]

        grad = Zygote.gradient(p) do x
            Zygote.forwarddiff(x) do x
                loss_integral(x, X_train_i, nr, ns, J, lb, ub)
            end
        end
        Flux.update!(opt_state, p, grad[1])
    end
end

function train_one_icrnn(ode_data_list, J, nr, ns, lb, ub, n_epoch, n_exp,
                          n_exp_train, n_exp_val, lr, beta1, beta2, lambda, seed)

    Random.seed!(seed)

    p = randn(Float32, nr * (ns + 1)) .* 1.0f-1

    opt = ADAMW(lr, (beta1, beta2), lambda)
    opt_state = Flux.setup(opt, p)

    train_loss = zeros(Float32, n_epoch)
    val_loss   = zeros(Float32, n_epoch)
    time_arr   = zeros(Float32, n_epoch)
    loss_epoch = zeros(Float32, n_exp)

    train_start = time()
    for epoch in 1:n_epoch
        update_integral(p, n_exp_train, ode_data_list, opt_state, nr, ns, J, lb, ub)

        for i_exp in 1:n_exp
            X_i = ode_data_list[i_exp, :, :]
            loss_epoch[i_exp] = loss_integral(p, X_i, nr, ns, J, lb, ub)
        end

        train_loss[epoch] = mean(loss_epoch[1:n_exp_train])
        val_loss[epoch]   = mean(loss_epoch[n_exp_train+1:end])
        time_arr[epoch]   = Float32(time() - train_start)
    end

    return p, train_loss, val_loss, time_arr
end

# Method 2: CRNN original architecture (direct ODE solver)
function crnn!(du, u, params, t)
    p, nr, ns, lb, ub = params
    w_in, w_b, w_out = p2vec(p, nr, ns)
    w_in_x = w_in' * @. log(clamp(u, lb, ub))
    du .= w_out * @. exp(w_in_x + w_b)
end

function predict_neuralode(X_i, p, tspan, tsteps, alg, nr, ns, lb, ub, abstol, reltol, maxiters)
    u0 = X_i[:, 1]
    params = (p, nr, ns, lb, ub)
    prob = ODEProblem(crnn!, u0, tspan, params; saveat=tsteps)
    pred = clamp.(Array(solve(prob, alg; dt=tsteps[2] - tsteps[1], abstol=abstol, reltol=reltol, maxiters=maxiters)), -ub, ub)
    return pred
end

function loss_neuralode(p, X_i, tspan, tsteps, alg, nr, ns, lb, ub, abstol, reltol, maxiters)
    pred = predict_neuralode(X_i, p, tspan, tsteps, alg, nr, ns, lb, ub, abstol, reltol, maxiters)
    return mae(pred, X_i)
end

function update_ode(p, n_exp_train, ode_data_list, opt_state, tspan, tsteps, alg, nr, ns, lb, ub, abstol, reltol, maxiters)
    train_indices = shuffle(1:n_exp_train)
    X_train = ode_data_list[train_indices, :, :]

    for i_exp in 1:n_exp_train
        X_train_i = X_train[i_exp, :, :]

        grad = Zygote.gradient(p) do x
            Zygote.forwarddiff(x) do x
                loss_neuralode(x, X_train_i, tspan, tsteps, alg, nr, ns, lb, ub, abstol, reltol, maxiters)
            end
        end
        Flux.update!(opt_state, p, grad[1])
    end
end

function train_one_crnn_ode(ode_data_list, tspan, tsteps, alg, nr, ns, lb, ub,
                             abstol, reltol, maxiters, n_epoch, n_exp,
                             n_exp_train, n_exp_val, lr, beta1, beta2, lambda, seed)

    Random.seed!(seed)

    p = randn(Float32, nr * (ns + 1)) .* 1.0f-1

    opt = ADAMW(lr, (beta1, beta2), lambda)
    opt_state = Flux.setup(opt, p)

    train_loss = zeros(Float32, n_epoch)
    val_loss   = zeros(Float32, n_epoch)
    time_arr   = zeros(Float32, n_epoch)
    loss_epoch = zeros(Float32, n_exp)

    train_start = time()
    for epoch in 1:n_epoch
        update_ode(p, n_exp_train, ode_data_list, opt_state, tspan, tsteps, alg,
                   nr, ns, lb, ub, abstol, reltol, maxiters)

        for i_exp in 1:n_exp
            X_i = ode_data_list[i_exp, :, :]
            loss_epoch[i_exp] = loss_neuralode(p, X_i, tspan, tsteps, alg, nr, ns, lb, ub,
                                               abstol, reltol, maxiters)
        end

        train_loss[epoch] = mean(loss_epoch[1:n_exp_train])
        val_loss[epoch]   = mean(loss_epoch[n_exp_train+1:end])
        time_arr[epoch]   = Float32(time() - train_start)
    end

    return p, train_loss, val_loss, time_arr
end

# Utilities
function p2vec(p, nr, ns; p_cutoff=0.0, b0=-10.0)
    w_b   = p[1:nr] .+ b0
    w_out = reshape(p[nr+1:end], ns, nr)

    if p_cutoff > 0
        w_out[abs.(w_out) .< p_cutoff] .= 0
    end

    w_in = max.(0, -w_out)
    return w_in, w_b, w_out
end

function display_p(p, nr, ns)
    w_in, w_b, w_out = p2vec(p, nr, ns)
    println("species (column)  reaction (row)")
    println("w_in")
    show(stdout, "text/plain", round.(w_in', digits=3))

    println("\nw_b")
    show(stdout, "text/plain", round.(exp.(w_b'), digits=3))

    println("\nw_out")
    show(stdout, "text/plain", round.(w_out', digits=3))
    println("\n\n")
end

# Parameters

n_epoch = 20000

# Optimiser parameters (AdamW)
lr     = 0.001
beta1  = 0.9
beta2  = 0.999
lambda = 1.0f-8

# Bounds to avoid predictions exploding
lb = 1.0f-5
ub = 1.0f1

# CRNN (ODE-solver) settings
alg = Tsit5();
abstol = 1e-5;
reltol = 1e-2;
maxiters = Int(1e5)

# Read training data
training_data = npzread("EM_training_data.npz")

T   = training_data["T"]
ks  = training_data["ks"]
ics = training_data["ics"]

denoised_data  = training_data["X_train"]
lownoise_data  = training_data["X_train_ln"]
highnoise_data = training_data["X_train_hn"]

J_all = training_data["J_all"]

println("T:           ", size(T))
println("ks:          ", size(ks))
println("ics:         ", size(ics))
println("denoised:     ", size(denoised_data))
println("low noise:    ", size(lownoise_data))
println("high noise:   ", size(highnoise_data))
println("J_all:        ", size(J_all))

n_ks  = size(ics, 1)
n_exp = size(ics, 2)
n_exp_train = 20
n_exp_val   = 10

model_name = "EM"
ns = 5
nr = 4
n_p = nr * (ns + 1)
species = ["A", "B", "C", "D", "E"]

noise_datasets = (denoised_data, lownoise_data, highnoise_data)
noise_names    = ["denoised", "low_noise", "high_noise"]
n_noise = length(noise_datasets)

# Result storage initialisation
icrnn_output_file       = "EM_iCRNN_training_results_MAE.npz"
crnn_output_file        = "EM_CRNN_training_results_MAE.npz"

# iCRNN
icrnn_train_loss_all = zeros(Float32, n_noise, n_ks, n_epoch)
icrnn_val_loss_all   = zeros(Float32, n_noise, n_ks, n_epoch)
icrnn_time_all       = zeros(Float32, n_noise, n_ks, n_epoch)
icrnn_p_all          = zeros(Float32, n_noise, n_ks, n_p)
icrnn_done_mask      = falses(n_noise, n_ks)

# CRNN
crnn_train_loss_all = zeros(Float32, n_noise, n_ks, n_epoch)
crnn_val_loss_all   = zeros(Float32, n_noise, n_ks, n_epoch)
crnn_time_all       = zeros(Float32, n_noise, n_ks, n_epoch)
crnn_p_all          = zeros(Float32, n_noise, n_ks, n_p)
crnn_done_mask      = falses(n_noise, n_ks)
crnn_failed_mask    = falses(n_noise, n_ks)

# Resume from disk if outputs already exist 
if isfile(icrnn_output_file)
    println("Found existing '$icrnn_output_file' -- resuming iCRNN, already-trained runs will be skipped.")
    prev = npzread(icrnn_output_file)
    icrnn_train_loss_all = prev["train_loss_all"]
    icrnn_val_loss_all   = prev["val_loss_all"]
    icrnn_time_all       = prev["time_all"]
    icrnn_p_all          = prev["p_all"]
    icrnn_done_mask      = prev["done_mask"] .> 0.5f0
end

if isfile(crnn_output_file)
    println("Found existing '$crnn_output_file' -- resuming CRNN, already-attempted runs will be skipped.")
    prev = npzread(crnn_output_file)
    crnn_train_loss_all = prev["train_loss_all"]
    crnn_val_loss_all   = prev["val_loss_all"]
    crnn_time_all       = prev["time_all"]
    crnn_p_all          = prev["p_all"]
    crnn_done_mask      = prev["done_mask"] .> 0.5f0
    crnn_failed_mask    = prev["failed_mask"] .> 0.5f0
end

function save_icrnn_progress()
    npzwrite(icrnn_output_file, Dict(
        "train_loss_all" => icrnn_train_loss_all,
        "val_loss_all"   => icrnn_val_loss_all,
        "time_all"       => icrnn_time_all,
        "p_all"          => icrnn_p_all,
        "done_mask"      => Float32.(icrnn_done_mask),
    ))
end

function save_crnn_progress()
    npzwrite(crnn_output_file, Dict(
        "train_loss_all" => crnn_train_loss_all,
        "val_loss_all"   => crnn_val_loss_all,
        "time_all"       => crnn_time_all,
        "p_all"          => crnn_p_all,
        "done_mask"      => Float32.(crnn_done_mask),
        "failed_mask"    => Float32.(crnn_failed_mask),
    ))
end

# Main interleaved loop: for each k, 
# train all noise levels (clean, low, high),
# train all approaches (iCRNN, CRNN), 
# then move to next k.

println("\nCRNN/iCRNN Recovery of " * model_name * " model.")
@printf("Noise levels: %d | k_to_train values: %d | Total (noise,k) pairs: %d\n", n_noise, n_ks, n_noise * n_ks)
@printf("Experiments: %5d | Training: %5d | Validation: %5d\n\n", n_exp, n_exp_train, n_exp_val)

total_runs = n_noise * n_ks
run_idx = 0

for k_idx in 1:n_ks
    for noise_idx in 1:n_noise
        data = noise_datasets[noise_idx]
        global run_idx += 1
        seed = 1000 * noise_idx + k_idx # a different seed for each (noise, k) pair

        ode_data_list = data[k_idx, :, :, :]
        J = J_all[k_idx, :, :]

        # Method 1: iCRNN training
        if !icrnn_done_mask[noise_idx, k_idx]
            @printf("[%4d/%4d] iCRNN       k_to_train=%3d noise=%-10s ... ", run_idx, total_runs, k_idx, noise_names[noise_idx])
            flush(stdout)

            t0 = time()
            p, train_loss, val_loss, time_arr = train_one_icrnn(
                ode_data_list, J, nr, ns, lb, ub, n_epoch, n_exp,
                n_exp_train, n_exp_val, lr, beta1, beta2, lambda, seed)
            elapsed = time() - t0

            @printf("done in %6.1fs | train: %.4e | val: %.4e\n", elapsed, train_loss[end], val_loss[end])

            icrnn_train_loss_all[noise_idx, k_idx, :] = train_loss
            icrnn_val_loss_all[noise_idx, k_idx, :]   = val_loss
            icrnn_time_all[noise_idx, k_idx, :]       = time_arr
            icrnn_p_all[noise_idx, k_idx, :]          = p
            icrnn_done_mask[noise_idx, k_idx] = true

            save_icrnn_progress()
        else
            @printf("[%4d/%4d] iCRNN       k_to_train=%3d noise=%-10s ... already done, skipping.\n", run_idx, total_runs, k_idx, noise_names[noise_idx])
        end

        # Method 2: CRNN 
        if !crnn_done_mask[noise_idx, k_idx]
            t = T[k_idx, :]
            tspan  = (t[1], t[end])
            tsteps = t

            @printf("[%4d/%4d] CRNN        k_to_train=%3d noise=%-10s ... ", run_idx, total_runs, k_idx, noise_names[noise_idx])
            flush(stdout)

            t0 = time()
            try
                p, train_loss, val_loss, time_arr = train_one_crnn_ode(
                    ode_data_list, tspan, tsteps, alg, nr, ns, lb, ub,
                    abstol, reltol, maxiters, n_epoch, n_exp,
                    n_exp_train, n_exp_val, lr, beta1, beta2, lambda, seed)
                elapsed = time() - t0

                @printf("done in %6.1fs | train: %.4e | val: %.4e\n", elapsed, train_loss[end], val_loss[end])

                crnn_train_loss_all[noise_idx, k_idx, :] = train_loss
                crnn_val_loss_all[noise_idx, k_idx, :]   = val_loss
                crnn_time_all[noise_idx, k_idx, :]       = time_arr
                crnn_p_all[noise_idx, k_idx, :]          = p
            catch e
                elapsed = time() - t0
                crnn_failed_mask[noise_idx, k_idx] = true
                @printf("FAILED after %6.1fs (%s)\n", elapsed, sprint(showerror, e))
                crnn_train_loss_all[noise_idx, k_idx, :] .= NaN32
                crnn_val_loss_all[noise_idx, k_idx, :]   .= NaN32
            end

            crnn_done_mask[noise_idx, k_idx] = true
            save_crnn_progress()
        else
            @printf("[%4d/%4d] CRNN        k_to_train=%3d noise=%-10s ... already done, skipping.\n", run_idx, total_runs, k_idx, noise_names[noise_idx])
        end
    end
end

println("\nInterleaved sweep finished. Results saved to $icrnn_output_file and $crnn_output_file")