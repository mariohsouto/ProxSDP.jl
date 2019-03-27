var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "#ProxSDP-Documentation-1",
    "page": "Home",
    "title": "ProxSDP Documentation",
    "category": "section",
    "text": "ProxSDP is a semidefinite programming (SDP) solver based on the paper \"Exploiting Low-Rank Structure in Semidefinite Programming by Approximate Operator Splitting\". ProxSDP solves general SDP problems by means of a first order proximal algorithm based on the primal-dual hybrid gradient, also known as Chambolle-Pock method. The main advantage of ProxSDP over other state-of-the-art solvers is the ability of exploit the low-rank property inherent to several SDP problems."
},

{
    "location": "#Overview-of-problems-ProxSDP-can-solve-1",
    "page": "Home",
    "title": "Overview of problems ProxSDP can solve",
    "category": "section",
    "text": "Any semidefinite programming problem in standard form;\nSemidefinite relaxations of nonconvex problems, e.g. max-cut, binary MIMO, optimal power flow, sensor localization;\nNuclear norm minimization problems, e.g. matrix completion."
},

{
    "location": "#Installing-1",
    "page": "Home",
    "title": "Installing",
    "category": "section",
    "text": "Currently ProxSDP only works with Julia 1.0.xTo add ProxSDP run:pkg> add ProxSDP"
},

{
    "location": "#Referencing-1",
    "page": "Home",
    "title": "Referencing",
    "category": "section",
    "text": "The first version of the paper can be found here.@article{souto2018exploiting,\n  title={Exploiting Low-Rank Structure in Semidefinite Programming by Approximate Operator Splitting},\n  author={Souto, Mario and Garcia, Joaquim D and Veiga, {\\\'A}lvaro},\n  journal={arXiv preprint arXiv:1810.05231},\n  year={2018}\n}"
},

{
    "location": "manual/#",
    "page": "Manual",
    "title": "Manual",
    "category": "page",
    "text": ""
},

{
    "location": "manual/#Manual-1",
    "page": "Manual",
    "title": "Manual",
    "category": "section",
    "text": ""
},

{
    "location": "manual/#Building-problems-with-JuMP.jl-1",
    "page": "Manual",
    "title": "Building problems with JuMP.jl",
    "category": "section",
    "text": "Currently the easiest ways to pass problems to ProxSDP is through JuMP or MathOptInterface (v0.8).The main caveat is that currently ProxSDP must have one and only one PSD variable, no other variables are allowed.In the test folder one can find MOI implementations of some problems: MIMO, Sensor Localization, Random SDPs and sdplib problems."
},

{
    "location": "manual/#Solver-arguments-1",
    "page": "Manual",
    "title": "Solver arguments",
    "category": "section",
    "text": "Argument Description Type Values (default)\nlog_verbose print evolution of the process Bool false\nlog_freq print evoluition of the process every n iterations Int 100\ntimer_verbose Outputs a time logger Bool false\nmax_iter Maximum number of iterations Int 1000000\ntol_primal Primal error tolerance Float64 1e-3\ntol_dual Dual error tolerance Float64 1e-3\ntol_psd Tolerance associated with PSD cone Float64 1e-15\ntol_soc Tolerance associated with SOC cone Float64 1e-15\ninitial_theta Initial over relaxation parameter Float64 1.0\ninitial_beta Initial primal/dual step ratio Float64 1.0\nmin_beta Minimum primal/dual step ratio Float64 1e-4\nmax_beta Maximum primal/dual step ratio Float64 1e+4\nconvergence_window Minimum number of iterations to update target rank Int 200\nmaxlinsearchsteps Maximum number of iterations for linesearch Int 1000\nfulleigdecomp Flag for using full eigenvalue decomposition Bool false"
},

{
    "location": "manual/#JuMP-example-1",
    "page": "Manual",
    "title": "JuMP example",
    "category": "section",
    "text": "A quick JuMP example:using ProxSDP, JuMP\n\n# Create a JuMP model using ProxSDP as the solver\nmodel = Model(with_optimizer(ProxSDP.Optimizer, log_verbose=true))\n\n# Create a Positive Semidefinite variable\n# Currently ProxSDP is only able to hold one PSD\n# variable and no other variable\n@variable(model, X[1:2,1:2], PSD)\n\n# but you can define pieces of the one PSD\n# variable to hold other variable as in the\n# Canonical SDP format\nx = X[1,1]\ny = X[2,2]\n\n# There is no limits on linear constraints\n# one can define as many as wanted\n@constraint(model, ub_x, x <= 2)\n\n@constraint(model, ub_y, y <= 30)\n\n@constraint(model, con, 1x + 5y <= 3)\n\n# ProxSDP supports maximization or minimization\n# of linear functions\n@objective(model, Max, 5x + 3 * y)\n\n# Then we can solve the model\nJuMP.optimize!(model)\n\n# And ask for results!\nJuMP.objective_value(model)\n\nJuMP.value(x)\n\nJuMP.value(y)"
},

{
    "location": "manual/#Referencing-1",
    "page": "Manual",
    "title": "Referencing",
    "category": "section",
    "text": "The first version of the paper can be found here.@article{souto2018exploiting,\n  title={Exploiting Low-Rank Structure in Semidefinite Programming by Approximate Operator Splitting},\n  author={Souto, Mario and Garcia, Joaquim D and Veiga, {\\\'A}lvaro},\n  journal={arXiv preprint arXiv:1810.05231},\n  year={2018}\n}"
},

]}
