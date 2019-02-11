using Documenter, ProxSDP

makedocs(
    modules = [ProxSDP],
    doctest  = false,
    clean    = true,
    format   = Documenter.HTML(),
    sitename = "ProxSDP.jl",
    authors = "Mario Souto, Joaquim D. Garcia and contributors.",
    pages = [
        "Home" => "index.md",
        "manual.md"
    ]
)

deploydocs(
    repo = "github.com/mariohsouto/ProxSDP.jl.git",
)