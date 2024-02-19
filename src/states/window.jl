# Note : this is intended to be a template for windowmps and windows of operators/environments

"
    Window(leftstate,window,rightstate)

    general struct of an object with a left, middle and right part.
"
struct Window{L,M,R}
    left::L
    middle::M
    right::R
end