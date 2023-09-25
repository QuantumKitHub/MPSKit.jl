# Note : this is intended to be a template for windowmps and windows of operators/environments but this clashes with abstractfinitemps.

"
    Window(leftstate,window,rightstate)

    general struct of an object with a left, middle and right part.
"
struct Window{L,M,R}
    left::L
    middle::M
    right::R
end

# do we need copy? 
# Base.copy(win::Window) = Window(copy(win.left),copy(win.middle),copy(win.right))
