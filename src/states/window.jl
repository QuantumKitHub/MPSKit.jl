# Note : this is intended to be a template for windowmps and windows of operators/environments but this clashes with abstractfinitemps.

"
    Window(leftstate,window,rightstate)

    general struct an object with a left, middle and right part.
"
struct Window{L,M,R}
    left::L
    middle::M
    right::R
end

# do we need copy? 
# Base.copy(win::Window) = Window(copy(win.left),copy(win.middle),copy(win.right))

# what kind of checks can we make to ensure left,middle and right fit together?