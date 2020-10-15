function correlation_length(above::InfiniteMPS; tol_angle=0.1,below=above,kwargs...)
    #get the transfer spectrum
    spectrum = transfer_spectrum(above;below=above,kwargs...);

    correlation_length(spectrum,above;tol_angle=tol_angle,below=below,kwargs...)
end

function correlation_length(spectrum,above::InfiniteMPS; tol_angle=0.1,below=above,kwargs...)
    #we also define a correlation length between different states
    (above === below) && (spectrum = spectrum[2:end])

    best_angle = mod1(angle(spectrum[1]), 2*pi)
    ind_at_angle = findall(x->x<tol_angle || abs(x-2*pi)<tol_angle, mod1.(angle.(spectrum).-best_angle, 2*pi))
    spectrum_at_angle = spectrum[ind_at_angle]

    lambdas = -log.(abs.(spectrum_at_angle));

    corlength = 1/first(lambdas);

    gap = Inf;
    if length(lambdas) > 2
        gap = lambdas[2]-lambdas[1]
    end

    return corlength, gap, best_angle
end
