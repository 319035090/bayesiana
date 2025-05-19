#Examen Práctico: Primer Parcial 
#Estadística Bayesiana 
#Tienda Tienda Sebastian 
#Número de cuenta: 319035090

#documentación de la función para calcular las media del parametro de localización, de precisión 
"""
    medias(xobs)
Función que calcula estimaciones puntuales de la media a posteriori (no informativas) para el parametro de localización,
el parametrode precisión y para una observación futura (predicción), de una fdp Expoencial(μ,θ) con μ ∈ R desconocido y 
θ > 0 tambien desconocido. 
```math 
f(x|μ,θ) = θexp(-θ(x - μ)),  x > μ
```

donde:

- `xobs` = vector con la muestra aleatoria observada

Entrega un tupla etiquetada con los siguientes elementos:

1. μpost = Media de la fdp a posteriori marginal no informativa de μ 
2. θpost = Media de la fdp a posteriori marginal no informativa de θ
3. pred = Media de la fpd predictiva a posteriori no informativa 

# Ejemplo
```
μ,θ = -1.5, 3.7; # valor teórico de los parámetros desconocidos
X = vaExponencial(μ,θ); #Del libro "xbExponencial.jl"
n = 1_000; # tamaño de muestra a simular
xx = X.sim(n) # simular muestra observada
med = medias(xx)
med.μpost
med.θpost 
med.pred
```
"""

#Cargamos las librerás que vamos a ocupar
using QuadGK, Plots, LaTeXStrings, CSV, DataFrames

#Cargamos los código xbExponencial y zExponencial utilizados en el curso 
include("xbExponencial.jl")

#xobs será nuestra muestr aleatoria observada proveniente de nuestro archivo CSV 
ma = CSV.read("319035090.csv", DataFrame)
n = nrow(ma)
xx = ma.xobs
post = bExpo(xx);
#Para calcular el valor de la media
function medias(xobs)
    n = length(xobs)
    sx = sum(xobs)
    xmin = minimum(xobs)
    # a posteriori marginal para μ
    dpostμ(t) = t*(((sx - n*xmin)/(sx - n*t))^n)*((n^2)/(sx - n*t))*(t < xmin)
    #Para Media 
    dmedμ, err1 = quadgk(dpostμ, -Inf, Inf)
    # a posteriori marginal de θ
    G = Gamma(n, 1 / (sx - n*xmin)) 
    dpostθ(t) = pdf(G, t)
    #dpostθ = n*(1 / (sx - n*xmin))
    dpostθ = mean(G)
    # Función de densidad predictiva a posteriori no informativa 
    function dpred(x)
        if x ≤ xmin 
            return x*(n/(n+1)) * ((sx - n*xmin)/(sx - n*x))^n * (n/(sx - n*x))
        else
            return x*(n/(n+1)) * ((sx - n*xmin)/(x + sx - (n+1)*xmin))^n * (n/(x + sx - (n+1)*xmin))
        end
    end
    medpred,b =quadgk(dpred, -Inf,Inf)
    return (μpost = dmedμ, θpost = dpostθ, pred = medpred)
end
med=medias(xx);

#a) Calcular estimaciones puntuales a posteriori (no informativas) para cada parámetro, 
#y para una observación futura (predicción), utilizando tanto la mediana como la media en cada caso.
#Estimación puntual a posteriori no informativa para la Mediana y la media de μ
medianaμ = post.qμ(0.5)
mediaμ = med.μpost
#Estimación puntual a posteriori no informativa para la Mediana y la media de θ
mediaθ = med.θpost
medianaθ = post.qθ(0.5)
#Estimacióm puntual predictiva no informativa de la media y la mediana 
medianapred = post.qp(0.5)
mediapred = med.pred

#b) Graficar la densidad a posteriori (no informativa) del parámetro de localización, 
#y agregar líneas verticales con los valores tanto de la mediana como de la media 
#(identificados claramente) e indicando ahí mismo sus valores.
post.qμ(0.5)
# densidad de μ

begin
    theme(:dark)
    t = range(post.qμ(0.001), post.qμ(0.999), length = 10000)
    plot(t, post.dμ.(t), lw = 3, label = "a posteriori no info", color = :red)
    xaxis!(L"μ")
    yaxis!("Densidad")
    plot!([post.qμ(0.5),post.qμ(0.5)], [0,500], color = :blue, label = "")
    plot!([mediaμ,mediaμ], [0,500], color = :yellow, label = "")
    scatter!([mediaμ], [0], ms = 6, color = :yellow, label = "mediaμ = $(round(mediaμ,digits=5))")
    scatter!([medianaμ], [0], ms = 6, color = :blue, label = "medianaμ = $(round(post.qμ(0.5),digits=5))")
end

#c) Graficar la densidad a posteriori (no informativa) del parámetro de precisión, 
#y agregar líneas verticales con los valores tanto de la mediana como de la media 
#(identificados claramente) e indicando ahí mismo sus valores.

begin
    theme(:dark)
    t = range(post.qθ(0.001), post.qθ(0.999), length = 10000)
    plot(t, post.dθ.(t), lw = 3, label = "a posteriori no info", color = :red)
    xaxis!(L"θ")
    yaxis!("Densidad")
    plot!([medianaθ,medianaθ], [0,.8], color = :blue, label = "")
    plot!([mediaθ,mediaθ], [0,.8], color = :yellow, label = "")
    scatter!([mediaθ], [0], ms = 6, color = :yellow, label = "mediaθ = $(round(mediaθ,digits=5))")
    scatter!([medianaθ], [0], ms = 6, color = :blue, label = "medianaθ = $(round(medianaθ,digits=5))")
end


#d) Graficar la densidad predictiva a posteriori (no informativa), y agregar líneas 
#verticales con los valores tanto de la mediana como de la media (identificados claramente)
# e indicando ahí mismo sus valores.
begin
    theme(:dark)
    t = range(post.qp(0.001), post.qp(0.999), length = 1_000)
    plot(t, post.dp.(t), lw = 3, label = "a posteriori no info", color = :red)
    xaxis!(L"θ")
    yaxis!("Densidad")
    plot!([medianapred,medianapred], [0,5], color = :blue, label = "")
    plot!([mediapred,mediapred], [0,5], color = :yellow, label = "")
    scatter!([mediapred], [0], ms = 6, color = :yellow, label = "mediapred = $(round(mediapred,digits=5))")
    scatter!([medianapred], [0], ms = 6, color = :blue, label = "medianapred = $(round(medianapred,digits=5))")
end

#e) Graficar con colores los conjuntos de nivel de la densidad conjunta a posteriori de los parámetros,
#y agregar un punto que se distinga claramente con coordenadas igual a la  mediana a posteriori de 
# cada parámetro.
sim_μθ = post.r(3_000) # simulación de conjunta (μ,θ)

post.qμ(0.5) # estimación puntual vía la mediana
median(sim_μθ[:, 1]) # estimación puntual vía simulación de la mediana 
mediaμ # estimación puntual vía la media 
mean(sim_μθ[:,1])  # estimación puntul vía simulación de la media
 
post.qθ(0.5) # estimación puntual vía la mediana
median(sim_μθ[:, 2]) # estimación puntual vía simulación de la mediana 
mediaθ  # estimación puntual vía la media 
mean(sim_μθ[:,2]) # estimación puntual vía simulación de la media 

# densidad conjunta de (μ,θ) a posteriori (3D)
begin 
    ngrid = 100
    x = collect(range(quantile(sim_μθ[:, 1], 0.03), maximum(sim_μθ[:, 1]), length = ngrid))
    y = collect(range(minimum(sim_μθ[:, 2]), maximum(sim_μθ[:, 2]), length = ngrid))
    z = zeros(ngrid, ngrid)
    for i ∈ eachindex(x), j ∈ eachindex(y)
        z[j, i] = post.d(x[i], y[j])
    end
    surface(x, y, z, xlabel = L"μ", ylabel = L"θ", size = (500, 500), 
            zlabel = L"p(μ,θ\,|\,\mathbf{x}_{obs})", title = "Densidad a Posteriori")
    scatter!([mediaμ], [mediaθ], [0.0], ms = 5, mc = :heat, label = "(μ,θ) = ($(round(mediaμ,digits=3)),$(round(mediaθ,digits=3)))")
    scatter!([medianaμ], [medianaθ], [0.0], ms = 5, mc = :plasma, label = "(M(μ),M(θ)) = ($(round(medianaμ,digits=3)),$(round(medianaθ,digits=3)))")
end

# densidad conjunta de (μ,θ) a posteriori (2D)
begin
    contour(x, y, z, xlabel = L"μ", ylabel = L"θ", fill = true, 
            title = "Densidad a Posteriori", size = (500,500))
    scatter!([mediaμ], [mediaθ], ms = 5, mc = :heat, label = "(μ,θ) = ($(round(mediaμ,digits=3)),$(round(mediaθ,digits=3)))")
    scatter!([medianaμ], [medianaθ], ms = 5, mc = :plasma, label = "(M(μ),M(θ)) = ($(round(medianaμ,digits=3)),$(round(medianaθ,digits=3)))")   
end

#f) Simular a partir de la distribución conjunta a posteriori de los parámetros una muestra
#aleatoria bivariada de tamaño 3 mil, y graficar los puntos obtenidos en un diagrama de dispersión
#(scatter plot), y agregar un punto con color distinto con coordenadas igual a la mediana a posteriori 
#de cada parámetro.
# simulación conjunta de (μ,θ) a posteriori
begin
    scatter(sim_μθ[:, 1], sim_μθ[:, 2], ms = 1, mc = :gray, size = (500,500), label = "")
    title!("Simulación Conjunta a Posteriori")
    xaxis!(L"μ")
    yaxis!(L"θ")
    scatter!([medianaμ], [medianaθ], ms = 5, mc = :cool, label = "(M(μ),M(θ)) = ($(round(medianaμ,digits=2)),$(round(medianaθ,digits=2)))")
end
