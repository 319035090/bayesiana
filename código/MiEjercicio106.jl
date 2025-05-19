#Código creado por Sebastian Tienda Tienda 
#Ejercicio 1.6) 
#Utilizando análisis conjugado, programe en Julia las distribuciones a posteriori (incluyendo las ver-
#siones no informativas) y predictivas a posteriori de los siguientes modelos de probabilidad, utilizando
#las f ́ormulas del Ap ́endice del libro de Bernardo y Smith (1994). Tambi ́en deber ́a incluir la opci ́on de
#generar simulaciones, ya sea a partir de la distribuci ́on a posteriori, o bien a partir de la distribuci ́on
#predicitva a posteriori:

using Distributions
using LaTeXStrings

#a) Binomial(m,θ) con parametro conocido m ∈ {1,2,…} pero parametro desconocido 0<θ<1
"""
    ACBinomial(; xobs = zeros(0), m = 1, α = 1.0, β = 1.0)

Distribuciones a posteriori y predictiva a posteriori para una distribución
Binomial(m, θ) donde 0 < θ < 1 es un parámetro desconocido y m ∈ {1,2,…}, con
función de masa de probabilidades:

   ```math
   f(x|θ) = \binom{m}{x} θ^x (1 - θ)^{m - x},   x = 0,1,…,m
   ```
con funcion a priori
   ```math
   π(θ|α,β) ≡ Beta(α, β)
   ```
con funcion a posterior 
   ```math
   p(θ|α,β) ≡ Beta(α + Σxᵢ, β + nm - Σxᵢ)
   ```
con funcion predictiva 
   ```math
   p(x|xobs,α,β) ≡ Beta-Binomial(m, α + Σxᵢ, β + nm - Σxᵢ)
   ```
Donde:
- `m`: Número de ensayos en la Binomial.
- `xobs` = vector con la muestra aleatoria observada (sin muestra si no se especifica)
- `α, β` = hiperparámetros de la distribución a priori Beta(α,β)

Entrega un tupla etiquetada con los siguientes elementos:
1. familia = distribución de probabilidad
2. d = función de densidad a posteriori para θ
3. p = función de distribución a posteriori para θ
4. q = función de cuantiles a posteriori para θ
5. r = función simuladora a posteriori para θ
6. dp = función de densidad predictiva a posteriori
7. pp = función de distribución predictiva a posteriori
8. qp = función de cuantiles predictiva a posteriori 
9. rp = función simuladora predictiva a posteriori 
10. n = tamaño de la muestra observada 
11. sx = suma de la muestra observada 
12. muestra = vector de la muestra observada
13. 'α, β' = valores de los hiperparámetros a priori 

Ejemplo:
θ = 0.4; # valor teórico del parámetro
X = ACBinomial();
n = 1_000; # tamaño de muestra a simular
xx = X.sim(n) # simular muestra observada
αpriori, βpriori = 2.0, 0.4 # hiperparámetros a priori
## modelos bayesianos:
prior = bExpoEsc(α = αpriori, β = βpriori);
post = bExpoEsc(xobs = xx, α = αpriori, β = βpriori);
postNoinfo = bExpoEsc(xobs = xx);
prior.familia
keys(post)
## estimación puntual de θ vía la mediana:
θ # valor teórico
prior.q(0.5) # a priori
post.q(0.5) # a posteriori 
postNoinfo.q(0.5) # a posteriori no informativa 
## estimación puntual predictiva vía la mediana: 
X.mediana # teórica 
prior.qp(0.5) # a priori 
post.qp(0.5) # a posteriori 
postNoinfo.qp(0.5) # a posteriori no informativa
```
"""
function ACBinomial(; xobs = zeros(0), m::Int = 1, α::Float64 = 1.0, β::Float64 = 1.0)
    n = length(xobs)
    ∑xᵢ = sum(xobs)  
    # Distribución a posteriori 
    post = Beta(α + ∑xᵢ, β + n * m - ∑xᵢ)
    dpost(θ) = pdf(post, θ)
    ppost(θ) = cdf(post, θ)
    qpost(u) = quantile(post, u)
    rpost(size) = rand(post, size)
    # Distribución predictiva
    pred = BetaBinomial(m, α + ∑xᵢ, β + n * m - ∑xᵢ)
    dpred(x) = pdf(pred, x)
    ppred(x) = cdf(pred, x)
    qpred(u) = quantile(pred, u)
    rpred(size) = rand(pred, size)

    return (
        familia = "Binomial con parámetro θ desconocido",
        d = dpost, p = ppost, q = qpost, r = rpost,
        dp = dpred, pp = ppred, qp = qpred, rp = rpred,
        n = n, sx = ∑xᵢ, muestra = xobs, α = α, β = β
    )
end