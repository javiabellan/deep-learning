# 8. Optimization

Desafortunadamente, nuestros ordenadores tienen una capacidad de cómputo limitada.
Por lo tanto es necesario necesario usar algunos truquillos para aligerar la cosa.
En este capítulo veremos como podemos hacer el proceso de **entrenamiento más rápido**.


### Gradient Descent
```
new_w = w - (lr)(derivative)
```

### Gradient Descent with Momentum
```
             _______________grad_actual_______________
new_w = w + ( - (lr)(derivative) + (p)(grad_anterior) )
```

Momentum `p` is usually `.9`. Other common values are `.5`, `.7`, `.9` and `.99`.


> Duda
> Se podría usar linear interpolation?
>
> ```
>              __________________grad_actual_________________
> new_w = w + ( - (lr)(1-p)(derivative) + (p)(grad_anterior) )
> ```


##### Nesterov Momentum


### RMSProp

### Adam (Momentum + RMSProp)

### AdaGrad

### Parameter inizialization

### 2nd order optimization
* Método de Newtown
* Conjugate Gradients
* BFGS


### Batch Normalization

* Coordinate Descent
* Polyak Averaging
* Supervised Pretraining



## References
* http://ruder.io/deep-learning-optimization-2017/
