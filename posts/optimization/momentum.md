# Gradient Descent

```
new_w = w - (lr)(derivative)
```

# Gradient Descent with Momentum

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


# Nesterov Momentum




### References

http://ruder.io/optimizing-gradient-descent/index.html#momentum
